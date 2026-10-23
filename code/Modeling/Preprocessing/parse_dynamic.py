from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""
Parse data/raw_dynamic/page_*.html (website listing pages, 10 records each) into tidy tables:

  data/tidy/dynamic_records.parquet    one row per record
  data/tidy/dynamic_logcs.parquet      (record_id, time_h, logc)
  data/tidy/dynamic_profiles.parquet   (record_id, variable in {Temperatures, Phs, Aws}, time_h, value)
"""
import glob
import json
import os
import re
import pandas as pd
from lxml import html

HERE = str(DATA/'Processed/Preparation')
RAW = str(DATA/'Raw/dynamic_records_html')
OUT = os.path.join(HERE, "data", "tidy")


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def parse_num(s):
    s = clean(s)
    m = re.match(r"^(-?\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, logcs, profiles = [], [], []
    files = sorted(glob.glob(os.path.join(RAW, "page_*.html")))
    for f in files:
        page = int(re.search(r"page_(\d+)", f).group(1))
        doc = html.fromstring(open(f, "rb").read())
        for chk in doc.xpath("//input[contains(@class,'exportchk')]"):
            rid = int(chk.get("data-recordid"))
            node = chk
            for _ in range(10):
                node = node.getparent()
                if node is None or node.xpath(".//input[contains(@id,'HiddenLogcs')]"):
                    break
            if node is None:
                continue
            def lbl(cls):
                v = node.xpath(f".//span[@id='{cls}' or contains(@id,'_{cls}_')]")
                return clean(v[0].text_content()) if v else None
            def hid(name):
                v = node.xpath(f".//input[contains(@id,'{name}_')]/@value")
                return v[0] if v else ""
            title = node.xpath(".//a[contains(@id,'LinkBtnGoToDetails')]")
            row = {
                "record_id": rid, "page": page,
                "title": clean(title[0].text_content()) if title else None,
                "organism": lbl("lblOrg"),
                "food_category": lbl("lblMatrix"),
                "food_name": lbl("lblFoodName"),
                "temperature_text": lbl("lblTemp"), "aw_text": lbl("lblAw"), "ph_text": lbl("lblPh"),
                "conditions_text": lbl("lblCond"),
                "source": lbl("lblSource"),
                "n_visits": parse_num(lbl("lblNbOfVisits")), "n_downloads": parse_num(lbl("lblNbDownloads")),
                "date_added": lbl("lblDateAdded"),
                "logc0_text": hid("HiddenLogc0"),
            }
            row["temperature"] = parse_num(row["temperature_text"] or "")
            row["aw"] = parse_num(row["aw_text"] or "")
            row["ph"] = parse_num(row["ph_text"] or "")
            for name, var in (("HiddenLogcs", None), ("HiddenTemps", "Temperatures"), ("HiddenPhs", "Phs"), ("HiddenAws", "Aws")):
                raw = hid(name)
                pts = json.loads(raw) if raw else []
                if var is None:
                    row["n_logc_points"] = len(pts)
                    for t, v in pts:
                        logcs.append({"record_id": rid, "time_h": float(t), "logc": float(v)})
                else:
                    row["n_" + var.lower() + "_points"] = len(pts)
                    for t, v in pts:
                        profiles.append({"record_id": rid, "variable": var, "time_h": float(t), "value": float(v)})
            recs.append(row)
    df = pd.DataFrame(recs).drop_duplicates("record_id")
    df.to_parquet(os.path.join(OUT, "dynamic_records.parquet"), index=False)
    df.to_csv(os.path.join(OUT, "dynamic_records.csv"), index=False)
    pd.DataFrame(logcs).drop_duplicates().to_parquet(os.path.join(OUT, "dynamic_logcs.parquet"), index=False)
    pd.DataFrame(profiles).drop_duplicates().to_parquet(os.path.join(OUT, "dynamic_profiles.parquet"), index=False)
    print(f"files={len(files)} records={len(df)} logc_points={len(logcs)} profile_points={len(profiles)}")
    print(df[["n_temperatures_points", "n_phs_points", "n_aws_points"]].gt(0).sum())


if __name__ == "__main__":
    main()
