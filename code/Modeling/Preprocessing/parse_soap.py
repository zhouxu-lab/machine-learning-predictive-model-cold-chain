from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""
Parse data/raw_soap/page_*.xml (SOAP GetResponses pages) into tidy tables:

  data/tidy/static_records.parquet     one row per response (record)
  data/tidy/static_logcs.parquet       one row per (record, time) log-count point
  data/tidy/static_conditions.parquet  one row per (record, condition)
  data/tidy/static_properties.parquet  one row per (record, property)
"""
import glob
import os
import pandas as pd
from lxml import etree

HERE = str(DATA/'Processed/Preparation')
RAW = str(DATA/'Raw/isothermal_records_soap_xml')
OUT = os.path.join(HERE, "data", "tidy")
NS = "{http://localhost:49573}"
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"


def txt(el, tag, cast=None):
    e = el.find(NS + tag)
    if e is None or e.get(XSI_NIL) == "true" or e.text is None:
        return None
    return cast(e.text) if cast else e.text


def points(el, tag):
    p = el.find(NS + tag)
    if p is None:
        return []
    return [(float(t.findtext(NS + "Time")), float(t.findtext(NS + "Value"))) for t in p.findall(NS + "TimeValuePoint")]


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, logcs, conds, props, profiles = [], [], [], [], []
    files = sorted(glob.glob(os.path.join(RAW, "page_*.xml")))
    for f in files:
        root = etree.parse(f, etree.XMLParser(recover=True, huge_tree=True)).getroot()
        for r in root.iter(NS + "Response"):
            rid = int(r.findtext(NS + "ID"))
            ex = r.find(NS + "Experiment")
            env = ex.find(NS + "Environment") if ex is not None else None
            src = ex.find(NS + "Source") if ex is not None else None
            row = {
                "record_id": rid,
                "combase_id": txt(r, "ComBaseID"),
                "experiment_id": int(ex.findtext(NS + "ID")) if ex is not None else None,
                "organism": txt(r, "OrganismDefinition"),
                "organism_id": txt(r, "OrganismID"),
                "specification": txt(r, "Specification"),
                "food_category": txt(ex, "MatrixDefinition") if ex is not None else None,
                "food_category_id": txt(ex, "MatrixID") if ex is not None else None,
                "food_name": txt(ex, "InOn") if ex is not None else None,
                "is_static": txt(ex, "IsStatic") if ex is not None else None,
                "temperature": txt(env, "Temperature", float) if env is not None else None,
                "ph": txt(env, "Ph", float) if env is not None else None,
                "aw": txt(env, "Aw", float) if env is not None else None,
                "conditions_text": txt(env, "DisplayableConditions") if env is not None else None,
                "properties_text": txt(r, "DisplayableProperties"),
                "specific_rate": txt(r, "SpecificRate", float),
                "logc0": txt(r, "Logc0", float),
                "assumed": txt(r, "Assumed"),
                "comment": txt(r, "Comment"),
                "measure_method": txt(r, "MeasureMethod"),
                "rate_calc_method": txt(r, "RateCalculationMethod"),
                "published": txt(r, "Published", int),
                "n_visits": txt(r, "NumberOfVisits", int),
                "n_downloads": txt(r, "NumberOfDownloads", int),
                "date_added": txt(r, "DateAdded"),
                "source_id": int(src.findtext(NS + "ID")) if src is not None else None,
                "source": txt(src, "Definition") if src is not None else None,
                "source_author": txt(src, "Author") if src is not None else None,
                "source_url": txt(src, "URL") if src is not None else None,
            }
            pts = points(r, "Logcs")
            row["n_logc_points"] = len(pts)
            recs.append(row)
            for t, v in pts:
                logcs.append({"record_id": rid, "time_h": t, "logc": v})
            if env is not None:
                for c in env.iter(NS + "Condition"):
                    conds.append({"record_id": rid, "condition_id": txt(c, "ID"), "condition": txt(c, "Definition"),
                                  "value": txt(c, "Value", float), "units": txt(c, "Units")})
                for kind in ("Temperatures", "Phs", "Aws"):
                    for t, v in points(env, kind):
                        profiles.append({"record_id": rid, "variable": kind, "time_h": t, "value": v})
            for p in r.iter(NS + "Property"):
                props.append({"record_id": rid, "property_id": txt(p, "ID"), "property": txt(p, "Definition"),
                              "value": txt(p, "Value", float), "units": txt(p, "Units")})
    df = pd.DataFrame(recs).drop_duplicates("record_id")
    df.to_parquet(os.path.join(OUT, "static_records.parquet"), index=False)
    pd.DataFrame(logcs).to_parquet(os.path.join(OUT, "static_logcs.parquet"), index=False)
    pd.DataFrame(conds).to_parquet(os.path.join(OUT, "static_conditions.parquet"), index=False)
    pd.DataFrame(props).to_parquet(os.path.join(OUT, "static_properties.parquet"), index=False)
    pd.DataFrame(profiles).to_parquet(os.path.join(OUT, "static_profiles.parquet"), index=False)
    df.to_csv(os.path.join(OUT, "static_records.csv"), index=False)
    print(f"files={len(files)} records={len(df)} logc_points={len(logcs)} conditions={len(conds)} properties={len(props)} profile_points={len(profiles)}")


if __name__ == "__main__":
    main()
