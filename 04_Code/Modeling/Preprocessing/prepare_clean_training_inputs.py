from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Prepare conservative censor-safe inputs without changing the frozen experiment.

ComBase documents -0.01 as a Not Detected code. Mixed quantified/censored
replicates are excluded by record because the archived means are already
contaminated; exact censored targets are removed and censored-origin records
are excluded. A future model could use detection-limit-aware likelihoods.
"""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

ROOT = PACKAGE_ROOT
BASE = DATA/'Processed/Preparation'
OUT = DATA/'Processed/Training'
ND = -0.01

def is_nd(a, reconstructed=False):
    # obs_y + y0 is reconstructed from float32 neural arrays and can differ
    # from the exact source code by ~1e-7 after subtraction/addition.
    return np.isclose(np.asarray(a, dtype=float), ND, rtol=0,
                      atol=1e-5 if reconstructed else 1e-8)

def main():
    raw = pd.read_parquet(BASE / "data/raw_tables/counts.parquet", columns=["record_id", "time_h", "logc"])
    raw["nd"] = is_nd(raw.logc)
    mixture = raw.groupby(["record_id", "time_h"]).nd.agg(["sum", "count"])
    mixed_ids = set(mixture.index.get_level_values(0)[(mixture["sum"] > 0) & (mixture["sum"] < mixture["count"])])

    features = pd.read_parquet(BASE / "features.parquet")
    assert {"record_id", "time_h", "logc", "logc0_obs"} <= set(features.columns)
    unsafe = is_nd(features.logc) | is_nd(features.logc0_obs) | features.record_id.isin(mixed_ids)
    clean = features.loc[~unsafe].copy()
    assert not is_nd(clean.logc).any() and not is_nd(clean.logc0_obs).any()
    assert clean.groupby("record_id").logc0_obs.nunique().eq(1).all()
    clean.to_parquet(OUT / "features.parquet", index=False)

    with (BASE / "deep_prepared.pkl").open("rb") as f:
        recs, orgs, mats = pickle.load(f)
    clean_recs = []
    removed_origin = removed_mixed = removed_targets = 0
    for rec in recs:
        if rec["record_id"] in mixed_ids:
            removed_mixed += 1
            continue
        if is_nd([rec["y0"]])[0]:
            removed_origin += 1
            continue
        y = np.asarray(rec["obs_y"], float) + float(rec["y0"])
        keep = ~is_nd(y, reconstructed=True)
        removed_targets += int((~keep).sum())
        if not keep.any():
            continue
        updated = rec.copy()
        for key in ("obs_idx", "obs_y", "obs_time"):
            updated[key] = np.asarray(rec[key])[keep]
        clean_recs.append(updated)
    assert all(not is_nd(np.asarray(r["obs_y"]) + r["y0"], reconstructed=True).any() for r in clean_recs)
    assert {int(r["record_id"]) for r in clean_recs} == set(map(int, clean.record_id.unique())), "Tree/neural record mismatch"
    with (OUT / "deep_prepared.pkl").open("wb") as f:
        pickle.dump((clean_recs, orgs, mats), f, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "raw_censored_count_entries": int(raw.nd.sum()),
        "mixed_censored_quantified_record_times": int(((mixture["sum"] > 0) & (mixture["sum"] < mixture["count"])).sum()),
        "mixed_record_ids": len(mixed_ids),
        "original_feature_rows": len(features),
        "clean_feature_rows": len(clean),
        "removed_feature_rows": int(unsafe.sum()),
        "original_deep_records": len(recs),
        "clean_deep_records": len(clean_recs),
        "removed_deep_censored_origin_records": removed_origin,
        "removed_deep_mixed_records": removed_mixed,
        "removed_deep_censored_later_targets": removed_targets,
        "official_code_reference": "https://combase.ars.usda.gov/DMFit_Help.aspx",
    }
    for name in ("features.parquet", "deep_prepared.pkl"):
        report[name + "_sha256"] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    (OUT / "clean_input_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
