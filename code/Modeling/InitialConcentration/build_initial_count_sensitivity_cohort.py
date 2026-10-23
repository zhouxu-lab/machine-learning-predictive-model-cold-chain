from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Freeze food records with a non-detected first count and later quantified counts.

These records remain outside clean-model training. Source-held-out model weights
can later be queried at four hypothetical initial counts on identical targets.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = PACKAGE_ROOT
OUT = RESULTS/'Predictions/Dynamic'
CAT = DATA/'Processed/record_catalog.csv'
FEATURES = DATA/'Processed/Preparation/features.parquet'
SPLITS = DATA/'Splits_and_Configurations/Primary/splits.json'

def main():
    c = pd.read_csv(CAT)
    f = pd.read_parquet(FEATURES)
    s = json.loads(SPLITS.read_text(encoding="utf-8"))
    fold_of_id = {int(rid): int(k) for k, fold in s.items() for rid in fold["test"]}
    eligible = c[c.qc_pass & c.food_record & c.profile_complete_1h &
                 c.first_is_nondetect & c.first_quant_h.notna() &
                 (c.count_points >= 3)].copy()
    raw = pd.read_parquet(DATA/'Processed/Preparation/data/raw_tables/counts.parquet')
    coded = np.isclose(raw.logc, -.01, rtol=0, atol=1e-8)
    mix = raw.assign(coded=coded).groupby(["record_id", "time_h"]).coded.agg(["sum", "count"])
    mixed_ids = set(mix.index.get_level_values(0)[(mix["sum"] > 0) & (mix["sum"] < mix["count"])])
    eligible = eligible[~eligible.record_id.isin(mixed_ids)]
    points = f[f.record_id.isin(eligible.record_id) &
               ~np.isclose(f.logc, -.01, rtol=0, atol=1e-8)].copy()
    assert points.logc0_obs.apply(lambda x: np.isclose(x, -.01, rtol=0, atol=1e-8)).all()
    points = points.merge(eligible[["record_id", "food_name", "food_category", "organism",
                                    "source_key", "refrigerated_exposure", "temp_min", "temp_max"]],
                          on="record_id", how="left", suffixes=("", "_catalog"), validate="many_to_one")
    assert points.source_key.eq(points.source_key_catalog).all()
    points = points.drop(columns="source_key_catalog")
    points["fold"] = points.record_id.map(fold_of_id)
    assert points.fold.notna().all()
    points = points.rename(columns={"logc": "observed_log", "refrigerated_exposure": "low_temperature_exposure"})
    display = ["record_id", "time_h", "observed_log", "source_key", "fold", "food_name",
               "food_category", "organism", "low_temperature_exposure", "temp_min", "temp_max"]
    points[display].to_csv(OUT / "initial_count_sensitivity_points.csv", index=False)
    low = points[points.low_temperature_exposure]
    report = {"food_dynamic_temp_complete_records": int(points.record_id.nunique()),
              "food_dynamic_temp_complete_quantified_later_points": len(points),
              "food_dynamic_temp_complete_sources": int(points.source_key.nunique()),
              "low_temperature_records": int(low.record_id.nunique()),
              "low_temperature_quantified_later_points": len(low),
              "low_temperature_sources": int(low.source_key.nunique()),
              "hypothetical_start_values_log10": [.5, 1., 1.5, 2.],
              "caveat": "First count is left-censored; none of the four values is a verified count or detection limit."}
    (OUT / "initial_count_sensitivity_cohort.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
