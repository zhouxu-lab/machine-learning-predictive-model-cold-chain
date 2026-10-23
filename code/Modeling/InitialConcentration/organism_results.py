from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Recheck organism coverage, changing food conditions, and coded non-detects.

Run with the project research environment. All outputs stay in the local
publication package because the underlying ComBase data are not released here.
"""



import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
PROJECT = PACKAGE_ROOT
CATALOG = DATA/'Processed/record_catalog.csv'
POINTS = RESULTS/'Predictions/Dynamic/censor_safe_matched_points.csv'
RAW_COUNTS = DATA/'Processed/Preparation/data/tidy/dynamic_logcs.parquet'
OUT = RESULTS/'Summary_Tables/Organism_Results'
OUT.mkdir(parents=True,exist_ok=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


catalog = pd.read_csv(CATALOG)
cohort = catalog.loc[catalog.parity_record].copy()
assert len(cohort) == 511 and cohort.source_key.nunique() == 10
points = pd.read_csv(POINTS)
assert points.model.nunique() == 8
assert points.groupby("model").size().eq(4247).all()
assert points.loc[points.model.eq("gbm"), ["record_id", "time_h"]].duplicated().sum() == 0
points = points.merge(
    cohort[["record_id", "organism", "food_name", "source_key"]],
    on=["record_id", "source_key"],
    validate="many_to_one",
)


def group_metrics(group: pd.DataFrame) -> pd.Series:
    endpoint = group.sort_values("time_h").groupby("record_id", as_index=False).tail(1)
    measured_growth = endpoint.observed_log - endpoint.origin_log >= 1
    predicted_growth = endpoint.predicted_log - endpoint.origin_log >= 1
    return pd.Series(
        {
            "records": group.record_id.nunique(),
            "sources": group.source_key.nunique(),
            "quantified_later": len(group),
            "pooled_mae": group.absolute_error.mean(),
            "source_balanced_mae": group.groupby("source_key").absolute_error.mean().mean(),
            "underprediction_gt_0p5_n": int((group.error_log < -0.5).sum()),
            "underprediction_gt_0p5_pct": 100 * (group.error_log < -0.5).mean(),
            "measured_growth_records": int(measured_growth.sum()),
            "missed_growth_records": int((measured_growth & ~predicted_growth).sum()),
            "nongrowth_records": int((~measured_growth).sum()),
            "false_growth_records": int((~measured_growth & predicted_growth).sum()),
        }
    )


organisms = (
    points.groupby(["organism", "model"], sort=True)
    .apply(group_metrics, include_groups=False)
    .reset_index()
)
organisms.to_csv(OUT / "organism_model_performance.csv", index=False)

coverage = (
    cohort.groupby("organism")
    .agg(records=("record_id", "size"), sources=("source_key", "nunique"))
    .reset_index()
    .sort_values("records", ascending=False)
)
coverage = coverage.merge(
    points.loc[points.model.eq("gbm")].groupby("organism").size().rename("quantified_later"),
    on="organism",
    validate="one_to_one",
)
coverage.to_csv(OUT / "organism_coverage.csv", index=False)

raw = pd.read_parquet(RAW_COUNTS)
raw = raw.loc[raw.record_id.isin(cohort.record_id)].merge(
    cohort[["record_id", "organism", "source_key", "first_quant_h"]],
    on="record_id",
    validate="many_to_one",
)
nd = raw.loc[np.isclose(raw.logc, -0.01)].copy()
assert len(nd) == 240 and nd.record_id.nunique() == 79
assert (nd.time_h > nd.first_quant_h).all()
last_quant = raw.loc[~np.isclose(raw.logc, -0.01)].groupby("record_id").time_h.max()
nd["last_quantified_h"] = nd.record_id.map(last_quant)
nd["after_last_quantified"] = nd.time_h > nd.last_quantified_h
nd["between_quantified"] = nd.time_h < nd.last_quantified_h
nd.to_csv(OUT / "excluded_nondetect_times_local.csv", index=False)

nd_summary = (
    nd.groupby(["organism", "source_key"])
    .agg(coded_later_entries=("record_id", "size"), records=("record_id", "nunique"), after_last_quantified=("after_last_quantified", "sum"))
    .reset_index()
    .sort_values("coded_later_entries", ascending=False)
)
nd_summary.to_csv(OUT / "excluded_nondetect_by_organism_source.csv", index=False)

source_coverage = (
    cohort.groupby("source_key")
    .agg(records=("record_id", "size"))
    .reset_index()
)
source_coverage = source_coverage.merge(
    points.loc[points.model.eq("gbm")].groupby("source_key").size().rename("quantified_later"),
    on="source_key",
    validate="one_to_one",
).sort_values("quantified_later", ascending=False)
source_coverage.to_csv(OUT / "evaluation_source_coverage.csv", index=False)

strict = cohort.loc[cohort.strict_refrigerated]
no_recorded_change = strict.loc[~strict.changing_ph_gt03 & ~strict.changing_aw_gt001]
summary = {
    "inputs_sha256": {str(p): digest(p) for p in [CATALOG, POINTS, RAW_COUNTS]},
    "primary": {
        "records": len(cohort),
        "sources": cohort.source_key.nunique(),
        "quantified_later": 4247,
        "nondetect_later": len(nd),
        "nondetect_records": nd.record_id.nunique(),
        "nondetect_after_last_quantified_entries": int(nd.after_last_quantified.sum()),
        "nondetect_after_last_quantified_records": nd.loc[nd.after_last_quantified, "record_id"].nunique(),
        "nondetect_between_quantified_entries": int(nd.between_quantified.sum()),
        "pH_missing_records": int(cohort.ph.isna().sum()),
        "aw_missing_records": int(cohort.aw.isna().sum()),
        "pH_range_gt_0p3_records": int(cohort.changing_ph_gt03.sum()),
        "aw_range_gt_0p01_records": int(cohort.changing_aw_gt001.sum()),
    },
    "strict_refrigerated": {
        "records": len(strict),
        "sources": strict.source_key.nunique(),
        "pH_range_gt_0p3_records": int(strict.changing_ph_gt03.sum()),
        "aw_range_gt_0p01_records": int(strict.changing_aw_gt001.sum()),
        "no_recorded_large_change_records": len(no_recorded_change),
        "no_recorded_large_change_sources": no_recorded_change.source_key.nunique(),
        "no_recorded_large_change_missing_pH": int(no_recorded_change.ph.isna().sum()),
        "no_recorded_large_change_missing_aw": int(no_recorded_change.aw.isna().sum()),
    },
}
(OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(organisms.loc[organisms.model.eq("gbm")].to_string(index=False))
