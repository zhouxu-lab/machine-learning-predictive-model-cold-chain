from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Summarize matched, source-held-out hypothetical-start predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = RESULTS/'Predictions/Dynamic'
MODELS = ["rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul"]
if all((HERE / "initial_count_sensitivity_predictions/node" / f"fold{k}.parquet").exists()
       for k in range(5)):
    MODELS.append("node")
VALUES = [.5, 1., 1.5, 2.]

def main():
    frames = []
    for model in MODELS:
        for fold in range(5):
            p = HERE / "initial_count_sensitivity_predictions" / model / f"fold{fold}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
    assert len(frames) == len(MODELS) * 5, f"Expected {len(MODELS) * 5} model-fold files; found {len(frames)}"
    pred = pd.concat(frames, ignore_index=True)
    keys = ["record_id", "time_h", "assumed_initial_log"]
    assert pred.groupby(keys + ["model"]).seed.nunique().eq(3).all()
    average = pred.groupby(keys + ["model", "fold", "source_key", "low_temperature_exposure"], as_index=False).agg(
        observed_log=("observed_log", "first"), predicted_log=("predicted_log", "mean"),
        seed_sd=("predicted_log", "std"))
    expected = pd.read_csv(HERE / "initial_count_sensitivity_points.csv")
    assert average.groupby("model").size().eq(len(expected) * len(VALUES)).all()
    baseline = average[average.model.eq("gbm")].copy()
    baseline["model"] = "persistence"
    baseline["predicted_log"] = baseline.assumed_initial_log
    baseline["seed_sd"] = 0.
    average = pd.concat([average, baseline], ignore_index=True)
    average["error_log"] = average.predicted_log - average.observed_log
    average["absolute_error"] = average.error_log.abs()
    average.to_csv(HERE / "initial_count_sensitivity_matched_points.csv", index=False)

    rows = []
    for subset, data in [("all_temperature_complete_food", average),
                         ("low_temperature_food", average[average.low_temperature_exposure])]:
        for (model, value), g in data.groupby(["model", "assumed_initial_log"]):
            by_src = g.groupby("source_key").absolute_error.mean()
            rows.append({"cohort": subset, "model": model, "assumed_initial_log": value,
                         "records": g.record_id.nunique(), "later_points": len(g),
                         "sources": g.source_key.nunique(), "pooled_mae": g.absolute_error.mean(),
                         "source_balanced_mae": by_src.mean(),
                         "record_balanced_mae": g.groupby("record_id").absolute_error.mean().mean(),
                         "bias_log": g.error_log.mean(),
                         "underprediction_over_0p5": (g.error_log < -.5).mean()})
    metrics = pd.DataFrame(rows).sort_values(["cohort", "model", "assumed_initial_log"])
    metrics.to_csv(HERE / "initial_count_sensitivity_metrics.csv", index=False)
    winners = metrics.loc[metrics.groupby(["cohort", "model"]).source_balanced_mae.idxmin(),
                          ["cohort", "model", "assumed_initial_log", "source_balanced_mae"]]
    report = {"status": "complete", "model_fold_files": len(frames),
              "same_record_time_keys_all_models": True,
              "best_assumed_values_posthoc_only": winners.to_dict(orient="records"),
              "warning": "Best values are descriptive outer-test minima, not verified initial counts or prospective selections. Cold subgroup has two sources."}
    (HERE / "initial_count_sensitivity_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
