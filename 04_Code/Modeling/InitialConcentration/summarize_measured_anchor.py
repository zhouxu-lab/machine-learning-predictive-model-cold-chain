from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Compare measured-anchor and hypothetical-origin predictions on shared targets."""
from pathlib import Path
import json
import pandas as pd

HERE = RESULTS/'Predictions/Dynamic'
PRIMARY = ("rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul")
VALUES = (0.5, 1.0, 1.5, 2.0)


def read_predictions(models, root):
    frames = []
    for model in models:
        for fold in range(5):
            p = HERE / root / model / f"fold{fold}.parquet"
            assert p.exists(), str(p)
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def metrics(frame):
    rows = []
    for (cohort, model, approach), g in frame.groupby(["cohort", "model", "approach"]):
        error = g.predicted_log - g.observed_log
        source_mae = g.assign(abs_error=error.abs()).groupby("source_key").abs_error.mean()
        record_mae = g.assign(abs_error=error.abs()).groupby("record_id").abs_error.mean()
        rows.append({"cohort": cohort, "model": model, "approach": approach,
                     "records": g.record_id.nunique(), "later_points": len(g),
                     "sources": g.source_key.nunique(), "pooled_mae": error.abs().mean(),
                     "source_balanced_mae": source_mae.mean(),
                     "record_balanced_mae": record_mae.mean(),
                     "bias_log": error.mean(), "underprediction_over_0p5": (error < -0.5).mean()})
    return pd.DataFrame(rows).sort_values(["cohort", "model", "approach"])


def main():
    models = list(PRIMARY)
    if all((HERE / "measured_anchor_predictions/node" / f"fold{k}.parquet").exists()
           for k in range(5)):
        models.append("node")
    measured = read_predictions(models, "measured_anchor_predictions")
    assert measured.groupby(["model", "record_id", "time_h"]).seed.nunique().eq(3).all()
    keys = ["record_id", "time_h"]
    average = measured.groupby(["model", *keys, "source_key", "low_temperature_exposure"],
                               as_index=False).agg(observed_log=("observed_log", "first"),
                                                    predicted_log=("predicted_log", "mean"))
    expected = pd.read_csv(HERE / "measured_anchor_targets.csv")
    assert average.groupby("model").size().eq(len(expected)).all()
    average["approach"] = "first_later_measured_count"

    assumed = read_predictions(models, "initial_count_sensitivity_predictions")
    assumed = assumed.merge(expected[keys], on=keys, validate="many_to_one")
    assert assumed.groupby(["model", *keys, "assumed_initial_log"]).seed.nunique().eq(3).all()
    other = assumed.groupby(["model", *keys, "source_key", "low_temperature_exposure",
                             "assumed_initial_log"], as_index=False).agg(
                                 observed_log=("observed_log", "first"),
                                 predicted_log=("predicted_log", "mean"))
    assert other.groupby(["model", "assumed_initial_log"]).size().eq(len(expected)).all()
    other["approach"] = other.assumed_initial_log.map(lambda v: f"assumed_start_{v:.1f}")
    combined = pd.concat([average, other.drop(columns="assumed_initial_log")], ignore_index=True)
    assert combined.groupby(["model", "approach"]).size().eq(len(expected)).all()
    combined["cohort"] = "all_temperature_complete_food"
    cold = combined[combined.low_temperature_exposure].copy()
    cold["cohort"] = "low_temperature_food"
    full = pd.concat([combined, cold], ignore_index=True)
    full.to_csv(HERE / "measured_anchor_matched_points.csv", index=False)
    result = metrics(full)
    result.to_csv(HERE / "measured_anchor_matched_metrics.csv", index=False)
    summary = {"status": "complete", "models": models,
               "common_future_observations": len(expected),
               "records": expected.record_id.nunique(),
               "low_temperature_records": expected[expected.low_temperature_exposure].record_id.nunique(),
               "note": "Measured anchor occurs later than the four hypothetical origins; comparisons quantify the value of an interim measured count and have different information timing. No model receives subsequent targets as input."}
    (HERE / "measured_anchor_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
