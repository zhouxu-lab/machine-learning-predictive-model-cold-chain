from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Aggregate matched source-held-out results after all censor-safe fits finish."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = PACKAGE_ROOT
HERE = RESULTS/'Predictions/Dynamic'
EXP = RESULTS/('Training/Primary' if '--from-training' in sys.argv else 'Predictions/Fitted/Primary')
NODE_EXP = RESULTS/('Training/NeuralODE' if '--from-training' in sys.argv else 'Predictions/Fitted/NeuralODE')
KINDS = ["rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul", "node"]

def main():
    cat = pd.read_csv(DATA/'Processed/record_catalog.csv')
    cohort = cat[cat.parity_record][["record_id", "strict_refrigerated", "source_key"]]
    assert len(cohort) == 511
    frames = []
    for kind in KINDS:
        for fold in range(5):
            for seed in range(3):
                experiment = NODE_EXP if kind == "node" else EXP
                stage = experiment / "runs" / kind / f"fold{fold}" / f"final_s{seed}"
                assert (stage / "complete.json").exists(), stage
                d = pd.read_parquet(stage / "predictions.parquet")
                assert not d.duplicated(["record_id", "time_h"]).any(), stage
                assert d.seed.eq(seed).all(), stage
                d = d.merge(cohort, on="record_id", how="inner", suffixes=("", "_catalog"), validate="many_to_one")
                assert d.source_key.eq(d.source_key_catalog).all()
                d["model"] = kind
                frames.append(d.drop(columns="source_key_catalog"))
    raw = pd.concat(frames, ignore_index=True)
    keys = ["record_id", "time_h", "model"]
    assert not raw.duplicated(keys + ["seed"]).any()
    assert raw.groupby(keys).seed.nunique().eq(3).all()
    for col in ["y_true", "logc0_obs"]:
        spread = raw.groupby(["record_id", "time_h"])[col].agg(["min", "max"])
        assert (spread["max"] - spread["min"]).le(1e-6).all(), col
    assert np.isfinite(raw[["y_true", "logc0_obs", "y_pred"]].to_numpy()).all()
    matched = raw.groupby(keys + ["source_key", "strict_refrigerated"], as_index=False).agg(
        observed_log=("y_true", "first"), origin_log=("logc0_obs", "first"),
        predicted_log=("y_pred", "mean"), seed_sd=("y_pred", "std"))
    assert matched.groupby("model").size().eq(4247).all()
    assert matched.groupby("model").record_id.nunique().eq(511).all()
    reference_keys = set(map(tuple, matched.loc[matched.model.eq("gbm"), ["record_id", "time_h"]].to_numpy()))
    for kind, group in matched.groupby("model"):
        assert set(map(tuple, group[["record_id", "time_h"]].to_numpy())) == reference_keys, kind
    baseline = matched[matched.model.eq("gbm")].copy()
    baseline["model"] = "persistence"
    baseline["predicted_log"] = baseline.origin_log
    baseline["seed_sd"] = 0.
    matched = pd.concat([matched, baseline], ignore_index=True)
    matched["error_log"] = matched.predicted_log - matched.observed_log
    matched["absolute_error"] = matched.error_log.abs()
    matched.to_csv(HERE / "censor_safe_matched_points.csv", index=False)

    rows = []
    for cohort_name, data in [("expanded", matched),
                              ("strict_refrigerated", matched[matched.strict_refrigerated])]:
        for model, g in data.groupby("model"):
            rows.append({"cohort": cohort_name, "model": model, "later_points": len(g),
                         "records": g.record_id.nunique(), "sources": g.source_key.nunique(),
                         "mae": g.absolute_error.mean(),
                         "source_balanced_mae": g.groupby("source_key").absolute_error.mean().mean(),
                         "rmse": np.sqrt(np.mean(g.error_log ** 2)),
                         "bias_log": g.error_log.mean(),
                         "underprediction_over_0p5": (g.error_log < -.5).mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(HERE / "censor_safe_model_metrics.csv", index=False)
    report = {"status": "complete", "model_fits": len(KINDS) * 5 * 3,
              "models": KINDS,
              "expanded_records": 511, "expanded_quantified_later_points": 4247,
              "comparison": "Same record/time keys as archived review; all fitted models trained without coded counts."}
    (HERE / "censor_safe_results_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
