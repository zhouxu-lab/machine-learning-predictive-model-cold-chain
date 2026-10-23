from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Score hypothetical initial counts with newly refitted source-held-out models.

This evaluates only later *quantified* counts on the same food-record keys.
The four starting values remain assumptions, never measured initial counts.
"""
from pathlib import Path
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
import joblib
import torch

ROOT = PACKAGE_ROOT
HERE = RESULTS/'Predictions/Dynamic'
EXP = CODE/'Modeling/Primary'
ORIGINAL = DATA/'Processed/Preparation'
sys.path.insert(0, str(EXP))
from common import splits  # noqa: E402
from models import make_model  # noqa: E402
from train import ABS_T, TensorData, at_obs, transform  # noqa: E402
sys.path.insert(0, str(ORIGINAL))
from baseline_gbm import FEATS, CATS  # noqa: E402

VALUES = (.5, 1., 1.5, 2.)
MODELS = ("rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul")
DEVICE = torch.device(os.environ.get("COMBASE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

def outcome_keys():
    points = pd.read_csv(HERE / "initial_count_sensitivity_points.csv")
    assert not points.duplicated(["record_id", "time_h"]).any()
    return points

def full_model_paths(kind, fold):
    stage = RESULTS/'Training/Primary/runs' / kind / f"fold{fold}"
    assert (stage / "selected.json").exists(), f"Missing selected config: {kind} fold {fold}"
    return [stage / f"final_s{seed}" for seed in range(3)]

def tree_predictions(kind, fold, points):
    base = pd.read_parquet(ORIGINAL / "features.parquet")
    base = base.merge(points[["record_id", "time_h"]], on=["record_id", "time_h"],
                      how="inner", validate="one_to_one")
    assert len(base) == len(points)
    output = []
    for seed, stage in enumerate(full_model_paths(kind, fold)):
        assert (stage / "complete.json").exists() and (stage / "model.joblib").exists()
        artifact = joblib.load(stage / "model.joblib")
        for value in VALUES:
            data = base.copy()
            data["logc0_obs"] = value
            for c in CATS:
                data[c] = data[c].fillna("__UNKNOWN__").astype(str)
            if kind == "rf":
                model = artifact
                p = model.predict(data[FEATS + CATS]) + value
            else:
                levels = artifact["category_levels"]
                for c in CATS:
                    data[c] = pd.Categorical(data[c], categories=levels[c])
                p = artifact["model"].predict(data[FEATS + CATS]) + value
            output.append(pd.DataFrame({"record_id": data.record_id.to_numpy(),
                                        "time_h": data.time_h.to_numpy(),
                                        "assumed_initial_log": value, "model": kind,
                                        "fold": fold, "seed": seed, "predicted_log": p}))
    return pd.concat(output, ignore_index=True)

def inference_tensor_data(recs, metadata):
    obj = object.__new__(TensorData)
    obj.recs = recs
    obj.lookup = {int(r["record_id"]): i for i, r in enumerate(recs)}
    obj.counts = np.array([len(r["obs_idx"]) for r in recs])
    obj.durations = np.array([r["dt"] for r in recs])
    maxm = int(obj.counts.max())
    n = len(recs)
    oi = np.zeros((n, maxm), np.float32)
    oy = np.zeros((n, maxm), np.float32)
    mask = np.zeros((n, maxm), np.float32)
    for i, r in enumerate(recs):
        k = len(r["obs_idx"])
        oi[i, :k] = r["obs_idx"]
        oy[i, :k] = r["obs_y"]
        mask[i, :k] = 1
    arrays = [np.stack([r["T"] for r in recs]), np.stack([r["x"] for r in recs]),
              np.array([metadata["orgmap"].get(int(r["org"]), 0) for r in recs]),
              np.array([metadata["matmap"].get(int(r["mat"]), 0) for r in recs]),
              np.array([r["dt"] for r in recs]), np.array([r["y0"] for r in recs]),
              oi, oy, mask]
    obj.arrays = [torch.as_tensor(x, device=DEVICE) for x in arrays]
    return obj

def neural_predictions(kind, fold, points, original_deep):
    ids = set(map(int, points.record_id.unique()))
    selected = [r for r in original_deep if int(r["record_id"]) in ids]
    assert {int(r["record_id"]) for r in selected} == ids
    output = []
    for seed, stage in enumerate(full_model_paths(kind, fold)):
        assert (stage / "complete.json").exists() and (stage / "model.pt").exists()
        state = torch.load(stage / "model.pt", map_location=DEVICE, weights_only=False)
        meta = state["preprocess"]
        model = make_model(kind, meta["no"], meta["nm"], meta["nx"], state["config"],
                           abs_time=ABS_T if kind.endswith("_abs") else None).to(DEVICE)
        model.load_state_dict(state["model"])
        model.eval()
        for value in VALUES:
            recs = []
            for r in selected:
                item = r.copy()
                original_absolute = np.asarray(r["obs_y"], dtype=np.float32) + np.float32(r["y0"])
                item["y0"] = np.float32(value)
                item["obs_y"] = (original_absolute - np.float32(value)).astype(np.float32)
                recs.append(item)
            recs = transform(kind, recs)
            data = inference_tensor_data(recs, meta)
            frames = []
            order = np.argsort(data.durations, kind="stable")
            with torch.no_grad():
                for start in range(0, len(order), 64):
                    ix = order[start:start + 64]
                    T, x, o, m, dt, y0, oi, _, _ = data.batch(ix)
                    Y, _ = model(T, x, o, m, dt, y0)
                    assert torch.isfinite(Y).all(), f"Nonfinite prediction: {kind}, fold {fold}, seed {seed}, initial {value}"
                    p = at_obs(Y, oi).cpu().numpy()
                    for j, i in enumerate(ix):
                        r = data.recs[i]
                        n = len(r["obs_time"])
                        frames.append(pd.DataFrame({"record_id": int(r["record_id"]),
                            "time_h": r["obs_time"], "predicted_log": p[j, :n] + value}))
            frame = pd.concat(frames, ignore_index=True)
            frame = frame.merge(points[["record_id", "time_h"]], on=["record_id", "time_h"],
                                how="inner", validate="one_to_one")
            assert len(frame) == len(points)
            frame["assumed_initial_log"] = value
            frame["model"] = kind
            frame["fold"] = fold
            frame["seed"] = seed
            output.append(frame)
    return pd.concat(output, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, help="Run one model, or omit for all")
    parser.add_argument("--fold", type=int, choices=range(5), help="Run one fold for preflight")
    args = parser.parse_args()
    points = outcome_keys()
    original_deep = None
    if args.model not in ("rf", "gbm"):
        with (ORIGINAL / "deep_prepared.pkl").open("rb") as f:
            original_deep, _, _ = pickle.load(f)
    kinds = (args.model,) if args.model else MODELS
    for kind in kinds:
        for fold in ([args.fold] if args.fold is not None else range(5)):
            subset = points[points.fold == fold]
            if subset.empty:
                continue
            train_sources = {r["source_key"] for r in original_deep if int(r["record_id"]) in splits()[str(fold)]["train"]} if original_deep is not None else set()
            assert not train_sources.intersection(set(subset.source_key))
            dest = HERE / "initial_count_sensitivity_predictions" / kind / f"fold{fold}.parquet"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            result = tree_predictions(kind, fold, subset) if kind in ("rf", "gbm") else neural_predictions(kind, fold, subset, original_deep)
            result = result.merge(subset[["record_id", "time_h", "observed_log", "source_key",
                                          "low_temperature_exposure"]],
                                  on=["record_id", "time_h"], how="left", validate="many_to_one")
            assert result.observed_log.notna().all()
            assert result.groupby(["record_id", "time_h", "assumed_initial_log"]).seed.nunique().eq(3).all()
            result.to_parquet(dest, index=False)
            print(json.dumps({"model": kind, "fold": fold, "rows": len(result), "path": str(dest)}), flush=True)

if __name__ == "__main__":
    main()
