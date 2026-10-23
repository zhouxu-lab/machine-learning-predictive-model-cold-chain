from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Score four hypothetical initial counts with the clean Neural Baranyi ODE fits."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from common import ROOT, splits
from models import make_model
from train import TensorData, at_obs, DEVICE

PROJECT = PACKAGE_ROOT
AUDIT = RESULTS/'Predictions/Dynamic'
ORIGINAL = DATA/'Processed/Preparation'
VALUES = (.5, 1., 1.5, 2.)


def tensor_data(recs, meta):
    data = object.__new__(TensorData)
    data.recs = recs
    data.lookup = {int(r["record_id"]): i for i, r in enumerate(recs)}
    data.counts = np.array([len(r["obs_idx"]) for r in recs])
    data.durations = np.array([r["dt"] for r in recs])
    maxm, n = int(data.counts.max()), len(recs)
    oi = np.zeros((n, maxm), np.float32)
    oy = np.zeros((n, maxm), np.float32)
    mask = np.zeros((n, maxm), np.float32)
    for i, r in enumerate(recs):
        k = len(r["obs_idx"])
        oi[i, :k], oy[i, :k], mask[i, :k] = r["obs_idx"], r["obs_y"], 1
    arrays = [np.stack([r["T"] for r in recs]), np.stack([r["x"] for r in recs]),
              np.array([meta["orgmap"].get(int(r["org"]), 0) for r in recs]),
              np.array([meta["matmap"].get(int(r["mat"]), 0) for r in recs]),
              np.array([r["dt"] for r in recs]), np.array([r["y0"] for r in recs]),
              oi, oy, mask]
    data.arrays = [torch.as_tensor(x, device=DEVICE) for x in arrays]
    return data


def main():
    points = pd.read_csv(AUDIT / "initial_count_sensitivity_points.csv")
    with (ORIGINAL / "deep_prepared.pkl").open("rb") as f:
        original, _, _ = pickle.load(f)
    for fold in range(5):
        subset = points[points.fold.eq(fold)]
        ids = set(map(int, subset.record_id.unique()))
        selected = [r for r in original if int(r["record_id"]) in ids]
        assert {int(r["record_id"]) for r in selected} == ids
        training_sources = {r["source_key"] for r in original
                            if int(r["record_id"]) in set(splits()[str(fold)]["train"])}
        assert training_sources.isdisjoint(set(subset.source_key))
        frames = []
        for seed in range(3):
            stage = RESULTS/'Training/NeuralODE/runs/node' / f"fold{fold}" / f"final_s{seed}"
            assert (stage / "complete.json").exists()
            state = torch.load(stage / "model.pt", map_location=DEVICE, weights_only=False)
            meta = state["preprocess"]
            model = make_model("node", meta["no"], meta["nm"], meta["nx"], state["config"]).to(DEVICE)
            model.load_state_dict(state["model"])
            model.eval()
            for value in VALUES:
                recs = []
                for r in selected:
                    item = r.copy()
                    absolute = np.asarray(r["obs_y"], dtype=np.float32) + np.float32(r["y0"])
                    item["y0"] = np.float32(value)
                    item["obs_y"] = (absolute - np.float32(value)).astype(np.float32)
                    recs.append(item)
                data = tensor_data(recs, meta)
                order = np.argsort(data.durations, kind="stable")
                with torch.no_grad():
                    for begin in range(0, len(order), 64):
                        ix = order[begin:begin + 64]
                        T, x, o, m, dt, y0, oi, _, _ = data.batch(ix)
                        Y, _ = model(T, x, o, m, dt, y0)
                        assert torch.isfinite(Y).all()
                        predictions = at_obs(Y, oi).cpu().numpy()
                        for j, i in enumerate(ix):
                            r = data.recs[i]
                            n = len(r["obs_time"])
                            frames.append(pd.DataFrame({
                                "record_id": int(r["record_id"]),
                                "time_h": r["obs_time"],
                                "predicted_log": predictions[j, :n] + value,
                                "assumed_initial_log": value,
                                "model": "node", "fold": fold, "seed": seed}))
        result = pd.concat(frames, ignore_index=True)
        result = result.merge(subset[["record_id", "time_h", "observed_log", "source_key",
                                      "low_temperature_exposure"]],
                              on=["record_id", "time_h"], how="inner", validate="many_to_one")
        assert result.observed_log.notna().all()
        assert result.groupby(["record_id", "time_h", "assumed_initial_log"]).seed.nunique().eq(3).all()
        dest = AUDIT / "initial_count_sensitivity_predictions/node" / f"fold{fold}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(dest, index=False)
        print(f"node fold {fold}: {len(result)} rows", flush=True)


if __name__ == "__main__":
    main()
