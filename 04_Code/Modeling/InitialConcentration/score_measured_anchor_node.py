from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Score the original Neural Baranyi ODE on measured-anchor segments."""
from pathlib import Path
import importlib.util
import pickle
import torch
import numpy as np
import pandas as pd

from run_measured_anchor import cohort, temperature_functions, segmented_records, ORIGINAL, HERE, ROOT
from run_initial_sensitivity import inference_tensor_data
from train import at_obs, DEVICE

NODE = CODE/'Modeling/NeuralODE'
spec = importlib.util.spec_from_file_location("measured_anchor_node_models", NODE / "models.py")
node_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node_models)


def main():
    targets = cohort()
    tfuns, _ = temperature_functions(targets)
    with (ORIGINAL / "deep_prepared.pkl").open("rb") as f:
        original, _, _ = pickle.load(f)
    segments = segmented_records(targets, tfuns, original)
    for fold in range(5):
        t = targets[targets.fold.eq(fold)]
        dest = HERE / "measured_anchor_predictions/node" / f"fold{fold}.parquet"
        if dest.exists():
            continue
        recs = [r for r in segments if int(r["record_id"]) in set(t.record_id)]
        frames = []
        for seed in range(3):
            stage = RESULTS/'Training/NeuralODE/runs/node' / f"fold{fold}" / f"final_s{seed}"
            assert (stage / "complete.json").exists()
            state = torch.load(stage / "model.pt", map_location=DEVICE, weights_only=False)
            meta = state["preprocess"]
            model = node_models.make_model("node", meta["no"], meta["nm"], meta["nx"],
                                           state["config"]).to(DEVICE)
            model.load_state_dict(state["model"])
            model.eval()
            data = inference_tensor_data(recs, meta)
            order = np.argsort(data.durations, kind="stable")
            with torch.no_grad():
                for start in range(0, len(order), 64):
                    ix = order[start:start + 64]
                    T, x, org, mat, dt, y0, oi, _, _ = data.batch(ix)
                    Y, _ = model(T, x, org, mat, dt, y0)
                    assert torch.isfinite(Y).all()
                    pred = at_obs(Y, oi).cpu().numpy()
                    for j, i in enumerate(ix):
                        r = data.recs[i]
                        n = len(r["obs_time"])
                        frames.append(pd.DataFrame({"record_id": int(r["record_id"]),
                            "time_h": r["obs_time"], "predicted_log": pred[j, :n] + float(r["y0"]),
                            "model": "node", "fold": fold, "seed": seed}))
        p = pd.concat(frames, ignore_index=True)
        p = p.merge(t[["record_id", "time_h", "observed_log", "source_key",
                       "low_temperature_exposure", "anchor_time_h", "anchor_log"]],
                    on=["record_id", "time_h"], validate="many_to_one")
        assert len(p) == 3 * len(t)
        dest.parent.mkdir(parents=True, exist_ok=True)
        p.to_parquet(dest, index=False)
        print(f"node measured anchor fold {fold}: {len(p)} rows", flush=True)


if __name__ == "__main__":
    main()
