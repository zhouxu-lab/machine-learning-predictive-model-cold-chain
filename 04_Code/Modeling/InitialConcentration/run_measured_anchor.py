from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Predict after the first quantified count in records with a non-detected origin.

Uses the censor-safe models fitted on other sources. The measured anchor is an
input, never a target; no later count is provided to the model during scoring.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import torch

HERE = RESULTS/'Predictions/Dynamic'
ROOT = PACKAGE_ROOT
ORIGINAL = DATA/'Processed/Preparation'
EXP = CODE/'Modeling/Primary'
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(EXP))
from models import make_model  # noqa: E402
from train import ABS_T, at_obs, transform, DEVICE  # noqa: E402
from baseline_gbm import FEATS, CATS, history_features  # noqa: E402
from models_common import temperature_function  # noqa: E402
from run_initial_sensitivity import inference_tensor_data, full_model_paths  # noqa: E402

KINDS = ("rf", "gbm", "gru_abs", "transformer_abs", "backbone", "mul")


def cohort():
    d = pd.read_csv(HERE / "initial_count_sensitivity_points.csv")
    d = d.sort_values(["record_id", "time_h"])
    assert not d.duplicated(["record_id", "time_h"]).any()
    eligible = d.groupby("record_id").size()
    d = d[d.record_id.isin(eligible[eligible >= 2].index)].copy()
    anchors = d.groupby("record_id", as_index=False).first()[
        ["record_id", "time_h", "observed_log"]].rename(
            columns={"time_h": "anchor_time_h", "observed_log": "anchor_log"})
    targets = d.merge(anchors, on="record_id", validate="many_to_one")
    targets = targets[targets.time_h > targets.anchor_time_h].copy()
    assert targets.record_id.nunique() == 266 and len(targets) == 1194
    assert len(targets[targets.low_temperature_exposure].record_id.unique()) == 60
    assert targets.groupby("source_key").fold.nunique().eq(1).all()
    return targets


def temperature_functions(targets):
    ids = set(map(int, targets.record_id.unique()))
    raw = pd.read_parquet(ORIGINAL / "features.parquet")
    meta = raw[raw.record_id.isin(ids)].drop_duplicates("record_id")
    profile = pd.read_parquet(ORIGINAL / "data/curated/profiles.parquet")
    profile = profile[(profile.variable == "Temperatures") & profile.record_id.isin(ids)]
    groups = {int(rid): g for rid, g in profile.groupby("record_id")}
    out = {}
    for r in meta.itertuples():
        tc = r.temperature if pd.notna(r.temperature) else r.T_mean
        out[int(r.record_id)] = temperature_function(groups.get(int(r.record_id)), tc)
    assert set(out) == ids
    return out, raw


def tree_features(targets, tfuns, raw):
    d = raw.merge(targets[["record_id", "time_h", "anchor_time_h", "anchor_log"]],
                  on=["record_id", "time_h"], validate="one_to_one")
    assert len(d) == len(targets)
    parts = []
    for rid, g in d.groupby("record_id", sort=False):
        f = history_features(g.time_h.to_numpy(), float(g.anchor_time_h.iloc[0]), tfuns[int(rid)])
        f.index = g.index
        parts.append(f)
    history = pd.concat(parts).loc[d.index]
    for c in history.columns:
        d[c] = history[c]
    d["logc0_obs"] = d.anchor_log
    assert np.allclose(d.elapsed_h, d.time_h - d.anchor_time_h)
    return d


def tree_predict(kind, fold, targets, features):
    d = features[features.record_id.isin(targets.record_id)].copy()
    out = []
    for seed, stage in enumerate(full_model_paths(kind, fold)):
        assert (stage / "complete.json").exists()
        artifact = joblib.load(stage / "model.joblib")
        x = d.copy()
        for c in CATS:
            x[c] = x[c].fillna("__UNKNOWN__").astype(str)
        if kind == "rf":
            p = artifact.predict(x[FEATS + CATS])
        else:
            for c in CATS:
                x[c] = pd.Categorical(x[c], categories=artifact["category_levels"][c])
            p = artifact["model"].predict(x[FEATS + CATS])
        out.append(pd.DataFrame({"record_id": x.record_id, "time_h": x.time_h,
                                 "predicted_log": np.asarray(p) + x.anchor_log.to_numpy(),
                                 "model": kind, "fold": fold, "seed": seed}))
    return pd.concat(out, ignore_index=True)


def segmented_records(targets, tfuns, original):
    ids = set(map(int, targets.record_id.unique()))
    old = {int(r["record_id"]): r for r in original if int(r["record_id"]) in ids}
    assert set(old) == ids
    out = []
    for rid, g in targets.groupby("record_id", sort=False):
        anchor_t = float(g.anchor_time_h.iloc[0])
        end_t = float(g.time_h.max())
        y0 = float(g.anchor_log.iloc[0])
        assert end_t > anchor_t
        r = old[int(rid)].copy()
        r["T"] = tfuns[int(rid)](np.linspace(anchor_t, end_t, 96)).astype(np.float32)
        r["dt"] = np.float32((end_t - anchor_t) / 95)
        r["y0"] = np.float32(y0)
        r["obs_time"] = g.time_h.to_numpy(dtype=np.float64)
        r["obs_idx"] = ((r["obs_time"] - anchor_t) / (end_t - anchor_t) * 95).astype(np.float32)
        r["obs_y"] = (g.observed_log.to_numpy(dtype=np.float32) - np.float32(y0)).astype(np.float32)
        out.append(r)
    return out


def neural_predict(kind, fold, targets, recs):
    recs = [r for r in recs if int(r["record_id"]) in set(targets.record_id)]
    recs = transform(kind, recs)
    out = []
    for seed, stage in enumerate(full_model_paths(kind, fold)):
        assert (stage / "complete.json").exists()
        state = torch.load(stage / "model.pt", map_location=DEVICE, weights_only=False)
        meta = state["preprocess"]
        model = make_model(kind, meta["no"], meta["nm"], meta["nx"], state["config"],
                           abs_time=ABS_T if kind.endswith("_abs") else None).to(DEVICE)
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
                    out.append(pd.DataFrame({"record_id": int(r["record_id"]),
                        "time_h": r["obs_time"], "predicted_log": pred[j, :n] + float(r["y0"]),
                        "model": kind, "fold": fold, "seed": seed}))
    return pd.concat(out, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=KINDS)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    targets = cohort()
    tfuns, raw = temperature_functions(targets)
    tree = tree_features(targets, tfuns, raw) if args.model in (None, "rf", "gbm") else None
    recs = None
    if args.model not in ("rf", "gbm"):
        with (ORIGINAL / "deep_prepared.pkl").open("rb") as f:
            original, _, _ = pickle.load(f)
        recs = segmented_records(targets, tfuns, original)
    targets.to_csv(HERE / "measured_anchor_targets.csv", index=False)
    kinds = (args.model,) if args.model else KINDS
    for kind in kinds:
        for fold in ([args.fold] if args.fold is not None else range(5)):
            t = targets[targets.fold.eq(fold)]
            if t.empty:
                continue
            dest = HERE / "measured_anchor_predictions" / kind / f"fold{fold}.parquet"
            if dest.exists():
                continue
            p = tree_predict(kind, fold, t, tree) if kind in ("rf", "gbm") else neural_predict(kind, fold, t, recs)
            p = p.merge(t[["record_id", "time_h", "observed_log", "source_key",
                           "low_temperature_exposure", "anchor_time_h", "anchor_log"]],
                        on=["record_id", "time_h"], validate="many_to_one")
            assert len(p) == 3 * len(t)
            assert p.groupby(["record_id", "time_h"]).seed.nunique().eq(3).all()
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.to_parquet(dest, index=False)
            print(json.dumps({"model": kind, "fold": fold, "rows": len(p)}), flush=True)


if __name__ == "__main__":
    main()
