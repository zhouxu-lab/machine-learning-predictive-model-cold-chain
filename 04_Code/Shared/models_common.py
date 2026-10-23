from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""
Shared utilities: dataset assembly for dynamic-condition prediction, metrics, splits.

Task definition (same for every model):
  Given a record's covariates (organism, matrix, pH, aw, coded conditions), its initial
  log count (first observation) and the full temperature profile T(t), predict log N at
  every later observation time. Evaluated on dynamic records only; models may train on
  static + dynamic records.
"""
import os
import numpy as np
import pandas as pd

HERE = str(DATA/'Processed/Preparation')
CUR = os.path.join(HERE, "data", "curated")

# organism groups of interest (name substrings, lower case)
PATHOGENS = ["listeria", "salmonella", "escherichia", "yersinia", "botulinum", "bacillus cereus",
             "perfringens", "staphylococcus aureus", "campylobacter", "vibrio", "aeromonas", "shigella", "cronobacter"]


def load():
    rec = pd.read_parquet(os.path.join(CUR, "records.parquet"))
    cur = pd.read_parquet(os.path.join(CUR, "curves.parquet"))
    pro = pd.read_parquet(os.path.join(CUR, "profiles.parquet"))
    return rec, cur, pro


def temperature_function(profile_df, T_const):
    """Linear between logged points; ordered same-time values preserve jumps."""
    from profile_utils import TemperatureProfile
    return TemperatureProfile(profile_df, T_const)


def build_dynamic_targets(rec, cur, pro, kinds=("dynamic",)):
    """One row per (record, observation time > 0): target logc, plus initial logc and time."""
    r = rec[rec.kind.isin(kinds) & rec.qc_pass & rec.T_mean.notna()]
    c = cur[cur.record_id.isin(r.record_id)].sort_values(["record_id", "time_h"])
    first = c.groupby("record_id").first().rename(columns={"time_h": "t0", "logc": "logc0_obs"})
    c = c.merge(first[["t0", "logc0_obs"]], left_on="record_id", right_index=True)
    c = c[c.time_h > c.t0]
    return c.merge(r, on="record_id", how="left")


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # bias / accuracy factors on the log10 scale (Ross 1996 form applied to log counts)
    bf = float(np.mean(err))
    resid = -err                       # observed - predicted (Oscar 2005): -1.0 fail-safe .. +0.5 fail-dangerous
    apz = float(np.mean((resid >= -1.0) & (resid <= 0.5)))
    within05 = float(np.mean(np.abs(err) <= 0.5))
    return {"n": int(len(y_true)), "rmse": rmse, "mae": mae, "mean_bias_log": bf, "apz": apz, "within_0.5log": within05}


def source_folds(source_keys, n_folds=5, seed=0):
    """Leave-sources-out folds, balanced by the number of rows per source: sources are
    shuffled, then assigned largest-first to the currently smallest fold (greedy)."""
    rng = np.random.default_rng(seed)
    keys = np.array([str(s) for s in source_keys], dtype=object)
    uniq, counts = np.unique(keys, return_counts=True)
    order = rng.permutation(len(uniq))
    uniq, counts = uniq[order], counts[order]
    order = np.argsort(-counts, kind="stable")
    load = np.zeros(n_folds); fold_of_src = {}
    for i in order:
        k = int(np.argmin(load)); fold_of_src[uniq[i]] = k; load[k] += counts[i]
    return np.array([fold_of_src[s] for s in keys])


PRED_ORDER = ["b0", "b1", "gru", "transformer", "node", "node_hist"]


def load_preds(common=True):
    """Out-of-source prediction tables of every model, merged with the record flags, plus the ensemble
    (mean of the learned models). With common=True (used for all comparisons in the paper) the learned
    models are restricted to the observations that every learned model predicted, and non-finite
    predictions are dropped; with common=False the raw tables are returned."""
    res = os.path.join(HERE, "results")
    rec = pd.read_parquet(os.path.join(HERE, "data", "curated", "records.parquet"))[["record_id", "response_type", "coldchain_strict", "has_abuse", "coldchain_rte", "T_range", "source_key"]]
    preds = {}
    for k in PRED_ORDER:
        p = os.path.join(res, f"{k}_predictions.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p).merge(rec, on="record_id", how="left")
            if common:
                d = d[np.isfinite(d.y_pred)]
            preds[k] = d
    if not common:
        return preds
    ml = [k for k in preds if k != "b0"]
    if len(ml) >= 2:
        base = None
        for k in ml:
            d = preds[k][["record_id", "time_h", "y_true", "y_pred", "organism", "coldchain", "fold", "response_type", "coldchain_strict", "has_abuse", "coldchain_rte", "T_range", "source_key"]].copy()
            d["time_h"] = d.time_h.round(3); d = d.rename(columns={"y_pred": f"p_{k}"})
            base = d if base is None else base.merge(d[["record_id", "time_h", f"p_{k}"]], on=["record_id", "time_h"], how="inner")
        for k in ml:
            keep = set(zip(base.record_id, base.time_h))
            dd = preds[k].copy(); dd["time_h"] = dd.time_h.round(3)
            preds[k] = dd[[t in keep for t in zip(dd.record_id, dd.time_h)]]
        base["y_pred"] = base[[f"p_{k}" for k in ml]].mean(axis=1)
        preds["ensemble"] = base.drop(columns=[f"p_{k}" for k in ml])
    return preds
