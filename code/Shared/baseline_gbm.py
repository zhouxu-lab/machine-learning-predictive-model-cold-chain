from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""
B1: gradient-boosting baseline on engineered temperature-history features.

For each (record, observation time t) we compute features from the temperature profile
between the first observation and t: mean T, max T, min T, T at t, time-temperature
integrals (sum of hours above 5/7/10/15 C), cumulative "Ratkowsky-like" growth potential
integral of max(T - Tmin_org, 0)^2 dt, elapsed time, initial log count, and record-level
covariates (organism, matrix, pH, aw, condition flags). Trained on static + dynamic
records; evaluated on dynamic records with leave-sources-out folds.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from models_common import load, build_dynamic_targets, metrics, temperature_function, source_folds, HERE
RES = str(RESULTS/'Training/Preparation')
os.makedirs(RES, exist_ok=True)
COND_FLAGS = {"anaerobic": "anaerobic", "vacuum": "vacuum", "co2": "carbon-dioxide", "nacl": "sodium chloride",
              "nitrite": "nitrite", "raw": r"\braw\b", "heated": "heated", "sterilised": "steril", "cut": r"\bcut\b",
              "modified_atm": "modified atmosphere", "lactic": "lactic acid", "acetic": "acetic", "sorbate": "sorb",
              "smoked": "smok", "mixed_flora": "indigenous flora"}
def history_features(times, t0, Tfun, tmin_org=-2.0):
    grid_feats = []
    for t in times:
        tt = np.linspace(t0, t, 64)
        T = Tfun(tt)
        summary = Tfun.summary(t0, t, tmin_org)
        grid_feats.append({
            "elapsed_h": t - t0, "T_at_t": float(Tfun(t)), "T_start_hist": float(Tfun(t0)),
            **summary, "n_shifts_gt3C": float(np.sum(np.abs(np.diff(T)) > 3)),
        })
    return pd.DataFrame(grid_feats)
def build(rec, cur, pro, kinds):
    tgt = build_dynamic_targets(rec, cur, pro, kinds=kinds)
    Tpro = pro[pro.variable == "Temperatures"]
    parts = []
    for rid, g in tgt.groupby("record_id", sort=False):
        Tc = g.temperature.iloc[0] if pd.notna(g.temperature.iloc[0]) else g.T_mean.iloc[0]
        Tfun = temperature_function(Tpro[Tpro.record_id == rid] if g.kind.iloc[0] == "dynamic" else None, Tc)
        f = history_features(g.time_h.values, g.t0.iloc[0], Tfun)
        f.index = g.index
        parts.append(pd.concat([g, f], axis=1))
    df = pd.concat(parts)
    ct = df.conditions_text.fillna("").str.lower()
    for k, pat in COND_FLAGS.items():
        df["c_" + k] = ct.str.contains(pat, regex=True).astype(int)
    df["nacl_pct"] = ct.str.extract(r"sodium chloride[^:]*:\s*([\d.]+)")[0].astype(float)
    df["ph_f"] = df.ph.fillna(6.5); df["aw_f"] = df.aw.fillna(0.99)
    df["ph_missing"] = df.ph.isna().astype(int); df["aw_missing"] = df.aw.isna().astype(int)
    df["target"] = df.logc - df.logc0_obs
    return df
FEATS = ["elapsed_h", "T_at_t", "T_mean_hist", "T_max_hist", "T_min_hist", "T_std_hist", "T_start_hist", "h_above5", "h_above7",
         "h_above10", "h_above15", "ratk_integral", "T_last_quarter_mean", "n_shifts_gt3C", "logc0_obs", "ph_f", "aw_f",
         "ph_missing", "aw_missing", "nacl_pct", "matrix_is_broth"] + ["c_" + k for k in COND_FLAGS]
CATS = ["organism", "food_category"]
