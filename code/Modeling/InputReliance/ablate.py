from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Refit selected input-group ablations on the matched isothermal source folds."""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = RESULTS/'Predictions/InputReliance'
HERE.mkdir(parents=True,exist_ok=True)
ROOT = PACKAGE_ROOT
PREVIEW = CODE/'Modeling/InputReliance'
sys.path.insert(0, str(PREVIEW))
from fit_static_models import data_and_folds, COLS, CATEGORICAL  # noqa: E402
from organism_groups import SPECIFIC  # noqa: E402

GROUPS = {
    "Food category": ["food_category"],
    "pH": ["ph_f", "ph_missing"],
    "Water activity": ["aw_f", "aw_missing"],
    "Packaging atmosphere": ["c_vacuum", "c_co2", "c_modified_atm", "c_anaerobic"],
    "Temperature and exposure": ["temperature", "exposure4", "hours4"],
    "Elapsed time": ["elapsed_h"],
}


def prepare_categories(train, test, cats):
    train = train.copy()
    test = test.copy()
    levels = {}
    for c in cats:
        levels[c] = sorted(train[c].fillna("__UNKNOWN__").astype(str).unique().tolist())
        train[c] = pd.Categorical(train[c].fillna("__UNKNOWN__").astype(str), categories=levels[c])
        test[c] = pd.Categorical(test[c].fillna("__UNKNOWN__").astype(str), categories=levels[c])
    return train, test


def fit_predict(train, test, cols, name):
    cats = [c for c in CATEGORICAL if c in cols]
    if name == "LightGBM":
        tr, te = prepare_categories(train[cols], test[cols], cats)
        fit = LGBMRegressor(
            objective="regression", n_estimators=800, learning_rate=.05,
            num_leaves=63, min_child_samples=40, colsample_bytree=.8,
            subsample=.8, subsample_freq=1, reg_lambda=1., random_state=0,
            deterministic=True, force_col_wise=True, n_jobs=8, verbosity=-1)
        fit.fit(tr, train.target, categorical_feature=cats)
        return fit.predict(te)
    num = [c for c in cols if c not in cats]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=-999), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    fit = Pipeline([("prep", prep), ("model", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, max_features=.8,
        n_jobs=8, random_state=0))])
    tr = train[cols].copy()
    te = test[cols].copy()
    for c in cats:
        tr[c] = tr[c].fillna("__UNKNOWN__").astype(str)
        te[c] = te[c].fillna("__UNKNOWN__").astype(str)
    fit.fit(tr, train.target)
    return fit.predict(te)


def main():
    data = data_and_folds().reset_index(drop=True)
    selected = data.organism.isin(SPECIFIC)
    rows = []
    for fold in range(5):
        start = time.time()
        train = data[data.fold.ne(fold)]
        test = data[data.fold.eq(fold)].reset_index(drop=True)
        origin = test.logc0_obs.to_numpy()
        truth = test.logc.to_numpy()
        for model in ("LightGBM", "Random Forest"):
            for removed, absent in [("None", [])] + list(GROUPS.items()):
                cols = [c for c in COLS if c not in absent]
                pred = fit_predict(train, test, cols, model) + origin
                out = test[["record_id", "time_h", "source_key", "organism"]].copy()
                out["model"] = model
                out["fold"] = fold
                out["removed"] = removed
                out["abs_error"] = np.abs(pred - truth)
                rows.append(out)
                print("fold", fold, model, removed, flush=True)
        print("fold_seconds", fold, round(time.time()-start, 1), flush=True)
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_parquet(HERE / "ablation_predictions.parquet", index=False)
    # Per-source mean error, then equal weighting over sources for each organism.
    a = all_rows[all_rows.organism.isin(SPECIFIC)].groupby(
        ["model", "organism", "source_key", "removed"], as_index=False).abs_error.mean()
    base = a[a.removed.eq("None")][["model", "organism", "source_key", "abs_error"]].rename(
        columns={"abs_error": "base_mae"})
    a = a.merge(base, on=["model", "organism", "source_key"], validate="many_to_one")
    a["delta_mae"] = a.abs_error - a.base_mae
    a.to_csv(HERE / "ablation_by_source.csv", index=False)
    a.groupby(["model", "organism", "removed"], as_index=False).agg(
        delta_mae=("delta_mae", "mean"), sources=("source_key", "nunique")
    ).to_csv(HERE / "ablation_summary.csv", index=False)
    # All-source aggregate, regardless of identity subset.
    overall = all_rows.groupby(["model", "source_key", "removed"], as_index=False).abs_error.mean()
    overall.groupby(["model", "removed"], as_index=False).abs_error.mean().to_csv(
        HERE / "ablation_all_sources.csv", index=False)
    print("saved", HERE, flush=True)


if __name__ == "__main__":
    main()

