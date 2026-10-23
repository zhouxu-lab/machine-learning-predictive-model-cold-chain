from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Exploratory matched RF/LightGBM held-out group perturbation analysis.

Reports model reliance (increase in absolute error after group-wise within-organism
permutation), not biological sensitivity. Reuses frozen LightGBM fits and folds.
"""
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
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

FITS = RESULTS/'Training/InputReliance/models'
NUMERIC = [c for c in COLS if c not in CATEGORICAL]
GROUPS = {
    "Starting count": ["logc0_obs"],
    "Elapsed time": ["elapsed_h"],
    "Temperature and exposure": ["temperature", "exposure4", "hours4"],
    "pH": ["ph_f", "ph_missing"],
    "Water activity": ["aw_f", "aw_missing"],
    "Food category": ["food_category"],
    "Salt": ["nacl_pct", "c_nacl"],
    "Packaging atmosphere": ["c_vacuum", "c_co2", "c_modified_atm", "c_anaerobic"],
    "Acids and preservatives": ["c_nitrite", "c_lactic", "c_acetic", "c_sorbate"],
    "Raw and heat treatment": ["c_raw", "c_heated", "c_sterilised"],
    "Cutting and smoking": ["c_cut", "c_smoked"],
    "Mixed flora": ["c_mixed_flora"],
}
assert not (set(COLS) - set(sum(GROUPS.values(), [])) - {"organism"})


def rf_fit(train):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=-999), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, max_features=.8,
        n_jobs=8, random_state=0,
    )
    pipe = Pipeline([("prep", prep), ("model", model)])
    x = train[COLS].copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna("__UNKNOWN__").astype(str)
    pipe.fit(x, train.target)
    return pipe


def lightgbm_predict(fit, x):
    x = x.copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c].fillna("__UNKNOWN__").astype(str),
                              categories=fit["category_levels"][c])
    return fit["model"].predict(x)


def rf_predict(fit, x):
    x = x.copy()
    for c in CATEGORICAL:
        x[c] = x[c].fillna("__UNKNOWN__").astype(str)
    return fit.predict(x)


def shuffled(x, labels, group, rng):
    changed = x.copy()
    for org in pd.unique(labels):
        idx = np.flatnonzero(labels == org)
        if len(idx) > 1:
            donor = rng.permutation(idx)
            changed.iloc[idx, changed.columns.get_indexer(group)] = x.iloc[
                donor, x.columns.get_indexer(group)].to_numpy()
    return changed


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    data = data_and_folds()
    data = data.reset_index(drop=True)
    is_specific = data.organism.isin(SPECIFIC)
    support = (data.loc[is_specific]
               .groupby("organism")
               .agg(points=("record_id", "size"), records=("record_id", "nunique"),
                    sources=("source_key", "nunique")))
    support.to_csv(HERE / "support.csv")
    print("data", len(data), data.record_id.nunique(), data.source_key.nunique(),
          "selected", int(is_specific.sum()), flush=True)
    results = []
    baseline = []
    for fold in range(5):
        start = time.time()
        train = data.loc[data.fold.ne(fold)]
        test = data.loc[data.fold.eq(fold) & is_specific].reset_index(drop=True)
        x = test[COLS].copy()
        y = test.logc.to_numpy()
        origin = test.logc0_obs.to_numpy()
        labels = test.organism.to_numpy()
        print("fold", fold, "train", len(train), "test", len(test), flush=True)
        rf = rf_fit(train)
        gbm = joblib.load(FITS / f"fold{fold}_seed0.joblib")
        assert gbm["features"] == COLS
        for model_name, fit, predict in [
            ("Random Forest", rf, rf_predict),
            ("LightGBM", gbm, lightgbm_predict),
        ]:
            base = predict(fit, x) + origin
            base_err = np.abs(base - y)
            b = test[["organism", "source_key", "record_id", "time_h"]].copy()
            b["model"] = model_name
            b["fold"] = fold
            b["abs_error"] = base_err
            baseline.append(b)
            for group_name, cols in GROUPS.items():
                for repeat in range(3):
                    rng = np.random.default_rng(20260928 + fold * 1000 + repeat * 100 + list(GROUPS).index(group_name))
                    changed = shuffled(x, labels, cols, rng)
                    changed_origin = (changed.logc0_obs.to_numpy()
                                      if group_name == "Starting count" else origin)
                    pred = predict(fit, changed) + changed_origin
                    d = test[["organism", "source_key"]].copy()
                    d["model"] = model_name
                    d["fold"] = fold
                    d["group"] = group_name
                    d["repeat"] = repeat
                    d["delta_ae"] = np.abs(pred - y) - base_err
                    results.append(d)
                print(model_name, group_name, flush=True)
        print("fold_seconds", fold, round(time.time() - start, 1), flush=True)
    base = pd.concat(baseline, ignore_index=True)
    base.to_parquet(HERE / "baseline_predictions.parquet", index=False)
    delta = pd.concat(results, ignore_index=True)
    by_source = (delta.groupby(["model", "organism", "source_key", "group", "repeat"],
                               as_index=False).delta_ae.mean())
    by_source.to_csv(HERE / "perturbation_by_source.csv", index=False)
    source_avg = by_source.groupby(["model", "organism", "source_key", "group"], as_index=False).delta_ae.mean()
    summary = source_avg.groupby(["model", "organism", "group"], as_index=False).agg(
        delta_mae=("delta_ae", "mean"),
        source_count=("source_key", "nunique"),
    )
    summary.to_csv(HERE / "group_importance.csv", index=False)
    base_source = base.groupby(["model", "organism", "source_key"], as_index=False).abs_error.mean()
    base_summary = base_source.groupby(["model", "organism"], as_index=False).agg(
        source_balanced_mae=("abs_error", "mean"), sources=("source_key", "nunique"))
    base_summary.to_csv(HERE / "baseline_accuracy.csv", index=False)
    print("saved", HERE, flush=True)


if __name__ == "__main__":
    main()

