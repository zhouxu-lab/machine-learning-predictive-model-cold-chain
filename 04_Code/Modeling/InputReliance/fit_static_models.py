from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Taxonomy-resolved held-out LightGBM attribution for refrigerated foods.

This is a separate static-temperature experiment. It does not alter the saved
dynamic models or claim that the static cohort validates dynamic trajectories.
"""
from pathlib import Path
import json
import re
import sys

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT
BASE = CODE/'Shared'
OUT = RESULTS/'Training/InputReliance'
sys.path.insert(0, str(BASE))
from models_common import source_folds  # noqa: E402

CONDITIONS = ["c_" + c for c in (
    "anaerobic", "vacuum", "co2", "nacl", "nitrite", "raw", "heated",
    "sterilised", "cut", "modified_atm", "lactic", "acetic", "sorbate",
    "smoked", "mixed_flora")]
NUMERIC = ["elapsed_h", "temperature", "exposure4", "hours4", "logc0_obs",
           "ph_f", "aw_f", "ph_missing", "aw_missing", "nacl_pct"] + CONDITIONS
CATEGORICAL = ["organism", "food_category"]
COLS = NUMERIC + CATEGORICAL
GROUPS = {
    "Temperature exposure": ["temperature", "exposure4", "hours4"],
    "pH": ["ph_f", "ph_missing"],
    "Water activity": ["aw_f", "aw_missing"],
    "Food category": ["food_category"],
    "Formulation and processing": ["nacl_pct"] + CONDITIONS,
}


def data_and_folds():
    full = pd.read_parquet(DATA/'Processed/Training/features.parquet')
    data = full[(full.kind.eq("static")) & (full.temperature >= 0) &
                (full.temperature <= 10) & (~full.matrix_is_broth.astype(bool)) &
                (full.logc0_obs >= 0)].copy()
    assert not data.duplicated(["record_id", "time_h"]).any()
    assert data.groupby("record_id").source_key.nunique().max() == 1
    data["exposure4"] = np.maximum(data.temperature - 4, 0) ** 2 * data.elapsed_h
    data["hours4"] = (data.temperature > 4).astype(float) * data.elapsed_h
    def resolved_identity(row):
        organism = row.organism
        spec = (row.specification or "") if pd.notna(row.specification) else ""
        fields = re.findall(r"(?i)(?:species|serotype\(s\)):\s*([^,]+)", spec)
        designation = " ".join(fields).lower()
        if organism == "Listeria monocytogenes/innocua":
            mono = "l.monocytogenes" in designation
            innocua = "l.innocua" in designation
            if mono and not innocua:
                return "Listeria monocytogenes (identified)"
            if innocua and not mono:
                return "Listeria innocua (identified)"
            return "Listeria species unresolved"
        if organism == "Salmonella spp":
            # Require one named Salmonella species in the designation field.
            taxa = re.findall(r"\bs\.\s*([a-z]+)", designation)
            if taxa and set(taxa) == {"typhimurium"}:
                return "Salmonella Typhimurium (identified)"
            if taxa and set(taxa) == {"enteritidis"}:
                return "Salmonella Enteritidis (identified)"
            return "Salmonella species unresolved"
        if organism == "Escherichia coli":
            if re.search(r"(?i)\bo157\b", designation):
                return "Escherichia coli O157 (identified)"
            return "Escherichia coli serotype unresolved"
        return organism

    identities = data.drop_duplicates("record_id")[["record_id", "organism",
                                                  "specification"]].copy()
    identities["resolved_identity"] = identities.apply(resolved_identity, axis=1)
    data = data.drop(columns="organism").merge(
        identities[["record_id", "resolved_identity"]], on="record_id",
        validate="many_to_one").rename(columns={"resolved_identity": "organism"})
    records = data.drop_duplicates("record_id")[["record_id", "source_key"]].copy()
    records["fold"] = source_folds(records.source_key, n_folds=5, seed=20260925)
    data = data.merge(records[["record_id", "fold"]], on="record_id",
                      validate="many_to_one")
    assert len(data) > 10000 and data.record_id.nunique() > 1000
    for fold in range(5):
        te = set(data.loc[data.fold.eq(fold), "source_key"])
        tr = set(data.loc[~data.fold.eq(fold), "source_key"])
        assert te.isdisjoint(tr)
    return data


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = data_and_folds()
    data[["record_id", "source_key", "fold"]].drop_duplicates().to_csv(
        OUT / "source_heldout_folds.csv", index=False)
    contributions = np.empty((len(data), len(COLS)))
    predicted = np.empty(len(data))
    predictions = []
    for fold in range(5):
        test_mask = data.fold.eq(fold).to_numpy()
        train = data.loc[~test_mask, COLS + ["target"]].copy()
        test = data.loc[test_mask, COLS].copy()
        for part in (train, test):
            for c in CATEGORICAL:
                part[c] = part[c].fillna("__UNKNOWN__").astype(str)
        levels = {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL}
        for c in CATEGORICAL:
            train[c] = pd.Categorical(train[c], categories=levels[c])
            test[c] = pd.Categorical(test[c], categories=levels[c])
        seed_contrib, seed_pred = [], []
        for seed in range(1):
            model = LGBMRegressor(
                objective="regression", n_estimators=800, learning_rate=.05,
                num_leaves=63, min_child_samples=40, colsample_bytree=.8,
                subsample=.8, subsample_freq=1, reg_lambda=1.,
                random_state=seed, deterministic=True, force_col_wise=True,
                n_jobs=8, verbosity=-1)
            model.fit(train[COLS], train.target, categorical_feature=CATEGORICAL)
            c = model.booster_.predict(test[COLS], pred_contrib=True)
            p = model.predict(test[COLS])
            assert np.max(np.abs(c.sum(axis=1) - p)) < 1e-7
            seed_contrib.append(c[:, :-1])
            seed_pred.append(p + data.loc[test_mask, "logc0_obs"].to_numpy())
            path = OUT / "models" / f"fold{fold}_seed{seed}.joblib"
            path.parent.mkdir(exist_ok=True)
            joblib.dump({"model": model, "features": COLS,
                         "category_levels": levels}, path, compress=3)
            print("fit", fold, seed, flush=True)
        contributions[test_mask] = np.mean(seed_contrib, axis=0)
        predicted[test_mask] = np.mean(seed_pred, axis=0)
        p = data.loc[test_mask, ["record_id", "time_h", "source_key",
                                 "organism", "food_category", "logc"]].copy()
        p["predicted_log"] = predicted[test_mask]
        p["fold"] = fold
        predictions.append(p)
    pd.concat(predictions).to_parquet(OUT / "heldout_predictions.parquet", index=False)
    out = data[["record_id", "time_h", "source_key", "organism", "food_category",
                "ph_missing", "aw_missing", "response_type"]].copy()
    out["absolute_error"] = np.abs(predicted - data.logc.to_numpy())
    out["underprediction_over_2"] = predicted - data.logc.to_numpy() < -2
    shap_frame = pd.DataFrame(contributions, columns=COLS)
    for name, cols in GROUPS.items():
        out[name] = np.abs(shap_frame[cols].sum(axis=1))
    out.to_parquet(OUT / "observation_attribution.parquet", index=False)
    support = out.groupby("organism").agg(
        points=("record_id", "size"), records=("record_id", "nunique"),
        sources=("source_key", "nunique"),
        pooled_mae=("absolute_error", "mean"),
        underprediction_over_2=("underprediction_over_2", "mean"),
        ph_missing=("ph_missing", "mean"), aw_missing=("aw_missing", "mean"))
    source = out.groupby(["organism", "source_key"])[list(GROUPS) +
                                                    ["absolute_error"]].mean().reset_index()
    source.to_csv(OUT / "source_level_attribution.csv", index=False)
    balanced = source.groupby("organism")[list(GROUPS) + ["absolute_error"]].mean()
    balanced = balanced.rename(columns={"absolute_error": "source_balanced_mae"})
    summary = support.join(balanced)
    summary["eligible"] = (summary.records >= 30) & (summary.sources >= 3)
    summary.to_csv(OUT / "organism_support_and_importance.csv")

if __name__=='__main__':main()
