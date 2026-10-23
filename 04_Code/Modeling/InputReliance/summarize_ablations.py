from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Summarize paired source-level ablations with bootstrap uncertainty."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = RESULTS/'Predictions/InputReliance'
a = pd.read_csv(HERE / "ablation_by_source.csv")
b = (pd.read_parquet(HERE / "ablation_predictions.parquet")
     .groupby(["model", "source_key", "removed"], as_index=False).abs_error.mean())
baseline = b[b.removed.eq("None")][["model", "source_key", "abs_error"]].rename(columns={"abs_error": "base_mae"})
b = b.merge(baseline, on=["model", "source_key"], validate="many_to_one")
b["delta_mae"] = b.abs_error - b.base_mae

rng = np.random.default_rng(20260928)
def summarize(frame, keys):
    rows = []
    for group_key, g in frame[frame.removed.ne("None")].groupby(keys):
        values = g.delta_mae.to_numpy()
        boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        row = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
        row.update(n_sources=len(values), delta_mae=values.mean(),
                   ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975),
                   positive_sources=int((values > 0).sum()))
        rows.append(row)
    return pd.DataFrame(rows)

summarize(a, ["model", "organism", "removed"]).to_csv(HERE / "ablation_organism_bootstrap.csv", index=False)
summarize(b, ["model", "removed"]).to_csv(HERE / "ablation_overall_bootstrap.csv", index=False)
