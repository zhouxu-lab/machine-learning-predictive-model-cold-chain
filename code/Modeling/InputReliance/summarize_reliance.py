from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Source-level uncertainty and paired agreement for exploratory importance results."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = RESULTS/'Predictions/InputReliance'
raw = pd.read_csv(HERE / "perturbation_by_source.csv")
source = raw.groupby(["model", "organism", "source_key", "group"], as_index=False).delta_ae.mean()
rng = np.random.default_rng(20260928)
rows = []
for (model, org, group), d in source.groupby(["model", "organism", "group"]):
    values = d.delta_ae.to_numpy()
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    rows.append({"model": model, "organism": org, "group": group,
                 "sources": len(values), "delta_mae": values.mean(),
                 "ci_low": np.quantile(draws, .025),
                 "ci_high": np.quantile(draws, .975),
                 "positive_sources": int((values > 0).sum()),
                 "negative_sources": int((values < 0).sum())})
summary = pd.DataFrame(rows)
summary.to_csv(HERE / "source_bootstrap_summary.csv", index=False)

paired = source.pivot_table(index=["organism", "source_key", "group"],
                            columns="model", values="delta_ae").reset_index()
paired.to_csv(HERE / "paired_source_importance.csv", index=False)
report = []
for org in summary.organism.unique():
    a = summary[summary.organism.eq(org)]
    pivot = a.pivot(index="group", columns="model", values="delta_mae")
    both = pivot[(pivot["LightGBM"] > 0) & (pivot["Random Forest"] > 0)]
    report.append({"organism": org, "positive_both": ", ".join(both.index),
                   "top_lgbm": pivot["LightGBM"].idxmax(),
                   "top_rf": pivot["Random Forest"].idxmax()})
pd.DataFrame(report).to_csv(HERE / "cross_model_agreement.csv", index=False)
print(summary[summary.organism.isin([
    "Listeria monocytogenes (identified)",
    "Escherichia coli O157 (identified)",
    "Brochothrix thermosphacta",
    "Staphylococcus aureus",
]) & summary.group.isin([
    "Starting count", "Elapsed time", "Temperature and exposure", "pH",
    "Packaging atmosphere", "Food category",
])].sort_values(["organism", "group", "model"]).to_string(index=False,
    float_format=lambda v: f"{v:.3f}"))
