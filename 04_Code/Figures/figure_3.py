from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
from figure_export import export
"""Refine manuscript Figures 1, 3, and 5 from frozen analysis outputs."""
from pathlib import Path
import json
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
HERE = RESULTS/'Figure_Source_Data'
PACKAGE = PACKAGE_ROOT
PROJECT = PACKAGE_ROOT
AUDIT = RESULTS/'Predictions/Dynamic'
PROFILES = DATA/'Processed/Archive/profiles.parquet'
USDA = RESULTS/'Figure_Source_Data/Official_Predictor/official_curves_raw.json'
RELIANCE = RESULTS/'Predictions/InputReliance/group_importance.csv'

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .8,
    "savefig.facecolor": "white",
})
MODEL_COLORS = {
    "gbm": "#008B8B", "rf": "#22577A", "transformer_abs": "#815B9C",
    "gru_abs": "#A25535", "node": "#596B91", "mul": "#387C4C",
}
MODEL_NAMES = {
    "gbm": "LightGBM", "rf": "Random Forest", "transformer_abs": "Transformer",
    "gru_abs": "GRU", "node": "Neural Baranyi ODE", "mul": "Baranyi–neural hybrid",
}
def temperature_axis_limits(values):
    high = float(np.max(values))
    if high <= 5:
        return 0, 5, [0, 2.5, 5]
    if high <= 8:
        return 0, 8, [0, 4, 8]
    if high <= 12:
        return 0, 12, [0, 6, 12]
    return 0, 35, [0, 15, 30]
def figure3(points):
    cases = [49920, 50473, 49361, 49375, 59265, 50780]
    titles = ["Semi skimmed milk", "Pâté", "Beef hamburger · Pseudomonads",
              r"Beef hamburger · $\it{B.\ thermosphacta}$", "Cooked cured pork shoulder",
              "White pudding"]
    profiles = pd.read_parquet(PROFILES)
    profiles = profiles[profiles.record_id.isin(cases) & profiles.variable.eq("Temperatures")]
    assert set(profiles.record_id) == set(cases)
    profiles = profiles.sort_values(["record_id", "time_h"])
    assert profiles.groupby("record_id").time_h.apply(lambda x: x.is_monotonic_increasing).all()
    assert not profiles.duplicated(["record_id", "time_h"]).any()
    profiles.to_csv(HERE / "Figure_3_temperature_source_data.csv", index=False)
    raw = json.loads(USDA.read_text(encoding="utf-8"))
    official = {int(item["record_id"]): item for item in raw}
    fig = plt.figure(figsize=(7.5, 10.15))
    outer = fig.add_gridspec(3, 2, left=.115, right=.98, top=.965, bottom=.105,
                             hspace=.39, wspace=.34)
    count_axes, temp_axes = [], []
    models = ["gbm", "rf", "transformer_abs", "gru_abs", "node", "mul"]
    for k, (rid, title) in enumerate(zip(cases, titles)):
        row, col = divmod(k, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3.1, 1], hspace=.04)
        count_ax = fig.add_subplot(inner[0])
        temp_ax = fig.add_subplot(inner[1], sharex=count_ax)
        count_axes.append(count_ax)
        temp_axes.append(temp_ax)
        d = points[points.record_id.eq(rid)]
        obs = d[d.model.eq("gbm")].sort_values("time_h")
        assert len(obs) > 2 and rid in official
        origin = float(obs.origin_log.iloc[0])
        count_ax.scatter([0], [origin], s=15, color="black", zorder=7)
        for model in models:
            m = d[d.model.eq(model)].sort_values("time_h")
            count_ax.plot(np.r_[0, m.time_h], np.r_[origin, m.predicted_log],
                          color=MODEL_COLORS[model], lw=1.05, alpha=.85,
                          label=MODEL_NAMES[model])
        u = pd.DataFrame(official[rid]["points"])
        count_ax.plot(u.time_h, u.predicted_log, color="#222222", lw=1.45,
                      ls="--", label="USDA Predictor")
        count_ax.scatter(obs.time_h, obs.observed_log, s=14, color="black",
                         zorder=8, label="Measured")
        count_ax.set_ylim(1, 9.2)
        count_ax.set_title(f"{chr(97+k)}  {title}", loc="left", fontsize=8,
                           fontweight="bold", pad=4)
        count_ax.set_ylabel("Microbial concentration\n(log₁₀ CFU/g)", fontsize=7.3, labelpad=2, linespacing=1.05)
        count_ax.tick_params(labelbottom=False, labelsize=7)
        count_ax.grid(color="#E9ECEE", lw=.45)
        count_ax.set_axisbelow(True)
        temp = profiles[profiles.record_id.eq(rid)]
        assert temp.time_h.iloc[0] == 0
        assert temp.time_h.iloc[-1] >= obs.time_h.max()
        temp_ax.plot(temp.time_h, temp.value, color="#4E6672", lw=1.1)
        low, high, ticks = temperature_axis_limits(temp.value)
        temp_ax.set_ylim(low, high)
        temp_ax.set_yticks(ticks)
        temp_ax.set_ylabel("°C", fontsize=7.2)
        temp_ax.set_xlabel("Time (h)", fontsize=7.2)
        temp_ax.tick_params(labelsize=6.8, length=2)
        temp_ax.grid(axis="y", color="#EDF0F1", lw=.45)
        temp_ax.set_axisbelow(True)
        temp_ax.set_xlim(0, max(float(temp.time_h.max()), float(u.time_h.max())))
    handles, labels = count_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, .008), fontsize=7.1)
    fig.canvas.draw()
    # Count and temperature rows have different intended heights. Audit each
    # comparable group at the rendered geometry.
    out = HERE / "Figure_3_food_trajectories_temperature_refined"
    require_matplotlib_panel_alignment(
        fig, axes=count_axes, panel_ids=list("abcdef"),
        row_groups=[["a", "b"], ["c", "d"], ["e", "f"]],
        column_groups=[["a", "c", "e"], ["b", "d", "f"]],
        json_out=str(out) + ".counts.alignment.json", strict=True)
    require_matplotlib_panel_alignment(
        fig, axes=temp_axes, panel_ids=list("abcdef"),
        row_groups=[["a", "b"], ["c", "d"], ["e", "f"]],
        column_groups=[["a", "c", "e"], ["b", "d", "f"]],
        json_out=str(out) + ".temperatures.alignment.json", strict=True)
    export(fig,'Figure_3')
FULL_GROUPS = [
    "Starting count", "Elapsed time", "Temperature and exposure", "pH",
    "Water activity", "Food category", "Salt", "Packaging atmosphere",
    "Acids and preservatives", "Raw and heat treatment", "Cutting and smoking",
    "Mixed flora",
]
SHORT_GROUPS = ["Initial", "Time", "Temp.", "pH", "aw", "Food", "Salt",
                "Atmos.", "Acids", "Heat", "Cut", "Flora"]
ORGANISMS = [
    "Listeria monocytogenes (identified)", "Escherichia coli O157 (identified)",
    "Brochothrix thermosphacta", "Salmonella Typhimurium (identified)",
    "Salmonella Enteritidis (identified)", "Yersinia enterocolitica",
    "Bacillus cereus", "Aeromonas hydrophila", "Staphylococcus aureus",
    "Listeria innocua (identified)", "Clostridium perfringens",
]
SHORT_ORGANISMS = [
    r"$L.\ monocytogenes$", r"$E.\ coli$ O157", r"$B.\ thermosphacta$",
    r"$S.$ Typhimurium", r"$S.$ Enteritidis", r"$Y.\ enterocolitica$",
    r"$B.\ cereus$", r"$A.\ hydrophila$", r"$S.\ aureus$",
    r"$L.\ innocua$", r"$C.\ perfringens$",
]

if __name__=="__main__":figure3(pd.read_csv(AUDIT/"censor_safe_matched_points.csv"))
