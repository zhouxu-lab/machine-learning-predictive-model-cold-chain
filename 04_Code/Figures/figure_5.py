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
def figure5():
    data = pd.read_csv(RELIANCE)
    assert set(data.group) == set(FULL_GROUPS)
    assert set(data.organism) == set(ORGANISMS)
    assert set(data.model) == {"LightGBM", "Random Forest"}
    assert len(data) == 2 * len(FULL_GROUPS) * len(ORGANISMS)
    data.to_csv(HERE / "Figure_5_source_data.csv", index=False)
    cmap = LinearSegmentedColormap.from_list(
        "loss_change", [(0, "#B87536"), (.2, "#F1D4B2"), (.5, "#F7F7F5"),
                        (.68, "#A9C9D6"), (1, "#164B6B")])
    norm = TwoSlopeNorm(vmin=-.15, vcenter=0, vmax=.6)
    fig = plt.figure(figsize=(7.2, 7.75))
    gs = fig.add_gridspec(2, 1, left=.215, right=.985, top=.975, bottom=.135,
                          hspace=.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    for ax, model, letter in zip(axes, ["LightGBM", "Random Forest"], ["a", "b"]):
        p = data[data.model.eq(model)].pivot(index="organism", columns="group", values="delta_mae")
        values = p.loc[ORGANISMS, FULL_GROUPS].to_numpy()
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
        ax.set_yticks(range(len(ORGANISMS)), SHORT_ORGANISMS, fontsize=6.9)
        ax.set_xticks(range(len(FULL_GROUPS)))
        if model == "Random Forest":
            ax.set_xticklabels(SHORT_GROUPS, fontsize=6.8)
            ax.tick_params(axis="x", pad=4)
        else:
            ax.set_xticklabels([])
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(FULL_GROUPS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ORGANISMS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f"{letter}  {model}", loc="left", fontsize=8,
                     fontweight="bold", pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.23, .057, .74, .015])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    colorbar.set_ticks([-.15, -.1, 0, .1, .3, .6])
    colorbar.set_ticklabels(["−0.15", "−0.10", "0", "0.10", "0.30", "≥0.60"])
    colorbar.ax.tick_params(labelsize=6.7, length=2)
    colorbar.set_label("Change in MAE after shuffling (log₁₀ units)",
                       fontsize=7.1, labelpad=3)
    out = HERE / "Figure_5_input_reliance_refined"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=axes, exclude_axes=[cax], json_out=str(out) + ".alignment.json", strict=True)
    export(fig,'Figure_5')

if __name__=="__main__":figure5()
