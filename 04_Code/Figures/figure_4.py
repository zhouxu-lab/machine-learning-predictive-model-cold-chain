from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
from figure_export import export
"""Three review-only layouts for matched initial-count sensitivity results."""
from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
ROOT = PACKAGE_ROOT
OUT = RESULTS/'Figure_Source_Data'
SOURCE = RESULTS/'Predictions/Dynamic/measured_anchor_matched_metrics.csv'
sys.path.insert(0, str(ROOT))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
MODELS = ["rf", "gbm", "gru_abs", "transformer_abs", "mul", "backbone", "node"]
LABELS = ["Random Forest", "LightGBM", "GRU", "Transformer",
          "Baranyi–neural hybrid", "Baranyi backbone", "Neural Baranyi ODE"]
SHORT = ["Random Forest", "LightGBM", "GRU", "Transformer",
         "Hybrid", "Baranyi backbone", "Neural Baranyi ODE"]
APPROACHES = ["assumed_start_0.5", "assumed_start_1.0", "assumed_start_1.5",
              "assumed_start_2.0", "first_later_measured_count"]
ASSUMED = np.array([0.5, 1.0, 1.5, 2.0])
COLORS = ["#c7dce8", "#91b9cd", "#4f8fb0", "#1c5b7d"]
ANCHOR = "#bc6c35"
INK = "#183c52"
MUTED = "#536a78"
def load_data():
    raw = pd.read_csv(SOURCE)
    sub = raw.loc[raw.cohort.eq("all_temperature_complete_food")].copy()
    assert len(sub) == 35 and not sub.duplicated(["model", "approach"]).any()
    assert sub[["records", "later_points", "sources"]].drop_duplicates().values.tolist() == [[266, 1194, 6]]
    frame = sub.pivot(index="model", columns="approach", values="source_balanced_mae").loc[MODELS, APPROACHES]
    assert np.isfinite(frame.to_numpy()).all()
    sub.loc[:, ["model", "approach", "records", "later_points", "sources", "source_balanced_mae"]].to_csv(
        OUT / "source_data.csv", index=False
    )
    return frame.to_numpy()
def style():
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
def option_a(values):
    """Performance curves; the measured anchor is visually disconnected."""
    fig, ax = plt.subplots(figsize=(7.17, 4.05), facecolor="white")
    fig.subplots_adjust(left=.13, right=.98, top=.96, bottom=.31)
    palette = ["#22577a", "#008b8b", "#a25535", "#815b9c", "#387c4c", "#a06d23", "#596b91"]
    ax.axvspan(-.48, .13, color="#f6f2ed", zorder=0)
    ax.axvline(.13, color="#b4aaa0", linewidth=1.0, zorder=1)
    for i, (label, color) in enumerate(zip(SHORT, palette)):
        ax.plot([-.22, .5], [values[i, 4], values[i, 0]],
                linestyle="--", linewidth=1.35, color=color, alpha=.72, zorder=2)
        ax.plot(ASSUMED, values[i, :4], marker="o", markersize=5.2,
                linewidth=2.0, color=color, label=label, zorder=3)
        ax.scatter([-.22], [values[i, 4]], marker="D", s=30,
                   facecolor=color, edgecolor="white", linewidth=.55, zorder=4)
    ax.set_xlim(-.48, 2.17)
    ax.set_ylim(1.35, 1.96)
    ax.set_xticks([-.22, .5, 1, 1.5, 2],
                  ["Skip non-detect;\nuse next\nmeasured count", "0.5", "1.0", "1.5", "2.0"])
    ax.set_yticks([1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    ax.grid(axis="y", color="#e3e9ed", linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("MAE (log₁₀ units)", color=INK)
    fig.text(.68, .195, "Assumed initial concentration (log₁₀ CFU/g)",
             ha="center", fontsize=8.2, color=INK)
    fig.legend(loc="lower center", bbox_to_anchor=(.53, .015), ncol=4,
               frameon=False, fontsize=7.4, handlelength=1.8, columnspacing=1.7)
    export(fig,'Figure_4')

if __name__=="__main__":
    style()
    option_a(load_data())
