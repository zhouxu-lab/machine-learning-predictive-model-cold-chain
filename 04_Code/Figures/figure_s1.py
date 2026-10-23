from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
from figure_export import export
"""Build Figures 1–3 from the saved censor-safe held-out predictions."""
from pathlib import Path
import json
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
                     'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
COLORS = {'gbm':'#008b8b','rf':'#22577a','transformer_abs':'#815b9c',
          'gru_abs':'#a25535','node':'#596b91','mul':'#387c4c',
          'backbone':'#a06d23','persistence':'#a3a7aa'}
NAMES = {'gbm':'LightGBM','rf':'Random Forest','transformer_abs':'Transformer',
         'gru_abs':'GRU','node':'Neural Baranyi ODE','mul':'Baranyi–neural hybrid',
         'backbone':'Baranyi backbone','persistence':'Starting-count persistence'}
def figure2(metrics):
    d=metrics[metrics.cohort.eq('expanded')].set_index('model')
    order=['gbm','rf','transformer_abs','gru_abs','node','mul','backbone','persistence']
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    y=np.arange(len(order))
    for metric,offset,marker,label in [('mae',-.13,'o','Pooled MAE'),('source_balanced_mae',.13,'D','Source-balanced MAE')]:
        for i,m in enumerate(order):
            ax.scatter(d.loc[m,metric],i+offset,marker=marker,s=34,
                       color=COLORS[m],edgecolor='white',linewidth=.4,zorder=4)
    for i,m in enumerate(order):
        ax.plot([d.loc[m,'mae'],d.loc[m,'source_balanced_mae']],[i-.13,i+.13],
                color=COLORS[m],lw=1.15,alpha=.8)
    ax.set_yticks(y,[NAMES[m] for m in order]);ax.invert_yaxis()
    ax.set_xlabel('Mean absolute error (log₁₀ units)')
    ax.set_xlim(.85,2.2);ax.grid(axis='x',color='#e1e3e5');ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',color='gray',linestyle='none',label='Pooled MAE'),
                       Line2D([],[],marker='D',color='gray',linestyle='none',label='Source-balanced MAE')],
              loc='upper center',bbox_to_anchor=(.52,-.13),ncol=2,frameon=False,fontsize=7)
    export(fig,'Figure_S1')
if __name__=="__main__":figure2(pd.read_csv(RESULTS/"Predictions/Dynamic/censor_safe_model_metrics.csv"))
