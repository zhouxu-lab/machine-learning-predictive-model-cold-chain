from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
from figure_export import export
"""Selected Option A, finalized for manuscript Figure 2.

Question: How accurate and transferable are the predictions on the same held out data?
Eight parity panels show observed and predicted microbial concentrations.
Annotations report source-balanced MAE and RMSE on identical held out points.
Repeated observations are not independent replicates; no confidence intervals are shown.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT

from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
                     'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
NAMES = {'gbm':'LightGBM','rf':'Random Forest','node':'Neural Baranyi ODE',
         'transformer_abs':'Transformer','gru_abs':'GRU','mul':'Baranyi–neural hybrid',
         'backbone':'Baranyi backbone','persistence':'Starting count persistence'}
ORDER = ['gbm','rf','node','transformer_abs','gru_abs','mul','backbone','persistence']
POINT = '#477E91'
MEAN = '#BA7134'
def main():
    p = pd.read_csv(RESULTS / 'Predictions/Dynamic/censor_safe_matched_points.csv')
    assert set(p.model) == set(ORDER)
    assert not p.duplicated(['model','record_id','time_h']).any()
    assert p.groupby('model').size().eq(4247).all()
    assert p.groupby('model').record_id.nunique().eq(511).all()
    assert p.groupby('model').source_key.nunique().eq(10).all()
    aligned=p.groupby(['record_id','time_h'])
    assert aligned.model.nunique().eq(8).all()
    # Frozen model exports differ only by float32/float64 storage precision.
    assert (aligned.observed_log.max()-aligned.observed_log.min()).max() < 1e-6
    assert p.groupby(['record_id','time_h']).source_key.nunique().eq(1).all()
    assert np.allclose(p.predicted_log-p.observed_log,p.error_log)
    assert np.allclose(p.error_log.abs(),p.absolute_error)
    p['squared_error'] = p.error_log ** 2
    source = p.groupby(['model','source_key']).agg(
        mae=('absolute_error','mean'),mse=('squared_error','mean'),
        observations=('record_id','size'),records=('record_id','nunique')).reset_index()
    metric = source.groupby('model').agg(source_mae=('mae','mean'),source_mse=('mse','mean'))
    metric['source_rmse'] = np.sqrt(metric.source_mse)
    source.to_csv(RESULTS/'Figure_Source_Data/source_level_errors.csv',index=False)
    metric.to_csv(RESULTS/'Figure_Source_Data/source_balanced_metrics.csv')
    assert abs(metric.loc['gbm','source_mae']-1.114)<.001
    assert abs(metric.loc['rf','source_mae']-1.342)<.001

    # Explicit square plotting areas avoid the wide empty gutters produced by
    # equal-aspect axes in wide subplot slots. Each panel retains its own labels.
    # Row gutters are 0.87 inches; square plot areas remain 1.41 inches wide.
    fig, axes = plt.subplots(4,2,figsize=(4.83,9.0),sharex=True,sharey=True)
    fig.subplots_adjust(left=.40/4.83,right=4.09/4.83,top=8.65/9.0,bottom=.40/9.0,
                        wspace=.87/1.41,hspace=.87/1.41)
    for i,(ax,model) in enumerate(zip(axes.flat,ORDER)):
        d=p[p.model.eq(model)]
        ax.scatter(d.observed_log,d.predicted_log,s=3,alpha=.18,color=POINT,edgecolors='none',rasterized=True)
        ax.plot([-1,11],[-1,11],color='#424A50',ls='--',lw=.8)
        ax.set_xlim(-1,11);ax.set_ylim(-1,11);ax.set_aspect('equal',adjustable='box')
        ax.set_xticks([0,5,10]);ax.set_yticks([0,5,10]);ax.tick_params(labelsize=8,labelbottom=True,labelleft=True,pad=2)
        ax.set_xlabel('Observed (log₁₀ CFU/unit)',fontsize=8,labelpad=2)
        ax.set_ylabel('Predicted (log₁₀ CFU/unit)',fontsize=8,labelpad=0)
        ax.set_title(f'{chr(97+i)}  {NAMES[model]}',loc='left',fontsize=8,fontweight='bold',pad=15)
        m=metric.loc[model]
        ax.text(0,1.025,f'Source MAE {m.source_mae:.2f}  |  RMSE {m.source_rmse:.2f}',
                transform=ax.transAxes,fontsize=7.5,va='bottom')
    # The source cohort includes CFU/g and surface counts in CFU/cm2.
    # 'unit' means the original sampling unit, defined in the figure caption.
    # Match the physical label-to-spine distance on x and y, including the
    # different widths/heights of numeric tick labels. No data are rescaled.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        x_gap = ax.bbox.y0 - ax.xaxis.label.get_window_extent(renderer).y1
        y_gap = ax.bbox.x0 - ax.yaxis.label.get_window_extent(renderer).x1
        ax.yaxis.labelpad += (x_gap - y_gap) * 72 / fig.dpi
    fig.canvas.draw()
    for ax in axes.flat:
        x_gap = ax.bbox.y0 - ax.xaxis.label.get_window_extent(renderer).y1
        y_gap = ax.bbox.x0 - ax.yaxis.label.get_window_extent(renderer).x1
        assert abs(x_gap-y_gap)*72/fig.dpi < .5
    export(fig,'Figure_2',list(axes.flat),tight=False)

if __name__=="__main__":main()
