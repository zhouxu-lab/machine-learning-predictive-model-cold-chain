from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import archive_labels as b
from figure_export import export
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
from matplotlib.patches import Patch
out=RESULTS/'Figure_Source_Data'
t=pd.read_csv(out/'Figure_S2_archive_organism_source_data.csv',index_col=0)
assert len(t)==52 and t.Total.sum()==61035
fig,axes=plt.subplots(1,2,figsize=(6.7,7.0))
fig.subplots_adjust(left=.285,right=.995,top=.96,bottom=.075,wspace=1.4)
for ax,part,label in zip(axes,[t.iloc[:26],t.iloc[26:]],['a','b']):
 b.draw(ax,part,'organism',label,size=6.1)
 ax.tick_params(axis='x',labelsize=6)
 ax.set_xlabel('Records',fontsize=7)
 ax.title.set_fontsize(8)
 for x in ax.texts:
  x.set_fontsize(6.2)
  x.set_x(x.get_position()[0]+float(part.Total.max())*.015)
axes[0].legend(handles=[Patch(facecolor=c,label=k) for c,k in zip(b.COLORS,['Constant temperature','Changing temperature','Temperature unavailable'])],loc='center right',bbox_to_anchor=(1.18,.45),fontsize=5.8,ncol=1,frameon=False,handlelength=1.0,labelspacing=.7)
fig.canvas.draw()
b.require_matplotlib_panel_alignment(fig,axes=axes,strict=True,json_out=str(out/'Figure_S2.alignment.json'))
export(fig,'Figure_S2',tight=False)
print(out)
