from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Figure 1 label revision; original counts and category assignments retained."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import archive_labels as base
from figure_export import export
HERE=RESULTS/'Figure_Source_Data'
food=pd.read_csv(HERE/'Figure_1_archive_food_source_data.csv',index_col=0)
organisms=pd.read_csv(HERE/'Figure_1_archive_organism_source_data.csv',index_col=0)
assert food.Total.sum()==61035
assert organisms.Total.sum()==56697
fig,axes=plt.subplots(2,1,figsize=(7.25,9.0))
fig.subplots_adjust(left=.445,right=.985,top=.975,bottom=.05,hspace=.25)
labels=['Constant temperature','Changing temperature','Temperature unavailable']
axes[0].legend(handles=[Patch(facecolor=c,label=k) for k,c in zip(labels,base.COLORS)],loc='center right',bbox_to_anchor=(.98,.54),ncol=1,frameon=False,fontsize=7.6,handlelength=1.5,labelspacing=.8)
base.draw(axes[0],food,'food_category','A  Food')
base.draw(axes[1],organisms,'organism','B  Organism')
export(fig,'Figure_1',axes)
