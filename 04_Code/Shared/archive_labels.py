from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Full archived ComBase coverage; temperature categories derived from reported data."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent

from audit_panel_alignment import require_matplotlib_panel_alignment
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
                     'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'savefig.facecolor': 'white'})
KINDS = ['Constant / single value', 'Variable temperature profile', 'Temperature unavailable']
COLORS = ['#537F91', '#B87936', '#D4DADF']
FOOD_NAMES = {
    'Vegetable or fruit and their products': 'Vegetable/fruit products',
    'Other, mixed, uncategorised or unknown type of food ': 'Other/mixed/unknown food',
    'Other, mixed, uncategorised or unknown type of food': 'Other/mixed/unknown food',
    'Other or unknown type of dairy': 'Other/unknown dairy',
    'Other or unknown type of meat': 'Other/unknown meat',
    'Infant_food': 'Infant food',
}
GENERA = {'Listeria','Escherichia','Bacillus','Clostridium','Aeromonas','Staphylococcus',
          'Yersinia','Brochothrix','Anoxybacillus','Carnobacterium','Alicyclobacillus',
          'Geobacillus','Shewanella','Photobacterium','Pediococcus','Enterobacter',
          'Serratia','Leuconostoc','Hafnia','Rahnella'}
def italic(text):
    return r'$\it{' + text.replace(' ', r'\ ') + '}$'
def organism_label(name):
    if name == 'Listeria monocytogenes/innocua':
        return italic('L. monocytogenes') + ' / ' + italic('L. innocua')
    if name.startswith('Halophilic bacteria'):
        return 'Halophilic bacteria (incl. ' + italic('S. xylosus') + ')'
    if name.startswith('Campylobacteraceae'):
        return 'Campylobacteraceae (incl. ' + italic('C. jejuni') + ')'
    if name.startswith('Shigellae'):
        return 'Shigellae (incl. ' + italic('S. flexneri') + ')'
    if name == 'Bacillus spoilage bacteria':
        return italic('Bacillus') + ' spoilage bacteria'
    parts = name.split()
    if len(parts) == 2 and parts[1] in ('spp', 'spp.', 'species'):
        return italic(parts[0]) + ' spp.'
    if parts[0] in GENERA and len(parts) >= 2:
        label = italic(parts[0][0] + '. ' + parts[1])
        return label + (' ' + ' '.join(parts[2:]) if len(parts) > 2 else '')
    return name
def source_counts(records, field):
    result = records.groupby([field, 'temperature_class']).size().unstack(fill_value=0)
    result = result.reindex(columns=KINDS, fill_value=0)
    result['Total'] = result.sum(axis=1)
    return result.sort_values('Total', ascending=False, kind='stable')
def draw(ax, table, field, title, size=7.8):
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for kind, color in zip(KINDS, COLORS):
        values = table[kind].to_numpy()
        ax.barh(y, values, left=left, color=color, height=.72)
        left += values
    labels = [FOOD_NAMES.get(x, x) if field == 'food_category' else organism_label(x)
              for x in table.index]
    ax.set_yticks(y, labels, fontsize=size)
    ax.invert_yaxis()
    high = float(table.Total.max())
    ax.set_xlim(0, high * 1.19)
    ax.tick_params(axis='y', length=2)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_xlabel('Records', fontsize=8)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=9, pad=6)
    for i, total in enumerate(table.Total):
        ax.text(total + high * .018, i, f'{total:,}', va='center', fontsize=7.4)
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
