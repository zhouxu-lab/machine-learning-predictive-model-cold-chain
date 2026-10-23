from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import pandas as pd
import numpy as np
from archive_labels import KINDS,source_counts
r=pd.read_parquet(DATA/'Processed/Archive/records.parquet')
p=pd.read_parquet(DATA/'Processed/Archive/profiles.parquet')
t=p[p.variable.eq('Temperatures')].groupby('record_id').value.agg(['min','max'])
assert len(r)==61035 and len(t)==6512
r['temperature_class']=np.select([r.record_id.isin(t.index),r.temperature.notna()],[KINDS[1],KINDS[0]],default=KINDS[2])
assert r.temperature_class.value_counts().reindex(KINDS).tolist()==[49527,6512,4996]
out=RESULTS/'Figure_Source_Data'
source_counts(r,'food_category').to_csv(out/'Figure_1_archive_food_source_data.csv')
a=source_counts(r,'organism');a.to_csv(out/'Figure_S2_archive_organism_source_data.csv');a.head(20).to_csv(out/'Figure_1_archive_organism_source_data.csv')
