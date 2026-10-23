from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import numpy as np
import pandas as pd
p=pd.read_csv(RESULTS/'Predictions/Dynamic/censor_safe_matched_points.csv')
w=p.groupby(['source_key','model']).absolute_error.mean().unstack('model')
assert len(w)==10 and w.notna().all().all()
rng=np.random.default_rng(20260930)
draw=rng.integers(0,len(w),size=(50000,len(w)))
rows=[]
for a,b in [('gbm','rf'),('mul','persistence')]:
    delta=(w[a]-w[b]).to_numpy()
    lo,hi=np.quantile(delta[draw].mean(axis=1),[.025,.975])
    rows.append(dict(model=a,reference=b,source_balanced_difference=delta.mean(),lower_95=lo,upper_95=hi,sources=len(w),sources_with_lower_error=int((delta<0).sum())))
pd.DataFrame(rows).to_csv(RESULTS/'Summary_Tables/paired_model_intervals.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
