from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Source-nested tree baselines. All preprocessing is fitted on training rows."""
from common import *
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from baseline_gbm import FEATS,CATS

def run(kind,outer,ci,seed,inner=None,pilot=False):
    identity(); d=ROOT/'pilot'/kind if pilot else job_dir(kind,outer,ci,seed,inner)
    if pilot:d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists() and not pilot:return json.loads((d/'complete.json').read_text())
    cfg=contract()['configs'][kind][ci]
    trids,teids=ids_for(outer,inner)
    df=pd.read_parquet(BASE/'features.parquet'); df.source_key=df.source_key.fillna('unk')
    tr=df[df.record_id.isin(trids)].copy();te=df[df.record_id.isin(teids)].copy()
    assert set(tr.source_key).isdisjoint(set(te.source_key))
    for x in [tr,te]:
        for c in CATS:x[c]=x[c].fillna('__UNKNOWN__').astype(str)
    if pilot:
        tr=tr.sample(min(30000,len(tr)),random_state=0);te=tr.iloc[:500].copy()
    start=time.time()
    if kind=='rf':
        prep=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),FEATS),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32),CATS)],sparse_threshold=1.)
        model=Pipeline([('preprocess',prep),('model',RandomForestRegressor(n_estimators=30 if pilot else cfg['trees'],min_samples_leaf=cfg['min_leaf'],max_features=cfg['max_features'],n_jobs=8,random_state=seed))])
        model.fit(tr[FEATS+CATS],tr.target)
        p=model.predict(te[FEATS+CATS])
        complexity=sum(t.tree_.node_count for t in model['model'].estimators_)
    else:
        cat_levels={c:sorted(tr[c].unique().tolist()) for c in CATS}
        for c in CATS:
            tr[c]=pd.Categorical(tr[c],categories=cat_levels[c]);te[c]=pd.Categorical(te[c],categories=cat_levels[c])
        model=LGBMRegressor(objective='regression',n_estimators=100 if pilot else cfg['n_estimators'],learning_rate=cfg['lr'],num_leaves=cfg['num_leaves'],min_child_samples=cfg['min_child_samples'],colsample_bytree=.8,subsample=.8,subsample_freq=1,reg_lambda=1.,random_state=seed,deterministic=True,force_col_wise=True,n_jobs=8,verbosity=-1)
        model.fit(tr[FEATS+CATS],tr.target,categorical_feature=CATS);p=model.predict(te[FEATS+CATS]);complexity=model.booster_.num_trees()
        model={'model':model,'category_levels':cat_levels,'features':FEATS+CATS}
    p=p+te.logc0_obs.to_numpy()
    out=te[['record_id','time_h','logc','logc0_obs','organism','coldchain','source_key']].rename(columns={'logc':'y_true'}).copy()
    out['y_pred']=p;out['fold']=outer;out['seed']=seed
    out.to_parquet(d/'predictions.parquet',index=False)
    joblib.dump(model,d/'model.joblib',compress=3)
    result=dict(kind=kind,outer=outer,inner=inner,config=ci,seed=seed,seconds=time.time()-start,metrics=metrics(out.y_true,out.y_pred),complexity=complexity,training_rows=len(tr),validation_or_test_rows=len(te),pilot=pilot,contract_sha256=sha((DATA/"Splits_and_Configurations"/'Primary'/'contract.json')),code_hashes=code_identity())
    atomic_json(d/'complete.json',result);print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['rf','gbm']);p.add_argument('outer',type=int);p.add_argument('config',type=int);p.add_argument('--seed',type=int,default=0);p.add_argument('--inner',type=int);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    run(a.kind,a.outer,a.config,a.seed,a.inner,a.pilot)
