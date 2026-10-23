from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Reproducible timestamp correction from preserved original record measurements.

The mean of reported log counts at each record/time is the benchmark target.
All individual values and their original ordering remain in raw_tables/counts.
QC checks the original individual count range, so averaging cannot hide an
out-of-range measurement. Temperature order and same-time jumps are retained.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from models_common import build_dynamic_targets,source_folds

ROOT=DATA/'Processed/Preparation'
RAW=ROOT/'data/raw_tables';OUT=ROOT/'data/curated'


def main():
    if (ROOT/'data_manifest.json').exists():
        raise RuntimeError('Corrected dataset is frozen; use a new version to change it')
    r=pd.read_parquet(RAW/'records_base.parquet')
    old=r.copy()
    computed=['n_pts','t_min','t_max','logc_min','logc_max','logc_first','logc_last','delta_logc','qc_pass',
              'T_min','T_max','T_mean','T_range','n_T_points','T_first','T_last','response_type','coldchain',
              'coldchain_strict','has_abuse','coldchain_rte']
    r=r.drop(columns=computed)
    raw=pd.read_parquet(RAW/'counts.parquet')
    assert len(raw)==420132 and r.record_id.nunique()==len(r)==61035
    g=raw.groupby(['record_id','time_h'],sort=True).logc
    c=g.agg(logc='mean',n_measurements='size',logc_sd='std',logc_min_reported='min',logc_max_reported='max').reset_index()
    q=c.groupby('record_id').agg(n_pts=('time_h','size'),t_min=('time_h','min'),t_max=('time_h','max'),
        logc_min=('logc','min'),logc_max=('logc','max'),logc_first=('logc','first'),logc_last=('logc','last'),
        raw_logc_min=('logc_min_reported','min'),raw_logc_max=('logc_max_reported','max'),n_raw_counts=('n_measurements','sum'))
    q['delta_logc']=q.logc_last-q.logc_first
    q['qc_pass']=(q.n_pts>=3)&(q.t_min>=0)&(q.t_max>0)&(q.raw_logc_min>-3)&(q.raw_logc_max<13)
    r=r.merge(q,on='record_id',how='left',validate='one_to_one')
    r['qc_pass']=r.qc_pass.fillna(False).astype(bool)
    pro=pd.read_parquet(RAW/'profiles.parquet').sort_values(['record_id','variable','time_h','point_index'],kind='stable')
    t=pro[pro.variable.eq('Temperatures')].groupby('record_id').agg(T_min=('value','min'),T_max=('value','max'),
        T_mean=('value','mean'),n_T_points=('value','size'),T_first=('value','first'),T_last=('value','last'))
    t['T_range']=t.T_max-t.T_min
    r=r.merge(t,on='record_id',how='left',validate='one_to_one')
    constant=r.kind.eq('static') | (r.kind.eq('dynamic')&r.T_mean.isna()&r.temperature.notna())
    for col in ['T_min','T_max','T_mean','T_first','T_last']:r.loc[constant,col]=r.loc[constant,'temperature']
    r.loc[constant,'T_range']=0.
    r['response_type']=np.select([r.delta_logc>.5,r.delta_logc<-.5],['growth','inactivation'],'no_change')
    r['coldchain']=r.kind.eq('dynamic')&r.qc_pass&(r.T_min<=10)&(r.T_max<45)
    r['coldchain_strict']=r.coldchain&(r.T_max<=15)
    r['has_abuse']=r.coldchain&(r.T_range>=5)
    text=(r.food_name.fillna('')+' | '+r.conditions_text.fillna('')+' | '+r.comment.fillna('')).str.lower()
    processing=text.str.contains(r'ferment|ripen|matur|salam|sausage|cured|curing|dried|dry |drying|lardo|prosciutto|speck|coppa|bresaola|pancetta|starter|sourdough|kimchi|yogurt|yoghurt|kefir',regex=True)
    liquid=r.food_name.fillna('').str.contains(r'preserving liquid|processing whey',case=False,regex=True)
    r['chilled_foods']=r.coldchain&~processing&(r.T_max<=25)&(r.T_min>=-5)&~r.matrix_is_broth&~liquid
    r['coldchain_rte']=r.chilled_foods # compatibility alias only, never label as ready-to-eat
    r['chilled_food_exclusion']=np.where(r.matrix_is_broth,'culture medium',np.where(liquid,'food-associated processing/preserving liquid',np.where(processing,'processing keyword',np.where(~r.chilled_foods,'temperature/QC criteria','included'))))
    c=c[c.record_id.isin(r.loc[r.qc_pass,'record_id'])]
    for name,d in [('records',r),('curves',c),('profiles',pro),('coldchain',r[r.coldchain])]:d.to_parquet(OUT/f'{name}.parquet',index=False)
    targets=build_dynamic_targets(r,c,pro)
    folds=targets.drop_duplicates('record_id')[['record_id','source_key','coldchain']].copy()
    folds['fold']=source_folds(folds.source_key)
    folds.to_parquet(ROOT/'folds.parquet',index=False)
    r[r.chilled_foods][['record_id','food_category','food_name','source','source_key']].to_csv(ROOT/'chilled_food_inventory.csv',index=False)
    counts={'all_records':len(r),'raw_counts':len(raw),'curated_count_times':len(c),
        'qc_static':int((r.kind.eq('static')&r.qc_pass).sum()),'qc_dynamic':int((r.kind.eq('dynamic')&r.qc_pass).sum()),
        'evaluated_dynamic_records':len(folds),'evaluated_dynamic_observations':len(targets),
        'coldchain_records':int(r.coldchain.sum()),'coldchain_observations':int(targets.coldchain.sum()),
        'chilled_food_records':int(r.chilled_foods.sum()),'chilled_food_observations':int(targets.chilled_foods.sum()),
        'qc_changed_records':int((r.set_index('record_id').qc_pass!=old.set_index('record_id').qc_pass).sum()),
        'response_changed_records':int((r.set_index('record_id').response_type!=old.set_index('record_id').response_type).sum()),
        'fold_record_counts':{str(k):int(v) for k,v in folds.fold.value_counts().sort_index().items()},
        'dynamic_source_groups':int(folds.source_key.nunique()),'coldchain_source_groups':int(folds[folds.coldchain].source_key.nunique()),
        'chilled_food_categories':r[r.chilled_foods].food_category.value_counts().to_dict()}
    (ROOT/'curation_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    files=[*OUT.glob('*.parquet'),ROOT/'folds.parquet',ROOT/'group_map.jsonl']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
