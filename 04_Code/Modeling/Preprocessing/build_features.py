from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import pickle
from models_common import load
from baseline_gbm import build
from deep_models import prepare
rec,cur,pro=load()
out=DATA/'Processed/Preparation'
build(rec,cur,pro,kinds=("static","dynamic")).to_parquet(out/'features.parquet',index=False)
with (out/'deep_prepared.pkl').open('wb') as f:pickle.dump(prepare(rec,cur,pro),f)
print('Prepared feature and neural inputs; next run prepare_clean_training_inputs.py.')
