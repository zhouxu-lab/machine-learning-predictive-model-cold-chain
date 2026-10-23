from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Isolated experiment utilities; accepted release is read-only."""
from pathlib import Path
import os, sys, json, hashlib, time, pickle
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = DATA/'Processed/Training'
RUNS = RESULTS/'Training'/'NeuralODE'/'runs'
CONFIG = DATA/'Splits_and_Configurations'/'NeuralODE'
ORIGINAL_BASE = CODE/'Shared'
sys.path.insert(0, str(ORIGINAL_BASE))
os.environ.setdefault('TORCH_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path, data):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
    replace(tmp,p)
def replace(a,b):
    for i in range(30):
        try: os.replace(a,b); return
        except PermissionError:
            if i==29: raise
            time.sleep(.2)
def contract(): return json.loads(((DATA/"Splits_and_Configurations"/'NeuralODE'/'contract.json')).read_text(encoding='utf-8'))
def identity():
    c=contract()
    for p,h in c['input_hashes'].items():
        assert sha(BASE/p)==h, 'Input changed: '+p
    return sha((DATA/"Splits_and_Configurations"/'NeuralODE'/'contract.json'))
def code_identity():
    return {p.name:sha(p) for p in ROOT.glob('*.py') if p.name in ['common.py','models.py','train.py','tree.py']}
def dataset():
    with (BASE/'deep_prepared.pkl').open('rb') as f: return pickle.load(f)
def splits(): return json.loads(((DATA/"Splits_and_Configurations"/'NeuralODE'/'splits.json')).read_text(encoding='utf-8'))
def metrics(true,pred):
    y=np.asarray(true,dtype=float); p=np.asarray(pred,dtype=float)
    assert len(y)==len(p) and np.isfinite(y).all() and np.isfinite(p).all(), 'Invalid predictions'
    e=p-y; ss=np.sum((y-y.mean())**2)
    return dict(n=len(y),sse=float(e@e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(e.mean()),r2=float(1-e@e/ss) if ss>0 else None,within_05=float(np.mean(abs(e)<=.5)),underprediction_gt2=float(np.mean(e < -2)))
def ids_for(outer,inner=None):
    s=splits()[str(outer)]
    return (s['train'],s['test']) if inner is None else (s['inner'][str(inner)]['train'],s['inner'][str(inner)]['valid'])
def job_dir(kind,outer,config,seed,inner=None):
    stage=f'inner{inner}_c{config}' if inner is not None else f'final_s{seed}'
    p=RUNS/kind/f'fold{outer}'/stage
    p.mkdir(parents=True,exist_ok=True)
    return p
