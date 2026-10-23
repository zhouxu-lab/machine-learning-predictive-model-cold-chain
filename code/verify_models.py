from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
import subprocess
import json
import hashlib
import numpy as np
if len(sys.argv)==1:
    for family in ['Primary','NeuralODE']:
        subprocess.run([sys.executable,'-B',__file__,family],check=True)
    raise SystemExit(0)
family=sys.argv[1]
sys.path.insert(0,str(CODE/'Modeling'/family))
from common import identity,dataset,splits,contract
from train import TensorData
from models import make_model
import torch
identity()
records,_,_=dataset()
sources={r['record_id']:r['source_key'] for r in records}
for outer in splits().values():
    assert {sources[i] for i in outer['train']}.isdisjoint({sources[i] for i in outer['test']})
    for inner in outer['inner'].values():
        assert {sources[i] for i in inner['train']}.isdisjoint({sources[i] for i in inner['valid']})
config=contract();split_path=DATA/'Splits_and_Configurations'/family/'splits.json'
assert hashlib.sha256(split_path.read_bytes()).hexdigest()==config['splits_sha256']
short=sorted(records,key=lambda r:r['dt'])[:2]
for kind in (['gru_abs','transformer_abs','backbone','mul'] if family=='Primary' else ['node']):
    sample=short
    kw={}
    if family=='Primary':
        from train import transform,ABS_T
        sample=transform(kind,sample)
        if kind.endswith('_abs'):kw['abs_time']=ABS_T
    data=TensorData(sample,[r['record_id'] for r in sample])
    model=make_model(kind,data.no,data.nm,data.nx,config['configs'][kind][0],**kw).to(data.arrays[0].device)
    model.eval()
    with torch.no_grad():
        args=data.batch(np.arange(len(sample)))
        pred,penalty=model(*args[:6])
    assert pred.shape==(2,96) and torch.isfinite(pred).all()
    assert torch.allclose(pred[:,0],torch.zeros(2,device=pred.device),atol=1e-6)
report={'family':family,'records':len(records),'input_and_split_hashes':'passed','source_separation':'passed','forward_passes':'passed','scope':'Input, split and model-execution checks; not independent retraining'}
out=RESULTS/'Verification';out.mkdir(exist_ok=True)
(out/(family+'_model_checks.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report))
