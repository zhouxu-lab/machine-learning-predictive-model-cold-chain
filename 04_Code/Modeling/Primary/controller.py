from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Durable, bounded local scheduler for approved experiments. No external actions."""
from common import *
import subprocess,msvcrt,traceback

KINDS=['backbone','gru_abs','transformer_abs','mul','rf','gbm']
def pool(kind):return 'tree' if kind in ('rf','gbm') else ('cpu' if kind in ('mul',) else 'gpu')
def completed(kind,k,ci,seed,inner):return (job_dir(kind,k,ci,seed,inner)/'complete.json').exists()
def failed(kind,k,ci,seed,inner):return (job_dir(kind,k,ci,seed,inner)/'failed.json').exists()
def read_scheduler_json(path):
    # Windows can briefly deny a read while a worker publishes an atomic file.
    # Retry the read; do not change selection data or treat it as a failed fit.
    for attempt in range(30):
        try:return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==29:raise
            time.sleep(.2)
def select(kind,k):
    dest=RUNS/kind/f'fold{k}'/'selected.json'
    if dest.exists():return read_scheduler_json(dest)
    scores=[]
    for ci in range(len(contract()['configs'][kind])):
        vals=[]
        for j in range(2):
            p=job_dir(kind,k,ci,0,j)/'complete.json'
            if not p.exists():return None
            vals.append(read_scheduler_json(p))
        sse=sum(v['metrics']['sse'] for v in vals);n=sum(v['metrics']['n'] for v in vals)
        scores.append(dict(config=ci,rmse=(sse/n)**.5,epoch=max(1,int(np.floor(np.mean([v.get('best_epoch',1) for v in vals])+.5)))))
    best=min(scores,key=lambda x:(x['rmse'],x['config']))
    result={**best,'candidates':scores,'selection_data':'inner source-held-out dynamic validation only','contract_sha256':sha((DATA/"Splits_and_Configurations"/'Primary'/'contract.json'))}
    atomic_json(dest,result);return result

def candidates(active):
    ids={x['key'] for x in active};pending=[]
    for k in range(5):
        for kind in KINDS:
            # A complete base outer fold enables correction fitting and shared frozen weights.
            if kind in ('mul','add'):
                bs=select('backbone',k)
                if bs is None or not all(completed('backbone',k,bs['config'],s,None) for s in [0,1,2]):continue
            sel=select(kind,k)
            todo=[(ci,0,j,None) for ci in range(4) for j in range(2)] if sel is None else [(sel['config'],s,None,sel['epoch']) for s in [0,1,2]]
            for ci,seed,inner,ep in todo:
                key=(kind,k,ci,seed,inner)
                if key in ids or completed(*key) or failed(*key):continue
                pending.append(dict(key=key,kind=kind,outer=k,config=ci,seed=seed,inner=inner,epochs=ep,pool=pool(kind)))
    return pending

def main():
    identity();assert (ROOT/'qa'/'checks.json').exists(),'Preflight tests missing'
    lock=(ROOT/'controller.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    atomic_json(ROOT/'controller_identity.json',dict(pid=os.getpid(),started=time.time(),contract=identity(),code=code_identity(),controller_sha256=sha(ROOT/'controller.py'),capacities={'tree':1,'gpu':1,'cpu':4}))
    active=[];limits={'tree':1,'gpu':1,'cpu':4}
    while True:
        for item in list(active):
            rc=item['proc'].poll()
            if rc is None:continue
            item['log'].close();active.remove(item)
            if rc!=0 or not completed(*item['key']):
                atomic_json(item['dir']/'failed.json',dict(returncode=rc,key=item['key'],time=time.time(),log=str(item['dir']/'log.txt')))
        try:jobs=candidates(active)
        except PermissionError as exc:
            print('WAIT for scheduler file access:',exc,flush=True)
            time.sleep(5)
            continue
        for item in jobs:
            if sum(x['pool']==item['pool'] for x in active)>=limits[item['pool']]:continue
            kind,k,ci,seed,inner=item['key'];d=job_dir(kind,k,ci,seed,inner)
            cmd=[sys.executable,str(ROOT/('tree.py' if item['pool']=='tree' else 'train.py')),kind,str(k),str(ci),'--seed',str(seed)]
            if inner is not None:cmd+=['--inner',str(inner)]
            elif item['pool']!='tree':cmd+=['--epochs',str(item['epochs'])]
            env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['COMBASE_DEVICE']='cpu' if item['pool']=='cpu' else 'cuda'
            log=(d/'log.txt').open('a',encoding='utf-8')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            active.append({**item,'proc':proc,'log':log,'dir':d,'start':time.time()})
            print('START',item['key'],'pid',proc.pid,flush=True)
        statuses={}
        for kind in KINDS:
            statuses[kind]={'inner_complete':len(list((RUNS/kind).glob('fold*/inner*/complete.json'))),'final_complete':len(list((RUNS/kind).glob('fold*/final*/complete.json'))),'failures':len(list((RUNS/kind).glob('fold*/*/failed.json')))}
        running=[{k:v for k,v in x.items() if k not in ['proc','log','dir']}|{'pid':x['proc'].pid} for x in active]
        done=all(v['final_complete']==15 for v in statuses.values())
        atomic_json(ROOT/'status.json',dict(state='training_complete' if done else 'running',updated=time.time(),models=statuses,active=running,controller_pid=os.getpid()))
        if done:
            print('ALL TRAINING COMPLETE',flush=True)
            if (ROOT/'analyse.py').exists():subprocess.run([sys.executable,str(ROOT/'analyse.py')],cwd=ROOT,check=True)
            break
        if not active and not jobs:
            atomic_json(ROOT/'status.json',dict(state='needs_attention',updated=time.time(),models=statuses,active=[],controller_pid=os.getpid()))
            raise RuntimeError('Unfinished experiment has no runnable jobs; inspect failed.json')
        time.sleep(5)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'controller_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
