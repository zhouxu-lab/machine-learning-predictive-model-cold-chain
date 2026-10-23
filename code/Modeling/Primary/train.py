from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Checkpointed nested-source training, with no later-count feedback."""
from common import *
import torch
from torch import nn
import pandas as pd
from models import make_model
DEVICE=torch.device(os.environ.get('COMBASE_DEVICE','cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_num_threads(2)

def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def save_pt(p,data):
    tmp=p.with_suffix('.tmp');torch.save(data,tmp);replace(tmp,p)
def rng_state():
    return dict(numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(r):
    np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])

class TensorData:
    def __init__(self,recs,trids):
        self.recs=recs;self.lookup={int(r['record_id']):i for i,r in enumerate(recs)}
        trset=set(trids);train=[r for r in recs if r['record_id'] in trset]
        self.orgmap={int(v):i+1 for i,v in enumerate(sorted({r['org'] for r in train}))}
        self.matmap={int(v):i+1 for i,v in enumerate(sorted({r['mat'] for r in train}))}
        self.no=len(self.orgmap)+1;self.nm=len(self.matmap)+1;self.nx=len(recs[0]['x'])
        maxm=max(len(r['obs_idx']) for r in recs);n=len(recs)
        oi=np.zeros((n,maxm),np.float32);oy=oi.copy();mask=oi.copy()
        for i,r in enumerate(recs):
            k=len(r['obs_idx']);oi[i,:k]=r['obs_idx'];oy[i,:k]=r['obs_y'];mask[i,:k]=1
        arrays=[np.stack([r['T'] for r in recs]),np.stack([r['x'] for r in recs]),np.array([self.orgmap.get(int(r['org']),0) for r in recs]),np.array([self.matmap.get(int(r['mat']),0) for r in recs]),np.array([r['dt'] for r in recs]),np.array([r['y0'] for r in recs]),oi,oy,mask]
        self.arrays=[torch.as_tensor(x,device=DEVICE) for x in arrays]
        self.counts=np.array([len(r['obs_idx']) for r in recs]);self.durations=np.array([r['dt'] for r in recs])
    def indices(self,ids):return np.array([self.lookup[int(i)] for i in ids])
    def batch(self,idx):
        ii=torch.as_tensor(idx,device=DEVICE);m=int(self.counts[idx].max())
        a=[x[ii] for x in self.arrays]
        a[6]=a[6][:,:m];a[7]=a[7][:,:m];a[8]=a[8][:,:m]
        return a
    def metadata(self):return dict(orgmap=self.orgmap,matmap=self.matmap,no=self.no,nm=self.nm,nx=self.nx)

# Revision v2: fixed absolute elapsed-time grid (h), identical for every record, so the
# input carries no information about when a record ends. Position 0 is the initial count.
ABS_T=np.r_[0.,np.geomspace(.25,30240.,95)].astype(np.float32)
BASE_OF={'mul':'backbone','add':'backbone','mul2':'backbone2'}
ODE_KINDS=('node','backbone','mul','add','backbone2','mul2')
def transform(kind,recs):
    if kind.endswith('_notemp'):
        return [{**r,'T':np.full_like(r['T'],20.)} for r in recs]
    if kind.endswith('_abs'):
        out=[]
        for r in recs:
            tg=np.arange(len(r['T']),dtype=np.float64)*float(r['dt'])
            T=np.interp(ABS_T,tg,r['T']).astype(np.float32)          # carry last value beyond the record
            te=np.asarray(r['obs_idx'],np.float64)*float(r['dt'])     # elapsed time of each later count
            oi=np.interp(te,ABS_T,np.arange(len(ABS_T),dtype=np.float64)).astype(np.float32)
            out.append({**r,'T':T,'obs_idx':oi})
        return out
    return recs

def at_obs(Y,oi):
    low=oi.floor().long().clamp(0,Y.shape[1]-2);w=(oi-low).clamp(0,1)
    return Y.gather(1,low)*(1-w)+Y.gather(1,low+1)*w

def evaluate(model,data,idx,outer,seed):
    model.eval();outs=[]
    order=idx[np.argsort(data.durations[idx],kind='stable')]
    with torch.no_grad():
        for p in range(0,len(order),256):
            ids=order[p:p+256];T,x,o,m,dt,y0,oi,oy,mask=data.batch(ids)
            Y,_=model(T,x,o,m,dt,y0)
            assert torch.isfinite(Y).all(), 'Non-finite trajectory: evaluation cannot drop records'
            pred=at_obs(Y,oi).cpu().numpy()
            for j,i in enumerate(ids):
                r=data.recs[i];n=len(r['obs_time'])
                outs.append(pd.DataFrame(dict(record_id=int(r['record_id']),time_h=r['obs_time'],y_true=r['obs_y'].astype(float)+float(r['y0']),y_pred=pred[j,:n].astype(float)+float(r['y0']),logc0_obs=float(r['y0']),organism=r['organism'],source_key=r['source_key'],coldchain=r['coldchain'],fold=outer,seed=seed)))
    return pd.concat(outs,ignore_index=True)

def run(kind,outer,ci,seed=0,inner=None,epochs_override=None,pilot=False,interrupt_epoch=None):
    ident=identity();cfg=contract()['configs'][kind][ci].copy()
    d=job_dir(kind,outer,ci,seed,inner)
    if pilot:d=ROOT/'pilot'/kind;d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists() and not pilot:return json.loads((d/'complete.json').read_text())
    seed_all(seed)
    recs,_,_=dataset();recs=transform(kind,recs);trids,teids=ids_for(outer,inner)
    if pilot:
        pool=set(trids);recs=[r for r in recs if r['record_id'] in pool]
        # Preserve a range of durations and record types; no outer test record used.
        rg=np.random.default_rng(7);take=rg.choice(len(recs),min(1024,len(recs)),replace=False);recs=[recs[int(i)] for i in take]
        trids=[r['record_id'] for r in recs];teids=trids[:64];cfg['epochs']=1
    data=TensorData(recs,trids);tr=data.indices(trids);te=data.indices(teids)
    if not pilot: assert {recs[i]['source_key'] for i in tr}.isdisjoint({recs[i]['source_key'] for i in te})
    model=make_model(kind,data.no,data.nm,data.nx,cfg,abs_time=ABS_T).to(DEVICE)
    backbone_dependency=None
    if kind in BASE_OF and not pilot:
        base=BASE_OF[kind];broot=RUNS
        selected=json.loads((broot/base/f'fold{outer}'/'selected.json').read_text())
        stage=f'inner{inner}_c{selected["config"]}' if inner is not None else f'final_s{seed}'
        bd=broot/base/f'fold{outer}'/stage
        bstate=torch.load(bd/'model.pt',map_location=DEVICE,weights_only=False)
        assert bstate['preprocess']==data.metadata(), 'Backbone preprocessing mismatch'
        model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in bstate['model'].items() if k.startswith('backbone.')})
        backbone_dependency=sha(bd/'model.pt')
    if kind in BASE_OF:model.freeze_backbone()
    epochs=epochs_override or cfg['epochs']
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'])
    expected=dict(contract_sha256=ident,code=code_identity(),kind=kind,outer=outer,inner=inner,config=ci,seed=seed,epochs=epochs,pilot=pilot,backbone_sha256=backbone_dependency)
    first=0;history=[];best=float('inf');best_epoch=0;elapsed=0.
    if (d/'resume.pt').exists() and not pilot:
        ck=torch.load(d/'resume.pt',map_location=DEVICE,weights_only=False)
        assert ck['identity']==expected,'Resume contract mismatch'
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);restore_rng(ck['rng'])
        first=ck['epoch'];history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];elapsed=ck['seconds']
    ode=kind in ODE_KINDS;bs=contract()['batching']['ode' if ode else 'seq']
    start=time.time()
    for epoch in range(first,epochs):
        model.train();order=tr[np.argsort(data.durations[tr]*(1+.1*np.random.rand(len(tr))))]
        chunks=[order[p:p+bs] for p in range(0,len(order),bs)];np.random.shuffle(chunks)
        sse=0.;nobs=0;skip=0;skipped=[]
        for idx in chunks:
            T,x,o,m,dt,y0,oi,oy,mask=data.batch(idx)
            opt.zero_grad(set_to_none=True);Y,reg=model(T,x,o,m,dt,y0)
            pred=at_obs(Y,oi);ss=((pred-oy).square()*mask).sum();loss=ss/mask.sum()+cfg.get('penalty',0)*reg
            if not torch.isfinite(loss):
                skip+=1;skipped.extend(int(recs[i]['record_id']) for i in idx);continue
            loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(gn):
                skip+=1;skipped.extend(int(recs[i]['record_id']) for i in idx);opt.zero_grad(set_to_none=True);continue
            opt.step();sse+=ss.item();nobs+=int(mask.sum().item())
        assert nobs>0,'No finite training updates'
        sched.step()
        row=dict(epoch=epoch+1,train_mse=sse/nobs,skipped_batches=skip,total_batches=len(chunks),skipped_record_ids=skipped)
        if inner is not None or pilot:
            pred=evaluate(model,data,te,outer,seed);met=metrics(pred.y_true,pred.y_pred);row['validation']=met
            if met['rmse']<best:
                best=met['rmse'];best_epoch=epoch+1
                save_pt(d/'model.pt',dict(model=model.state_dict(),preprocess=data.metadata(),config=cfg,kind=kind,epoch=best_epoch,identity=expected))
                pred.to_parquet(d/'predictions.parquet',index=False)
        history.append(row)
        ck=dict(identity=expected,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),epoch=epoch+1,history=history,best=best,best_epoch=best_epoch,seconds=elapsed+time.time()-start)
        save_pt(d/'resume.pt',ck)
        atomic_json(d/'progress.json',dict(kind=kind,outer=outer,inner=inner,config=ci,seed=seed,epoch=epoch+1,epochs=epochs,seconds=ck['seconds'],train_mse=row['train_mse'],skipped_batches=skip))
        print(kind,outer,inner,ci,seed,'epoch',epoch+1,'mse',round(row['train_mse'],4),'seconds',round(ck['seconds'],1),flush=True)
        if interrupt_epoch==epoch+1:return {'interrupted':True}
    if inner is None and not pilot:
        save_pt(d/'model.pt',dict(model=model.state_dict(),preprocess=data.metadata(),config=cfg,kind=kind,epoch=epochs,identity=expected))
        pred=evaluate(model,data,te,outer,seed);pred.to_parquet(d/'predictions.parquet',index=False);best_epoch=epochs
    else:pred=pd.read_parquet(d/'predictions.parquet')
    result=dict(kind=kind,outer=outer,inner=inner,config=ci,seed=seed,epochs=epochs,best_epoch=best_epoch,metrics=metrics(pred.y_true,pred.y_pred),seconds=elapsed+time.time()-start,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),skipped_batches=sum(x['skipped_batches'] for x in history),total_batches=sum(x['total_batches'] for x in history),history=history,identity=expected)
    atomic_json(d/'complete.json',result);print('COMPLETE',kind,outer,inner,ci,seed,result['metrics'],flush=True);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('outer',type=int);p.add_argument('config',type=int);p.add_argument('--seed',type=int,default=0);p.add_argument('--inner',type=int);p.add_argument('--epochs',type=int);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    run(a.kind,a.outer,a.config,a.seed,a.inner,a.epochs,a.pilot)
