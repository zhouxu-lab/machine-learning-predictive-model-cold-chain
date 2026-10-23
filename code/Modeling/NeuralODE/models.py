from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'data').is_dir())
sys.path.insert(0,str(_PACKAGE/'code'))
sys.path.insert(0,str(_PACKAGE/'code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Independent models for the approved experiment; no author code copied."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import deep_models as old

LN10=math.log(10.)

class Cov(old.Cov):
    def __init__(self,no,nm,nx,d=32):
        super().__init__(no,nm,nx,d)
        self.eo.padding_idx=0; self.em.padding_idx=0
        with torch.no_grad(): self.eo.weight[0].zero_(); self.em.weight[0].zero_()

class Sequence(nn.Module):
    def __init__(self,kind,no,nm,nx,width):
        super().__init__(); self.kind=kind; self.cov=Cov(no,nm,nx)
        if kind=='transformer':
            self.inp=nn.Linear(35,width); self.pos=nn.Parameter(torch.randn(1,96,width)*.02)
            layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=2*width,dropout=.1,batch_first=True)
            self.enc=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.register_buffer('causal',torch.triu(torch.full((96,96),float('-inf')),diagonal=1))
        else:
            cls={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}[kind]
            self.rec=cls(35,width,batch_first=True)
        self.out=nn.Sequential(nn.Linear(width,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,T,x,org,mat,dt,y0):
        c=self.cov(x,org,mat); g=T.shape[1]
        inp=torch.cat((c[:,None,:].expand(-1,g,-1),((T-20)/20)[:,:,None],torch.log1p(dt)[:,None,None].expand(-1,g,1),(y0/5)[:,None,None].expand(-1,g,1)),-1)
        if self.kind=='transformer': h=self.enc(self.inp(inp)+self.pos,mask=self.causal)
        else: h,_=self.rec(inp)
        z=self.out(h)[:,:,0]
        return z-z[:,:1], z.new_zeros(())

class ParametricBackbone(nn.Module):
    def __init__(self,no,nm,nx):
        super().__init__(); self.no=no;self.nm=nm; self.context_dim=nx+no+nm
        self.rate=nn.Linear(self.context_dim+3,2)
        self.capacity=nn.Linear(self.context_dim+1,1)
        self.lag=nn.Linear(self.context_dim,1)
        for layer in [self.rate,self.capacity,self.lag]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.rate.bias.copy_(torch.tensor([-5.5,-8.5]));self.capacity.bias.fill_(2.);self.lag.bias.fill_(-2.)
    def fields(self,T,x,org,mat,dt,y0):
        ctx=torch.cat((x,F.one_hot(org,self.no).to(x.dtype),F.one_hot(mat,self.nm).to(x.dtype)),-1)
        tn=(T-20)/20
        change=(torch.gradient(T,dim=1)[0]/dt[:,None]*10).clamp(-50,50)/50
        z=self.rate(torch.cat((ctx[:,None,:].expand(-1,T.shape[1],-1),tn[:,:,None],tn.square()[:,:,None],change[:,:,None]),-1))
        mu,kd=(3*torch.sigmoid(z)).unbind(-1)
        cap=LN10*(F.softplus(self.capacity(torch.cat((ctx,y0[:,None]),-1))[:,0])+1)
        lag=self.lag(ctx)[:,0]
        return mu,kd,cap,lag,ctx

class CurrentBackbone(old.NeuralBaranyi):
    def __init__(self,no,nm,nx):
        super().__init__(no,nm,nx,history=False);self.cov=Cov(no,nm,nx);self.context_dim=32
    def fields(self,T,x,org,mat,dt,y0):
        ctx=self.cov(x,org,mat)
        cap=LN10*(F.softplus(self.ymax(torch.cat((ctx,y0[:,None]),-1))[:,0])+1)
        lag=self.q0(ctx)[:,0]
        change=(torch.gradient(T,dim=1)[0]/dt[:,None]*10).clamp(-50,50)
        mu,kd=self.rates(ctx,T,change)
        return mu,kd,cap,lag,ctx

def log_exprel(z):
    """log((exp(z)-1)/z), finite with finite derivatives near zero and at large |z|."""
    pos=z.clamp(min=1e-4)
    neg=(-z).clamp(min=1e-4)
    a=pos+torch.log(-torch.expm1(-pos))-torch.log(pos)
    b=torch.log(-torch.expm1(-neg))-torch.log(neg)
    small=z/2+z.square()/24-z.pow(4)/2880
    return torch.where(z.abs()<1e-3,small,torch.where(z>0,a,b))

def logistic_flow(y,a,b,cap,h):
    """Exact frozen-coefficient flow of y'=a-b*exp(y-cap), in relative ln units."""
    z=a*h
    logterm=torch.log(b.clamp_min(1e-30))+torch.log(h.clamp_min(1e-30))+y-cap+log_exprel(z)
    return y+z-torch.logaddexp(torch.zeros_like(logterm),logterm)

class StructuredODE(nn.Module):
    def __init__(self,kind,no,nm,nx,width=16,sub_h=4.,max_sub=8):
        super().__init__();self.kind=kind;self.sub_h=sub_h;self.max_sub=max_sub
        self.backbone=CurrentBackbone(no,nm,nx) if kind=='node' else ParametricBackbone(no,nm,nx)
        self.correction=kind in ('mul','add')
        if self.correction:
            self.context=nn.Linear(self.backbone.context_dim+1,width)
            self.state=nn.Linear(2,width,bias=False)
            self.head=nn.Linear(width,1)
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def freeze_backbone(self):
        for p in self.backbone.parameters():p.requires_grad_(False)
    def residual(self,ct,y,cap,alpha):
        state=torch.stack((torch.tanh((y-cap)/5),alpha),-1)
        return torch.tanh(self.head(torch.tanh(ct+self.state(state)))[:,0])
    def backbone_trajectory(self,mu,kd,cap,lq,dt,ns,maxsteps):
        # Vectorized closed-form composition of exactly the same frozen-coefficient
        # substeps used below. Double precision protects cumulative cancellation.
        j=torch.arange(maxsteps,device=mu.device)[None,:]
        active=j<ns[:,None];w=((j+.5)/ns[:,None]).clamp(max=1)
        m=mu[:,:-1,None]+w[:,None,:]*(mu[:,1:,None]-mu[:,:-1,None])
        d=kd[:,:-1,None]+w[:,None,:]*(kd[:,1:,None]-kd[:,:-1,None])
        h=(dt/ns)[:,None,None]*active[:,None,:]
        mh=(m*h).flatten(1).double()
        midq=lq[:,None].double()+torch.cumsum(mh,1)-.5*mh
        growth=m.flatten(1).double()*torch.sigmoid(midq)
        hh=h.expand(-1,mu.shape[1]-1,-1).flatten(1).double()
        z=(growth-d.flatten(1).double())*hh;cum=torch.cumsum(z,1)
        logterms=torch.log(growth.clamp_min(1e-30))+torch.log(hh.clamp_min(1e-30))+cum-z-cap[:,None].double()+log_exprel(z)
        logterms=torch.where(hh>0,logterms,torch.full_like(logterms,-1e30))
        yy=cum-F.softplus(torch.logcumsumexp(logterms,1))
        yy=yy[:,maxsteps-1::maxsteps].to(mu.dtype)
        return torch.cat((torch.zeros_like(yy[:,:1]),yy),1)/LN10
    def forward(self,T,x,org,mat,dt,y0):
        mu,kd,cap,lq,ctx=self.backbone.fields(T,x,org,mat,dt,y0)
        if self.correction:
            ct=self.context(torch.cat((ctx[:,None,:].expand(-1,T.shape[1],-1),((T-20)/20)[:,:,None]),-1))
        ns=torch.ceil(dt/self.sub_h).clamp(1,self.max_sub).to(torch.long)
        h=dt/ns; maxsteps=int(ns.max().item())
        if not self.correction:
            return self.backbone_trajectory(mu,kd,cap,lq,dt,ns,maxsteps),mu.new_zeros(())
        y=torch.zeros_like(y0);out=[y];reg=y.new_zeros(());den=y.new_zeros(())
        for k in range(T.shape[1]-1):
            for j in range(maxsteps):
                active=j<ns; w=((j+.5)/ns).clamp(max=1)
                m=mu[:,k]+w*(mu[:,k+1]-mu[:,k]);d=kd[:,k]+w*(kd[:,k+1]-kd[:,k])
                alpha=torch.sigmoid(lq+.5*h*m)
                growth=m*alpha; a=growth-d; b=growth
                if self.correction:
                    cm=ct[:,k,:]+w[:,None]*(ct[:,k+1,:]-ct[:,k,:])
                    r=self.residual(cm,y,cap,alpha)
                    if self.kind=='mul': c=torch.exp(math.log(4)*r); aa=a*c;bb=b*c
                    else: aa=a+.1*r;bb=b
                    mid=logistic_flow(y,aa,bb,cap,h/2)
                    r=self.residual(cm,mid,cap,alpha)
                    if self.kind=='mul': c=torch.exp(math.log(4)*r);a=a*c;b=b*c
                    else:a=a+.1*r
                    reg=reg+((math.log(4)*r).square()*h*active).sum();den=den+(h*active).sum()
                yn=logistic_flow(y,a,b,cap,h)
                y=torch.where(active,yn,y);lq=torch.where(active,lq+h*m,lq)
            out.append(y)
        return torch.stack(out,1)/LN10,reg/den.clamp_min(1.)

def make_model(kind,no,nm,nx,cfg):
    if kind in ('rnn','lstm','gru','transformer'): return Sequence(kind,no,nm,nx,cfg['width'])
    return StructuredODE(kind,no,nm,nx,cfg.get('width',16))
