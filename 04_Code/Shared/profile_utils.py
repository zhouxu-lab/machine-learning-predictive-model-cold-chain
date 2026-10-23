from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""Ordered piecewise-linear profiles with explicit, right-continuous jumps."""
import numpy as np


class TemperatureProfile:
    def __init__(self, profile, constant):
        if profile is None or len(profile)==0:
            self.t=np.array([0.]);self.left=self.right=np.array([float(constant)])
            return
        keys=['time_h']+(['point_index'] if 'point_index' in profile else [])
        d=profile.sort_values(keys,kind='stable')
        g=d.groupby('time_h',sort=True).value
        self.t=np.asarray(g.first().index,float)
        self.left=g.first().to_numpy(float);self.right=g.last().to_numpy(float)
        if not (np.isfinite(self.t).all() and np.isfinite(self.left).all() and np.isfinite(self.right).all()):
            raise ValueError('Non-finite temperature profile')

    def __call__(self, times, side='right'):
        x=np.asarray(times,float)
        if len(self.t)==1:
            return np.where(x<self.t[0],self.left[0],self.right[0])
        i=np.searchsorted(self.t,x,side='right')-1
        j=np.clip(i,0,len(self.t)-2)
        w=(x-self.t[j])/(self.t[j+1]-self.t[j])
        out=self.right[j]+w*(self.left[j+1]-self.right[j])
        out=np.where(i<0,self.left[0],out)
        out=np.where(i>=len(self.t)-1,self.right[-1],out)
        if side=='left':
            q=np.clip(np.searchsorted(self.t,x),0,len(self.t)-1)
            out=np.where(x==self.t[q],self.left[q],out)
        return out

    def segments(self,a,b):
        knots=np.r_[a,self.t[(self.t>a)&(self.t<b)],b]
        return np.diff(knots),self(knots[:-1]),self(knots[1:],side='left')

    def summary(self,a,b,tmin=-2.):
        dt,v,w=self.segments(a,b);duration=b-a
        mean=float(np.sum(dt*(v+w)/2)/duration)
        second=float(np.sum(dt*(v*v+v*w+w*w)/3)/duration)
        out={'T_mean_hist':mean,'T_std_hist':float(np.sqrt(max(0.,second-mean*mean))),
             'T_min_hist':float(min(v.min(),w.min(),self(b))),
             'T_max_hist':float(max(v.max(),w.max(),self(b)))}
        for threshold in (5,7,10,15):
            lo=np.minimum(v,w);hi=np.maximum(v,w)
            fraction=np.divide(hi-threshold,hi-lo,out=(v>threshold).astype(float),where=hi!=lo)
            out[f'h_above{threshold}']=float(np.sum(dt*np.clip(fraction,0,1)))
        v=v-tmin;w=w-tmin
        both=(v>=0)&(w>=0)
        integral=np.where(both,dt*(v*v+v*w+w*w)/3,0.)
        cross=(v*w<0)
        pos=np.maximum(v,w)
        positive_dt=dt*np.divide(pos,np.abs(v-w),out=np.zeros_like(pos),where=v!=w)
        integral=np.where(cross,positive_dt*pos*pos/3,integral)
        out['ratk_integral']=float(integral.sum())
        dt,v,w=self.segments(a+.75*duration,b)
        out['T_last_quarter_mean']=float(np.sum(dt*(v+w)/2)/(.25*duration))
        return out
