"""Endpoint-neutral, weighted-orthogonal reconstruction detail for Option B."""
import numpy as np
from scipy.interpolate import BSpline
from track1.numerics import basis
from .residual import KNOTS
from .temporal import fit
from .causal_kinematics import estimate

def detail_basis(t,derivative=0):
 t=np.asarray(t,float);spline=BSpline(KNOTS,np.eye(7),3,extrapolate=False);b=np.nan_to_num(spline(t));q=(1-t)**2
 if derivative==0:return q[...,None]*b
 bp=np.nan_to_num(spline(t,nu=1));qp=-2*(1-t)
 if derivative==1:return qp[...,None]*b+q[...,None]*bp
 if derivative==2:return 2*b+2*qp[...,None]*bp+q[...,None]*np.nan_to_num(spline(t,nu=2))
 raise ValueError('Detail derivatives supported through acceleration')

def fit_detail(time,h,variance,prior,quality,options,scale,count=3,strength=.1,cutoff=1.,coarse_model=None):
 time=np.asarray(time,float);keep=(time<=cutoff)&(time>=cutoff-1);t=time[keep]+(1-cutoff);h=np.asarray(h)[keep];variance=np.asarray(variance)[keep];prior=np.asarray(prior)[keep];quality=np.asarray(quality)[keep]
 if len(t)<8 or count not in [2,3]:raise ValueError('Detail requires >=8 past observations and two/three atoms')
 coarse=coarse_model if coarse_model is not None else fit(t,h,variance,prior,quality,options,scale);phi=basis(2*t-1);raw=detail_basis(t);weights=np.median(coarse['weights'][:,:2],axis=1);constraint=phi.T@(weights[:,None]*raw);_,_,v=np.linalg.svd(constraint,full_matrices=True);transform=v[-count:].T;psi=raw@transform;error=h[:,:2]-phi@coarse['coefficients'][:,:2];noise=np.maximum(np.median(variance[:,:2],axis=0),1e-12);energy=np.sum(weights[:,None]*error**2,axis=0)/weights.sum();excess=np.maximum(energy-noise,0);gate=excess/(excess+4*noise+1e-20);a=np.linalg.solve(psi.T@(weights[:,None]*psi)+np.eye(count)*strength,psi.T@(weights[:,None]*error));a*=gate;power=np.sum(weights[:,None]*(psi@a)**2,axis=0)/weights.sum();a*=np.minimum(1.,np.sqrt(energy/np.maximum(power,1e-20)))
 return dict(coarse=coarse['coefficients'],detail=a,transform=transform,count=count,cutoff=cutoff,orthogonality_error=float(np.max(abs(phi.T@(weights[:,None]*psi)))),gate=gate,weights=weights)

def evaluate(model,time,derivative=0):
 t=np.atleast_1d(np.asarray(time,float))-model['cutoff']+1
 if np.any((t<0)|(t>1)):raise ValueError('History detail is a reconstruction feature within its window; no extrapolation')
 result=(basis(2*t-1,derivative)*2**derivative)@model['coarse'];result[:,:2]+=detail_basis(t,derivative)@model['transform']@model['detail'];return result

def multiscale_summary(case,coarse,scale):
 t=case['time'];h=case['h'];keep=t<=1.;t=t[keep];h=h[keep];variance=case['variance'][keep];endpoint_velocity=(2*basis(1.,1)@coarse)[:2];values=[];valid=[]
 for window in [.5,.25]:
  state=estimate(t,h[:,:2],variance[:,:2],1.,window=window);values.extend(state.velocity-endpoint_velocity if state.support_valid else [0.,0.]);valid.extend([state.support_valid]*2)
 error=((h[:,:2]-basis(2*t-1)@coarse[:,:2])/scale[:2])**2;recent=t>=.75;values.extend([float(error[recent].mean()/max(float(error.mean()),1e-12)) if recent.any() else 0.,float(np.mean(case['quality'][keep][recent])) if recent.any() else 0.]);valid.extend([bool(recent.any())]*2)
 return np.asarray(values),np.asarray(valid)
