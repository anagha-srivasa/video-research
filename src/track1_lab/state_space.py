"""Causal continuous-white-acceleration Kalman CV comparison, CPU float64.

No model/regime posterior is invented. IMM/CA remain separate future ablations.
Robust innovation inflation is a heuristic; posterior calibration is unverified.
"""
import numpy as np

def checked_covariance(covariance,relative_tolerance=1e-10):
 p=np.asarray(covariance,float)
 if p.ndim!=2 or p.shape[0]!=p.shape[1] or not np.isfinite(p).all():raise ValueError('Invalid covariance shape or nonfinite values')
 p=(p+p.T)/2;eigen,vectors=np.linalg.eigh(p);scale=max(float(np.max(abs(eigen))),1e-15)
 if eigen.min()< -relative_tolerance*scale:raise ValueError('Substantial negative covariance eigenvalue')
 if eigen.min()<0:p=(vectors*np.maximum(eigen,0))@vectors.T
 return (p+p.T)/2

def transition(dt,spectral_density):
 if not np.isfinite([dt,spectral_density]).all() or dt<0 or spectral_density<0:raise ValueError('Noncausal transition or invalid process density')
 eye=np.eye(2);f=np.block([[eye,dt*eye],[np.zeros((2,2)),eye]]);q=spectral_density*np.block([[dt**3/3*eye,dt**2/2*eye],[dt**2/2*eye,dt*eye]]);return f,q

def filter_states(time,position,variance,cutoffs,spectral_density=.01,robust=True):
 t=np.asarray(time,float);p=np.asarray(position,float);variance=np.asarray(variance,float);cutoffs=np.asarray(cutoffs,float)
 if p.shape!=(len(t),2) or variance.shape!=p.shape or np.any(np.diff(t)<=0) or np.any(np.diff(cutoffs)<0) or not np.isfinite(t).all() or not np.isfinite(cutoffs).all():raise ValueError('Strict ordered observation timestamps and ordered finite cutoffs required; late/duplicate arrivals must be handled explicitly')
 h=np.c_[np.eye(2),np.zeros((2,2))];i=0;state=None;cov=None;last=None;rows=[];innovations=[]
 for cutoff in cutoffs:
  while i<len(t) and t[i]<=cutoff:
   if not np.isfinite(p[i]).all() or not np.isfinite(variance[i]).all() or np.any(variance[i]<0):raise ValueError('Invalid consumed observation')
   r=np.diag(np.maximum(variance[i],1e-12))
   if state is None:
    state=np.r_[p[i],0.,0.];cov=np.block([[r,np.zeros((2,2))],[np.zeros((2,2)),np.eye(2)]]);innovation=0.
   else:
    f,q=transition(float(t[i]-last),spectral_density);state=f@state;cov=checked_covariance(f@cov@f.T+q);error=p[i]-h@state;s=h@cov@h.T+r;innovation=float(error@np.linalg.solve(s,error))
    if robust:r*=max(1.,innovation/9.)
    s=h@cov@h.T+r;k=np.linalg.solve(s,h@cov).T;state=state+k@error;a=np.eye(4)-k@h;cov=checked_covariance(a@cov@a.T+k@r@k.T)
   last=float(t[i]);innovations.append(innovation);i+=1
  if state is None:rows.append(dict(valid=False,state=None,covariance=None,latest_input_time=None,observation_count=0));continue
  f,q=transition(float(cutoff-last),spectral_density);snapshot=f@state;posterior=checked_covariance(f@cov@f.T+q);rows.append(dict(valid=i>=3 and cutoff-last<=.1+1e-9,state=snapshot,covariance=posterior,latest_input_time=last,observation_count=i,age_seconds=float(cutoff-last),innovation_nees=innovations[-1],covariance_calibrated=False))
 return rows
