"""Option B: causal local motion, independent of functional coefficients.

Local constant velocity is the explicit baseline, not a learned forecaster.
The robust alternative and conditional covariance require independent review.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class KinematicState:
 position:object
 velocity:object
 covariance:object
 method:str
 support_valid:bool
 failure_reason:object
 observation_count:int
 latest_input_time:object
 age_seconds:object
 covariance_calibrated:bool=False
 derivative_reliable:object=None
 acceleration:object=None

def estimate(time,position,variance,cutoff,window=.2,robust=False):
 time=np.asarray(time,float);position=np.asarray(position,float);variance=np.asarray(variance,float)
 if position.shape!=(len(time),2) or variance.shape!=position.shape:raise ValueError('Expected time[N], position[N,2], variance[N,2]')
 if not np.isfinite(time).all() or not np.isfinite(position).all() or not np.isfinite(variance).all() or np.any(variance<0):raise ValueError('Kinematic inputs must be finite with nonnegative variance')
 keep=(time<=cutoff)&(time>=cutoff-window);t=time[keep];p=position[keep];var=variance[keep];order=np.argsort(t);t=t[order];p=p[order];var=var[order];n=len(t);method='robust_local_linear' if robust else 'constant_velocity_local_ols';reason=None;age=float(cutoff-t[-1]) if n else None
 if n<3:reason='too_few_recent_observations'
 elif np.any(np.diff(t)<=0):reason='duplicate_timestamps'
 elif t[-1]-t[0]<.04-1e-9:reason='insufficient_time_span'
 elif age>.1+1e-9:reason='stale_observations'
 elif np.diff(t).max()>.12+1e-9:reason='observation_gap'
 if reason:return KinematicState(None,None,None,method,False,reason,n,float(t[-1]) if n else None,age)
 x=np.c_[np.ones(n),t-cutoff];weights=np.ones(n)
 for _ in range(4 if robust else 1):
  root=np.sqrt(weights);beta=np.linalg.lstsq(x*root[:,None],p*root[:,None],rcond=None)[0];residual=p-x@beta
  if robust:
   sigma=np.maximum(1.4826*np.median(abs(residual-np.median(residual,axis=0)),axis=0),np.maximum(np.sqrt(np.median(var,axis=0)),1e-9));distance=np.linalg.norm(residual/sigma,axis=1);new=np.minimum(1.,2.5/np.maximum(distance,1e-12))
   if _<3:weights=new
 gram=x.T@(weights[:,None]*x);bread=np.kron(np.linalg.inv(gram),np.eye(2));observed=(residual.T*weights)@residual/max(weights.sum()-2,1);noise=np.diag(np.average(var,axis=0,weights=weights));eigen,vectors=np.linalg.eigh(observed-noise);discrepancy=(vectors*np.maximum(eigen,0))@vectors.T;meat=np.zeros((4,4))
 for xi,wi,vi in zip(x,weights,var):meat+=wi**2*np.kron(np.outer(xi,xi),np.diag(vi)+discrepancy)
 covariance=bread@meat@bread.T;covariance=(covariance+covariance.T)/2;covariance+=np.eye(4)*max(np.trace(covariance)*1e-12,1e-18)
 return KinematicState(beta[0],beta[1],covariance,method,True,None,n,float(t[-1]),age)
