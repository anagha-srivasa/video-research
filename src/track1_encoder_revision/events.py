"""Past-only innovation evidence, independent of the position-estimator bank."""
import collections
import numpy as np

class InnovationEvidence:
 def __init__(self,noise_z=3.5,window=.12,on_samples=1):self.noise_z=noise_z;self.window=window;self.on_samples=on_samples;self.samples=collections.deque();self.pending=[];self.last=None;self.cusum=0.;self.active=False;self.high=0;self.low=0;self.direction=None;self.last_reversal=-np.inf
 def update(self,timestamp,position,variance,decision_time=None):
  t=float(timestamp);decision=t if decision_time is None else float(decision_time);p=np.array(position,float,copy=True);v=np.asarray(variance,float)
  if not np.isfinite([t,decision]).all() or t>decision or p.shape!=(2,) or not np.isfinite(p).all() or v.shape!=(2,) or not np.isfinite(v).all() or np.any(v<0):raise ValueError('Finite past native-pixel observations and diagonal variances required')
  if self.last is not None and t<=self.last:raise ValueError('Strictly increasing observations required')
  reset=self.last is not None and t-self.last>.12
  if reset:self.samples.clear();self.cusum=0.;self.active=False;self.high=self.low=0;self.direction=None
  self.last=t
  while self.samples and self.samples[0][0]<t-.2:self.samples.popleft()
  score=None;speed=None;kind=None
  if len(self.samples)>=4:
   tt=np.array([x[0] for x in self.samples]);pp=np.array([x[1] for x in self.samples]);xx=np.c_[np.ones(len(tt)),tt-t];beta=np.linalg.lstsq(xx,pp,rcond=None)[0];res=pp-xx@beta;noise=np.maximum(np.array([x[2] for x in self.samples]).mean(0)+v+np.mean(res**2,axis=0),.25);innovation=p-beta[0];score=float(np.sqrt(np.sum(innovation**2/noise)));self.cusum=min(50.,max(0.,self.cusum+score-2.5))
   recent=np.r_[pp,p[None]];times=np.r_[tt,t];keep=times>=t-self.window;recent=recent[keep];times=times[keep]
   if len(times)<3:
    self.samples.append((t,p,v.copy()))
    return dict(innovation_score=score,cusum=self.cusum,speed_native_px_s=None,active=self.active,support_valid=False,reset=reset,latest_input_time=t,position_estimator_modified=False)
   pairs=np.triu_indices(len(times),1);velocity=np.median((recent[pairs[1]]-recent[pairs[0]])/(times[pairs[1]]-times[pairs[0]])[:,None],axis=0);speed=float(np.linalg.norm(velocity));line=recent-velocity[None]*(times-t)[:,None];robust_noise=np.maximum(1.4826*np.median(abs(line-np.median(line,axis=0)),axis=0),np.sqrt(v));se=robust_noise/np.sqrt(max(np.sum((times-times.mean())**2),1e-9));trend_z=float(np.linalg.norm(velocity/np.maximum(se,1e-9)));strong=speed>=100 and trend_z>=self.noise_z and (score>=2. or self.cusum>=4.);self.high=self.high+1 if strong else 0;self.low=self.low+1 if speed<=50 or trend_z<1.5 else 0
   if not self.active and self.high>=self.on_samples:self.active=True;kind='motion_onset';self.direction=velocity
   elif self.active and self.low>=3:self.active=False;kind='motion_stop'
   elif self.active and self.direction is not None and np.dot(velocity,self.direction)<-100**2 and self.cusum>=6 and t-self.last_reversal>.15:kind='motion_reversal';self.last_reversal=t;self.direction=velocity
   if self.active and speed>=100 and kind!='motion_reversal' and self.direction is None:self.direction=velocity
   if kind:self.pending.append(dict(kind=kind,evidence_time=t,decision_time=decision,innovation_score=score,cusum=self.cusum,speed_native_px_s=speed,independent_reference=False))
  self.samples.append((t,p,v.copy()))
  return dict(innovation_score=score,cusum=self.cusum,speed_native_px_s=speed,active=self.active,support_valid=len(self.samples)>=5,reset=reset,latest_input_time=t,position_estimator_modified=False)
 def drain(self,cutoff):
  if not np.isfinite(cutoff):raise ValueError('Invalid decision cutoff')
  result=[r for r in self.pending if r['decision_time']<=cutoff];self.pending=[r for r in self.pending if r['decision_time']>cutoff];return result
