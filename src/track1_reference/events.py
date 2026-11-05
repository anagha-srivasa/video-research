"""Separate causal event evidence. Position predictions remain the fixed bank."""
import json
import numpy as np
from . import REPORT
from .protocol import write

class MotionEvidence:
 def __init__(self):self.samples=[];self.active=False;self.high=0;self.low=0;self.last=None
 def update(self,timestamp,position,decision_time=None):
  p=np.array(position,dtype=float,copy=True)
  if decision_time is not None and (not np.isfinite(decision_time) or timestamp>decision_time):raise ValueError('Future event observation')
  if p.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(timestamp):raise ValueError('Finite 2D observations required')
  if self.last is not None and timestamp<=self.last:raise ValueError('Events require strictly increasing observations')
  if self.last is not None and timestamp-self.last>.12:self.samples=[];self.active=False;self.high=self.low=0
  self.last=timestamp;self.samples.append((timestamp,p));self.samples=[s for s in self.samples if s[0]>=timestamp-.10-1e-9]
  if len(self.samples)<4:return None
  tt=np.array([s[0] for s in self.samples]);pp=np.array([s[1] for s in self.samples]);velocity=np.median(np.diff(pp,axis=0)/np.diff(tt)[:,None],axis=0);speed=float(np.linalg.norm(velocity));self.high=self.high+1 if speed>=100 else 0;self.low=self.low+1 if speed<=50 else 0;kind=None
  if not self.active and self.high>=2:self.active=True;kind='motion_onset'
  elif self.active and self.low>=3:self.active=False;kind='motion_stop'
  if kind:return dict(kind=kind,evidence_time=float(timestamp),decision_time=float(timestamp if decision_time is None else decision_time),speed_native_px_s=speed)
  return None

def interval_scores(detections,truth,quiet,tolerance=.25):
 matched=set();delays=[];false=0
 for detection in sorted(detections,key=lambda x:x['decision_time']):
  t=detection['decision_time'];options=[(i,r) for i,r in enumerate(truth) if i not in matched and r['start']<=t<=r['end']+tolerance]
  if options:
   i,r=min(options,key=lambda x:abs(t-x[1]['end']));matched.add(i);delays.append([t-r['end'],t-r['start']])
  elif any(a<=t<=b for a,b in quiet):false+=1
 merged=[]
 for a,b in sorted(quiet):
  if not np.isfinite([a,b]).all() or b<a:raise ValueError('Invalid quiet interval')
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
  else:merged.append((a,b))
 seconds=sum(b-a for a,b in merged)
 return dict(events=len(truth),recalled=len(matched),recall=len(matched)/len(truth) if truth else None,detection_delay_intervals_seconds=delays,quiet_false_alarms=false,quiet_seconds=seconds,false_alarms_per_quiet_minute=false*60/seconds if seconds else None)

def event_stream(t,p,var,cutoffs):
 from track1_lab.causal_kinematics import estimate
 detector=MotionEvidence();last=-np.inf;previous_nearest=None;active={'cv':False,'nearest':False};events={m:[] for m in ['causal_evidence','cv','nearest']};states=[]
 for cutoff in cutoffs:
  fresh=np.flatnonzero((t>last)&(t<=cutoff));last=cutoff
  # Accumulate transient evidence even when an event completes between updates.
  for i in fresh:
   event=detector.update(t[i],p[i],cutoff)
   if event and event['evidence_time']>=cutoffs[0] and event['kind']=='motion_onset':events['causal_evidence'].append(event)
  cv=estimate(t,p,var,cutoff)
  if not cv.support_valid:
   states.append(dict(cutoff=float(cutoff),cv_valid=False,causal_evidence_valid=len(detector.samples)>=4,nearest_valid=bool(np.any((t<=cutoff)&(t>=cutoff-.1)))))
   continue
  index=np.flatnonzero(t<=cutoff)[-1];nearest=p[index];speeds={'cv':float(np.linalg.norm(cv.velocity)),'nearest':float(np.linalg.norm(nearest-previous_nearest[1])/(cutoff-previous_nearest[0])) if previous_nearest else 0.};previous_nearest=(cutoff,nearest)
  for method,speed in speeds.items():
   now=speed>=100
   if now and not active[method]:events[method].append(dict(decision_time=float(cutoff),evidence_time=float(cutoff),kind='motion_onset'))
   active[method]=now
  states.append(dict(cutoff=float(cutoff),latest_input=float(t[index]),cv_valid=True,causal_evidence_valid=len(detector.samples)>=4,nearest_valid=True))
 return events,states

def synthetic():
 from track1_lab.kinematics import truth,KINDS
 records=[]
 for seed in range(90):
  rng=np.random.default_rng(20261029+seed);kind=KINDS[seed%5];duration=[.1,.25,.5][seed//5%3];noise=[.5,2.,5.][seed//15%3];onset=2.+rng.uniform(-.15,.15);clock=np.arange(0,4.201,.02);p,_=truth(clock,kind,onset,duration);p+=rng.normal(0,noise,p.shape);corruption=['clean','outlier','missing'][seed//3%3]
  if corruption=='outlier':p[np.argmin(abs(clock-onset-.04))]+=[80,-55]
  keep=np.ones(len(clock),bool)
  if corruption=='missing':keep&=~((clock>=onset-.04)&(clock<=onset+.11))
  tt=clock[keep];pp=p[keep];var=np.ones_like(pp)*noise**2;grid=np.arange(1.,4.,.002);_,velocity=truth(grid,kind,onset,duration);speed=np.linalg.norm(velocity,axis=1);moving=speed>=100;starts=np.flatnonzero(moving&~np.r_[moving[0],moving[:-1]]);targets=[dict(start=float(grid[i]),end=float(grid[i])) for i in starts] if kind in ['pickup','burst'] else [];quiet_mask=speed<1;edges=np.diff(np.r_[False,quiet_mask,False].astype(int));quiet=[(float(grid[a]),float(grid[min(b,len(grid)-1)])) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
  for hop in [.125,.25]:
   detected,states=event_stream(tt,pp,var,np.arange(1.,4.001,hop))
   for method,events in detected.items():records.append(dict(seed=seed,split='synthetic_train' if seed<60 else 'synthetic_validation',kind=kind,hop_seconds=hop,method=method,**interval_scores(events,targets,quiet)))
 summary=[]
 for part in ['synthetic_train','synthetic_validation']:
  for hop in [.125,.25]:
   for method in ['causal_evidence','cv','nearest']:
    rs=[r for r in records if (r['split'],r['hop_seconds'],r['method'])==(part,hop,method)];n=sum(r['events'] for r in rs);recalled=sum(r['recalled'] for r in rs);quiet=sum(r['quiet_seconds'] for r in rs);delay=[x for r in rs for x in r['detection_delay_intervals_seconds']];summary.append(dict(split=part,hop_seconds=hop,method=method,events=n,recall=recalled/n if n else None,median_detection_delay_seconds=float(np.median([x[0] for x in delay])) if delay else None,false_alarms_per_quiet_minute=sum(r['quiet_false_alarms'] for r in rs)*60/quiet if quiet else None))
 write(REPORT/'event_synthetic.json',dict(summary=summary,per_episode=records,independent_reference=False,position_estimator='Fixed switching bank untouched; event evidence is a separate channel',nearest_event_baseline='Thresholded change between consecutive nearest-position updates',quiet_rate='Unmatched detections per true quiet minute, not fraction of quiet samples',thresholds=dict(on_px_s=100,off_px_s=50,minimum_observations=4,high_samples=2,low_samples=3),promoted=False));return summary

if __name__=='__main__':synthetic();print('Synthetic event diagnostics only; independent evaluation awaits adjudicated references')
