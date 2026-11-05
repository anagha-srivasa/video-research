"""Causal three-hypothesis motion and confidence-gated 2D background view."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from track1_lab.causal_kinematics import estimate
from track1_lab.motion import project
from . import *

MODES=['stationary','constant_velocity','transition']

def components(t,p,var,cutoff):
 keep=(t<=cutoff)&(t>=cutoff-.35);t=t[keep];p=p[keep];var=var[keep]
 if len(t)<3 or t[-1]-t[0]<.04-1e-9 or cutoff-t[-1]>.1+1e-9 or np.any(np.diff(t)<=0) or max(np.diff(t))>.12+1e-9:raise ValueError('Unsupported causal motion')
 noise=np.maximum(np.median(var,axis=0),1.);models=[]
 for degree in [0,1,2]:
  if degree==0:beta=np.zeros((3,2));beta[0]=p[-1];residual=p[-min(4,len(p)):]-beta[0];cov=np.diag(noise+np.median(residual**2,axis=0));models.append((beta,cov));continue
  local=t>=cutoff-(.2 if degree==1 else .35);tt=(t[local]-cutoff)/.35;pp=p[local];x=np.polynomial.polynomial.polyvander(tt,degree);weights=np.ones(len(tt))
  for _ in range(3):
   ridge=np.diag([1e-9]+[.001]+([.03] if degree==2 else []));b=np.linalg.solve(x.T@(weights[:,None]*x)+ridge,x.T@(weights[:,None]*pp));res=pp-x@b;scale=np.maximum(1.4826*np.median(abs(res-np.median(res,axis=0)),axis=0),np.sqrt(noise));weights=np.minimum(1.,2.5/np.maximum(np.linalg.norm(res/scale,axis=1),1e-12))
  beta=np.zeros((3,2));beta[:degree+1]=b;beta[1]/=.35;beta[2]*=2/.35**2;cov=np.diag(noise+np.mean(res**2,axis=0));models.append((beta,cov))
 return models

def predict_components(models,horizon):
 points=[]
 for degree,(b,cov) in enumerate(models):
  tau=.15;displacement=b[1]*horizon
  if degree==2:displacement+=b[2]*tau*(horizon-tau*(1-np.exp(-horizon/tau)))
  points.append(b[0]+displacement)
 return np.asarray(points)

def switching(time,position,variance,cutoff,noise_px=12.):
 t=np.asarray(time,float);p=np.asarray(position,float);var=np.asarray(variance,float)
 if p.shape!=(len(t),2) or var.shape!=p.shape or not all(np.isfinite(a).all() for a in [t,p,var]) or np.any(var<0):raise ValueError('Invalid bank observations')
 keep=(t<=cutoff)&(t>=cutoff-.65);order=np.argsort(t[keep]);t=t[keep][order];p=p[keep][order];var=var[keep][order]
 if len(t)<4 or np.any(np.diff(t)<=0):raise ValueError('Insufficient or duplicate observations')
 weights=np.array([.5,.4,.1]);updates=0
 for i in range(3,len(t)):
  try:models=components(t[:i],p[:i],var[:i],t[i-1])
  except ValueError:continue
  prediction=predict_components(models,t[i]-t[i-1]);s=np.array([np.diag(c)+var[i]+noise_px**2 for _,c in models]);loglike=-.5*np.sum((prediction-p[i])**2/s+np.log(s),axis=1);prior=.94*weights+.06*np.array([.5,.4,.1]);score=np.log(prior)+loglike;weights=np.exp(score-score.max());weights/=weights.sum();updates+=1
 models=components(t,p,var,cutoff)
 return dict(models=models,weights=weights,latest_input_time=float(t[-1]),updates=updates,calibrated=False)

def predict(bank,horizon):
 if not np.isfinite(horizon) or horizon<0 or horizon>.55+1e-9:raise ValueError('Declared diagnostic horizon is 0 to 0.55 s including source timestamp tolerance')
 points=predict_components(bank['models'],horizon);mean=bank['weights']@points;cov=np.zeros((2,2))
 for w,point,(_,c) in zip(bank['weights'],points,bank['models']):cov+=w*(c*(1+(horizon/.2)**2)+np.outer(point-mean,point-mean))
 return mean,(cov+cov.T)/2

def affine_pair(rgb0,rgb1,mask0,mask1):
 import cv2
 a=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY);points=cv2.goodFeaturesToTrack(a,maxCorners=600,qualityLevel=.015,minDistance=8,mask=(~mask0).astype(np.uint8)*255)
 if points is None or len(points)<40:return None,dict(reason='few_background_features',features=0 if points is None else len(points))
 q,status,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(21,21),maxLevel=3);back,ok,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3);pp=points[:,0];qq=q[:,0];inside=(qq[:,0]>=0)&(qq[:,0]<b.shape[1])&(qq[:,1]>=0)&(qq[:,1]<b.shape[0]);safe=np.clip(np.round(qq).astype(int),[0,0],[b.shape[1]-1,b.shape[0]-1]);good=(status.ravel()>0)&(ok.ravel()>0)&inside&(np.linalg.norm(back[:,0]-pp,axis=1)<1.)&~mask1[safe[:,1],safe[:,0]];pp=pp[good];qq=qq[good]
 if len(pp)<40:return None,dict(reason='few_consistent_background_features',features=len(pp))
 affine,inlier=cv2.estimateAffinePartial2D(pp,qq,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=1500,confidence=.995)
 if affine is None:return None,dict(reason='ransac_failed',features=len(pp))
 inlier=inlier.ravel().astype(bool);fraction=float(inlier.mean());cells=np.floor(pp[inlier]/np.array([a.shape[1]/4,a.shape[0]/3])).astype(int);coverage=len(set(map(tuple,cells)))/12;error=np.linalg.norm(pp@affine[:,:2].T+affine[:,2]-qq,axis=1);scale=float(np.sqrt(np.linalg.det(affine[:,:2])));rotation=float(abs(np.arctan2(affine[1,0],affine[0,0])));valid=fraction>=.65 and coverage>=.5 and np.quantile(error[inlier],.95)<=2 and .9<=scale<=1.1 and rotation<=np.deg2rad(12)
 return (np.vstack([affine,[0,0,1]]) if valid else None),dict(reason=None if valid else 'confidence_gate',features=len(pp),inlier_fraction=fraction,grid_coverage=coverage,scale=scale,rotation_rad=rotation,p95_residual_small_px=float(np.quantile(error[inlier],.95)))

def camera_worker(v):
 import cv2
 from track1_retrain.context import Frames
 from data_v2.geometry import render
 cv2.setNumThreads(2);cv2.setRNGSeed(CFG['seed']);episodes=pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist();episodes=[e for e in episodes if e['video_id']==v];frames=pq.read_table(ROOT/'dataset/context_v3/frames.parquet').to_pylist();frames=[f for f in frames if f['video_id']==v];masks=pq.read_table(W/f'camera_masks.{v}.parquet').to_pylist();bytime=collections.defaultdict(list)
 for r in masks:bytime[r['timestamp_us']].append(r)
 result=[];reader=Frames()
 try:
  for e in episodes:
   local=[r for r in frames if e['start_us']<=r['timestamp_us']<e['end_us']];chosen={}
   for t in range(e['start_us'],e['end_us'],100000):
    available=[r for r in local if r['timestamp_us']<=t]
    if available:f=available[-1];chosen[f['timestamp_us']]=f
   prior=None;transform=np.eye(3);epoch=0
   for k,(timestamp,f) in enumerate(sorted(chosen.items())):
    rgb=cv2.resize(reader.get_frame(v,timestamp),(640,360),interpolation=cv2.INTER_AREA);mask=np.zeros((360,640),bool)
    for r in bytime[timestamp]:
     rings=json.loads(r['contours_json']);sx=640/r['annotation_width'];sy=360/r['annotation_height']
     for ring in rings:ring['points']=(np.asarray(ring['points'])*[sx,sy]).tolist()
     mask|=render(rings,(360,640)).astype(bool)
    mask=cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool);a=None;detail=dict(reason='initial_reference')
    if prior is not None and bytime[timestamp] and prior[3]:a,detail=affine_pair(prior[1],rgb,prior[2],mask)
    elif prior is not None:detail=dict(reason='missing_foreground_exclusion')
    if a is None:epoch+=1;transform=np.eye(3)
    else:transform=a@transform
    native=np.diag([3.,3.,1.])@transform@np.diag([1/3,1/3,1.]);result.append(dict(video_id=v,context_id=e['context_id'],timestamp_us=timestamp,epoch=f'{e["context_id"]}:{epoch}',valid=a is not None,reference_only=a is None,matrix=native.ravel().tolist(),camera_confidence_available=a is not None,**detail,physical_camera_pose=False));prior=(timestamp,rgb,mask,bool(bytime[timestamp]))
    if k%16==0:reader.close();guard()
 finally:reader.close()
 save(D/f'camera.{v}.parquet',result);return dict(video_id=v,frames=len(result),accepted_pairs=sum(r['valid'] for r in result))

def camera(monitor):
 import duckdb
 for v in VIDEOS:
  with duckdb.connect() as db:
   db.execute('SET threads=1');db.execute("SET memory_limit='1GB'");table=db.execute('SELECT m.timestamp_us,m.annotation_width,m.annotation_height,m.contours_json FROM read_parquet(?) m JOIN read_parquet(?) f ON m.video_id=f.video_id AND m.timestamp_us=f.timestamp_us WHERE m.video_id=?',[str(ROOT/f'canonical/v2/masks.{v}.parquet'),str(ROOT/'dataset/context_v3/frames.parquet'),v]).fetch_arrow_table();pq.write_table(table,W/f'camera_masks.{v}.parquet',compression=None)
 result=pool(camera_worker,VIDEOS,monitor,workers=2);result.sort(key=lambda r:r['video_id']);atomic_json(R/'camera.json',dict(videos=result,method='confidence-gated background partial affine; annotated foreground excluded with dilation',coordinate_mode='2D image alignment, not metric camera/object separation',future_camera_observed=False,unannotated_foreground_possible=True,thresholds_fixed_before_evaluation=True,independent_preregistration=False))

def lookup_camera(rows,timestamp):
 available=[r for r in rows if r['timestamp_us']<=timestamp]
 if not available:return None
 r=available[-1]
 return r if timestamp-r['timestamp_us']<=120000 else None

def camera_relative(times,positions,cutoff,rows):
 current=lookup_camera(rows,round(cutoff*1e6))
 if current is None or not current['valid']:raise ValueError('No confident current background view')
 transformed=[];selected=[]
 for t,p in zip(times,positions):
  r=lookup_camera(rows,round(t*1e6))
  if r is None or r['epoch']!=current['epoch']:raise ValueError('Camera epoch break')
  h=np.array(r['matrix']).reshape(3,3);transformed.append((np.linalg.inv(h)@np.r_[p,1])[:2]);selected.append(r)
 recent=[r for r in rows if r['epoch']==current['epoch'] and cutoff-.35<=r['timestamp_us']/1e6<=cutoff]
 if len(recent)<3:raise ValueError('Insufficient camera history')
 tt=np.array([r['timestamp_us']/1e6-cutoff for r in recent]);xx=np.c_[np.ones(len(tt)),tt];matrices=np.array([r['matrix'] for r in recent]);beta=np.linalg.lstsq(xx,matrices,rcond=None)[0]
 return np.array(transformed),beta,current['epoch']

def future_camera(beta,horizon):
 h=(beta[0]+horizon*beta[1]).reshape(3,3);h[2]=[0,0,1]
 if not np.isfinite(h).all() or np.linalg.det(h[:2,:2])<=.25 or np.linalg.det(h[:2,:2])>=4:raise ValueError('Unstable camera continuation')
 return h

def queries(v):
 from track1_retrain.temporal import load,proposals
 source=SOURCES['context'];rows,data=load(v,16,source/'spatial');episodes=[e for e in pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist() if e['video_id']==v];cases=proposals(v,rows,episodes);camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());bg=pq.read_table(D/f'camera.{v}.parquet').to_pylist();trackids=collections.defaultdict(list)
 for r in rows:trackids[r['entity_track_id']].append(r['row_index'])
 chosen=[]
 for case in cases:
  ids=np.array(case['indices'],int)
  if len(ids)<4:continue
  t=np.array([rows[i]['timestamp_us']/1e6 for i in ids]);cutoff=case['window_end_us']/1e6;p=project(np.array(data['h'][ids,:2]),camera);variance=np.ones_like(p)*4
  try:bank=switching(t,p,variance,cutoff)
  except ValueError:continue
  if bank['latest_input_time']>cutoff:raise AssertionError('Future input')
  if not estimate(t,p,variance,cutoff).support_valid:continue
  chosen.append((case,t,p,variance,bank))
 chosen.sort(key=lambda q:__import__('hashlib').sha256((q[0]['entity_track_id']+str(q[0]['window_end_us'])).encode()).hexdigest());chosen=chosen[:96];records=[];states=[];targets_missing=0
 for case,t,p,var,bank in chosen:
  cutoff=case['window_end_us']/1e6;allids=trackids[case['entity_track_id']];episode=next(e for e in episodes if e['episode_id']==case['episode_id']);background=[r for r in bg if r['context_id']==episode['context_id']];relative=None;cam_error=None
  keep=t>=cutoff-.65
  try:
   rel,beta,cam_epoch=camera_relative(t[keep],p[keep],cutoff,background);relative=(switching(t[keep],rel,var[keep],cutoff),beta)
  except ValueError as e:cam_error=str(e)
  states.append(dict(video_id=v,split=split(v),entity_track_id=case['entity_track_id'],cutoff_us=case['window_end_us'],hypothesis_weights=bank['weights'].tolist(),hypothesis_names=MODES,probabilities_calibrated=False,latest_input_us=round(bank['latest_input_time']*1e6),camera_valid=relative is not None,camera_failure_reason=cam_error,raw_image_position=(bank['weights']@np.array([b[0] for b,c in bank['models']])).tolist(),raw_image_velocity=(bank['weights']@np.array([b[1] for b,c in bank['models']])).tolist(),position_covariance=predict(bank,0)[1].tolist(),joint_position_velocity_covariance=None,velocity_covariance_calibrated=False,camera_epoch=cam_epoch if relative else None,camera_future_affine_parameters=relative[1].tolist() if relative else None,relative_hypothesis_weights=relative[0]['weights'].tolist() if relative else None,relative_position=predict(relative[0],0)[0].tolist() if relative else None,relative_velocity=(relative[0]['weights']@np.array([b[1] for b,c in relative[0]['models']])).tolist() if relative else None,reference_semantics='retrospective visible source centroid, not physical landmark'))
  local=t>=cutoff-.2
  for horizon in [.1,.25,.5]:
   candidate=[i for i in allids if rows[i]['timestamp_us']>case['window_end_us'] and rows[i]['timestamp_us']<episode['end_us'] and abs(rows[i]['timestamp_us']/1e6-cutoff-horizon)<=.04]
   if not candidate:targets_missing+=1;continue
   target=min(candidate,key=lambda i:abs(rows[i]['timestamp_us']/1e6-cutoff-horizon));actual=rows[target]['timestamp_us']/1e6-cutoff;truth=project(np.asarray(data['h'][target,:2]),camera);cv=estimate(t,p,var,cutoff);predictions={'nearest':p[-1],'cv':cv.position+actual*cv.velocity,'switching':predict(bank,actual)[0]};cov=predict(bank,actual)[1]
   if relative:
    try:
     relpred=predict(relative[0],actual)[0];h=future_camera(relative[1],actual);predictions['camera_switching']=(h@np.r_[relpred,1])[:2]
    except ValueError:pass
   for method,point in predictions.items():
    error=point-truth;records.append(dict(video_id=v,split=split(v),context_id=episode['context_id'],entity_track_id=case['entity_track_id'],cutoff_us=case['window_end_us'],horizon_seconds=horizon,actual_horizon_seconds=actual,method=method,error_native_px=float(np.linalg.norm(error)),prediction=point.tolist(),target=truth.tolist(),hypothesis_weights=bank['weights'].tolist(),standardized_radius=float(np.sqrt(error@np.linalg.solve(cov,error))) if method=='switching' else None,camera_available=relative is not None,physical_reference=False,latest_input_us=round(bank['latest_input_time']*1e6)))
 save(R/f'motion.{v}.parquet',records);save(D/f'motion_states.{v}.parquet',states);return dict(video_id=v,states=len(states),measurements=len(records),missing_future_targets=targets_missing)

def run(monitor):
 results=pool(queries,VIDEOS,monitor,workers=5);results.sort(key=lambda r:r['video_id']);records=[]
 for v in VIDEOS:records+=pq.read_table(R/f'motion.{v}.parquet').to_pylist()
 save(R/'motion.parquet',records);summary=[];groups=collections.defaultdict(list)
 for r in records:groups[(r['video_id'],r['entity_track_id'],r['cutoff_us'],r['horizon_seconds'])].append(r)
 for cohort,required in [('raw',{'nearest','cv','switching'}),('camera_supported',{'nearest','cv','switching','camera_switching'})]:
  paired=[r for rows in groups.values() if {r['method'] for r in rows}>=required for r in rows if r['method'] in required]
  for part in ['train','validation']:
   for h in [.1,.25,.5]:
    for method in sorted(required):
     rows=[r for r in paired if (r['split'],r['horizon_seconds'],r['method'])==(part,h,method)];rmse=[]
     for v in sorted({r['video_id'] for r in rows}):rmse.append(float(np.sqrt(np.mean([r['error_native_px']**2 for r in rows if r['video_id']==v]))))
     if rmse:summary.append(dict(cohort=cohort,split=part,horizon_seconds=h,method=method,queries=len(rows),videos=len(rmse),mean_video_rmse_native_px=float(np.mean(rmse))))
 # For simultaneous 95% split-conformal bands the exchangeable units are the
 # five training participants, not thousands of overlapping windows.
 scores=[]
 for v in CFG['train']:
  vals=[r['standardized_radius'] for r in records if r['video_id']==v and r['method']=='switching'];scores.append(max(vals) if vals else None)
 n=sum(x is not None for x in scores);rank=int(np.ceil((n+1)*.95));quantile=sorted(x for x in scores if x is not None)[rank-1] if rank<=n else None
 atomic_json(R/'motion.json',dict(videos=results,matched_metrics=summary,method='causal robust polynomial hypothesis bank with prequential weights; not IMM',fixed_parameters=dict(window_seconds=.35,score_history_seconds=.65,transition_acceleration_decay_seconds=.15,likelihood_noise_native_px=12,prior=[.5,.4,.1]),selected_on_validation=False,promoted=False,weights_calibrated=False,kinematic_covariance_calibrated=False,simultaneous_source95_band=dict(unit='training participant maximum across all sampled contexts/horizons',training_units=n,order_statistic_rank=rank,finite=quantile is not None,quantile=quantile,reason='Insufficient independent calibration units for a finite 95% split-conformal threshold' if quantile is None else None,physical_coverage_verified=False,exchangeability_established=False,formal_calibration_split=False),limits=['Retrospective source-centroid motion includes deformation, occlusion and camera effects.','All choices fixed before validation; no claim of physical regime probabilities.','Camera-view continuation extrapolates only past background evidence; no future camera frame is used.','Only two validation participants; overlapping windows are not independent experiments.'],track2_started=False))

def synthetic_worker(seed):
 from track1_lab.kinematics import truth,KINDS
 rng=np.random.default_rng(20261029+seed);kind=KINDS[seed%5];duration=[.1,.25,.5][seed//5%3];noise=[.5,2.,5.][seed//15%3];corruption=['clean','outlier','missing'][seed//3%3];onset=2.+rng.uniform(-.15,.15);clock=np.arange(0,4.201,.02);positions,_=truth(clock,kind,onset,duration);measured=positions+rng.normal(0,noise,positions.shape)
 if corruption=='outlier':measured[np.argmin(abs(clock-(onset+.04)))]+=[80.,-55.]
 keep=np.ones(len(clock),bool)
 if corruption=='missing':keep&=~((clock>=onset-.04)&(clock<=onset+.11))
 clock=clock[keep];measured=measured[keep];variance=np.ones_like(measured)*noise**2;out=[]
 for hop in [.125,.25]:
  states=[]
  for cutoff in np.arange(1.,4.001,hop):
   try:b=switching(clock,measured,variance,cutoff)
   except ValueError:continue
   cv=estimate(clock,measured,variance,cutoff)
   if cv.support_valid:states.append((cutoff,b,cv,measured[np.flatnonzero(clock<=cutoff)[-1]]))
  times=[];pp=collections.defaultdict(list);vv=collections.defaultdict(list)
  for t in np.arange(1.25,3.801,.02):
   recent=[s for s in states if s[0]<=t]
   if not recent:continue
   cutoff,b,cv,last=recent[-1];h=t-cutoff
   if h>hop+.021:continue
   point,_=predict(b,h);velocity=b['weights']@np.array([m[0][1]+(m[0][2]*.15*(1-np.exp(-h/.15)) if j==2 else 0) for j,m in enumerate(b['models'])]);times.append(t)
   for name,p,v in [('nearest',last,np.zeros(2)),('cv',cv.position+h*cv.velocity,cv.velocity),('switching',point,velocity)]:pp[name].append(p);vv[name].append(v)
  times=np.array(times);target,velocity=truth(times,kind,onset,duration);true_speed=np.linalg.norm(velocity,axis=1);true_active=true_speed>100;true_starts=times[true_active&~np.r_[False,true_active[:-1]]];event=kind in ['pickup','burst'] and len(true_starts)>0;quiet=true_speed<1
  for method in pp:
   p=np.array(pp[method]);v=np.array(vv[method]);active=np.linalg.norm(v,axis=1)>100;starts=times[active&~np.r_[False,active[:-1]]];matches=starts[abs(starts-true_starts[0])<=.25] if event else []
   out.append(dict(seed=seed,split='synthetic_train' if seed<60 else 'synthetic_validation',kind=kind,corruption=corruption,hop_seconds=hop,method=method,samples=len(times),position_mse_px=float(np.mean(np.sum((p-target)**2,axis=1))),velocity_mse_px_s=float(np.mean(np.sum((v-velocity)**2,axis=1))),event_eligible=event,event_recalled=bool(len(matches)) if event else None,false_alarm_fraction=float(np.mean(active[quiet])) if quiet.any() else None))
 return out

def synthetic(monitor):
 result=pool(synthetic_worker,range(90),monitor,workers=12);rows=sorted([r for rr in result for r in rr],key=lambda r:(r['seed'],r['hop_seconds'],r['method']));save(R/'synthetic_motion.parquet',rows);summary=[]
 for part in ['synthetic_train','synthetic_validation']:
  for hop in [.125,.25]:
   for method in ['nearest','cv','switching']:
    rs=[r for r in rows if (r['split'],r['hop_seconds'],r['method'])==(part,hop,method)];events=[r for r in rs if r['event_eligible']];quiet=[r for r in rs if r['false_alarm_fraction'] is not None];summary.append(dict(split=part,hop_seconds=hop,method=method,episodes=len(rs),position_rmse_px=float(np.sqrt(np.mean([r['position_mse_px'] for r in rs]))),velocity_rmse_px_s=float(np.sqrt(np.mean([r['velocity_mse_px_s'] for r in rs]))),event_recall=float(np.mean([r['event_recalled'] for r in events])) if events else None,quiet_false_alarm_fraction=float(np.mean([r['false_alarm_fraction'] for r in quiet])) if quiet else None))
 atomic_json(R/'synthetic_motion.json',dict(summary=summary,independent_real_reference=False,reference='Exact analytic motion, shared truth/corruptions with earlier Track 1 diagnostics',methods_matched_at_identical_cached_cutoffs=True,promoted=False))
