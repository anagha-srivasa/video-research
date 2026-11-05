"""Causal functional-state checks against independently computed RGB feature-motion proxies."""
from collections import defaultdict
import json,time
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_index,save_shard
from track1.numerics import basis
from refinement.protocol import rank
from .common import config,path,allowed,split,pool,report,LOG,identity
from .appearance import stream
from .temporal import fit


def project(x,camera):
 import cv2
 x=np.asarray(x,float);shape=x.shape;xx=np.c_[x.reshape(-1,2),np.ones(x.size//2)];p,_=cv2.projectPoints(xx,np.zeros(3),np.zeros(3),np.asarray(camera['intrinsic_native']),np.asarray(camera['distortion']));return p.reshape(shape)

def velocity_native(position,velocity,camera):
 eps=1e-6;return (project(position+velocity*eps,camera)-project(position-velocity*eps,camera))/(2*eps)


def select_episodes(v,rows,state):
 tracks=defaultdict(list)
 for r in rows:
  if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
 candidates=[]
 for track,rs in tracks.items():
  times=np.array([r['timestamp_us'] for r in rs]);ids=np.array([r['row_index'] for r in rs])
  if len(rs)<40:continue
  for r in [r for r in rs if r['source_kind']=='manual_sparse']:
   start=r['timestamp_us'];end=start+1500000;local=times[(times>=start)&(times<=end)]
   if len(local)<25 or local[-1]<end-100000 or max(np.diff(local))>150000:continue
   before=(times>=start-1000000)&(times<=start)
   if before.sum()<15 or times[before][-1]-times[before][0]<650000:continue
   ix=ids[(times>=start)&(times<=end)];p=state['centre_native'][ix];ts=local/1e6;speed=float(np.median(np.linalg.norm(np.diff(p,axis=0)/np.diff(ts)[:,None],axis=1)));category='stationary_proxy' if speed<30 else 'fast_proxy' if speed>250 else 'moderate_proxy';candidates.append(dict(video_id=v,episode_uid=rank(track+str(start))[:24],entity_track_id=track,start_us=start,end_us=end,mask_uid=r['mask_uid'],entity_name=r['entity_name'],category=category,speed_native_px_per_s=speed,row_index=r['row_index']))
 chosen=[];used=set()
 for category in ['fast_proxy','stationary_proxy','moderate_proxy']:
  for r in sorted([r for r in candidates if r['category']==category],key=lambda r:rank(r['episode_uid'])):
   if r['entity_track_id'] in used:continue
   chosen.append(r);used.add(r['entity_track_id'])
   if sum(r['category']==category for r in chosen)>=7:break
 for r in sorted(candidates,key=lambda r:rank(r['episode_uid'])):
  if len(chosen)>=20:break
  if r['entity_track_id'] not in used:chosen.append(r);used.add(r['entity_track_id'])
 return chosen[:20],tracks


def worker(job):
 v,step,prefix=(job,.05,'') if isinstance(job,str) else job
 import cv2
 from PIL import Image,ImageDraw
 from pipeline import Dataset
 cv2.setNumThreads(1);allowed([v]);sp=ROOT/'derived/track1_refined_v2/spatial';s=load_shard(sp/f'{v}.pt');rows=pq.read_table(sp/f'{v}.parquet').to_pylist();episodes,tracks=select_episodes(v,rows,s);camera=json.loads((sp/f'{v}.camera.json').read_text());width,height=camera['native_size'];resize=np.array([960/width,540/height]);frames=pq.read_table(ROOT/f'canonical/v1/frames.{v}.parquet',columns=['timestamp_us']).column(0).to_numpy();wanted=defaultdict(list);contexts={}
 for e in episodes:
  q=np.arange(e['start_us'],e['end_us']+1,int(round(step*1e6)));ix=np.clip(np.searchsorted(frames,q),0,len(frames)-1);times=np.unique(frames[ix]);times=np.unique(np.r_[e['start_us'],times]);e['frame_times_us']=times.tolist();contexts[e['episode_uid']]=dict(episode=e,positions=[],times=[],survival=[],inliers=[],images=[],initial_count=0)
  for t in times:wanted[int(t)].append(e['episode_uid'])
 with Dataset() as ds:
  for c in contexts.values():
   mask=ds.get_mask(c['episode']['mask_uid']);c['mask']=cv2.resize(mask.astype(np.uint8),(960,540),interpolation=cv2.INTER_NEAREST)*255
 for number,(ts,rgb) in enumerate(stream(v,wanted,960,540)):
  if number%100==0:LOG.info('%s image-flow frames %d/%d',v,number,len(wanted))
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  for uid in wanted[ts]:
   c=contexts[uid];e=c['episode']
   if 'previous' not in c:
    mask=cv2.erode(c['mask'],np.ones((3,3),np.uint8));points=cv2.goodFeaturesToTrack(gray,maxCorners=160,qualityLevel=.01,minDistance=4,mask=mask);c['initial_count']=0 if points is None else len(points);c['points']=points;c['previous']=gray.copy();c['centre']=s['centre_native'][e['row_index']].astype(float)*resize;c['time0']=ts;c['active']=points is not None and len(points)>=6
   elif c['active']:
    p=c['points'];q,status,error=cv2.calcOpticalFlowPyrLK(c['previous'],gray,p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));back,ok,_=cv2.calcOpticalFlowPyrLK(gray,c['previous'],q,None,winSize=(21,21),maxLevel=3);good=(status.ravel()>0)&(ok.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<1.)&(q[:,0,0]>=0)&(q[:,0,0]<960)&(q[:,0,1]>=0)&(q[:,0,1]<540);p=p[good];q=q[good]
    if len(p)<6:c['active']=False
    else:
     a,inlier=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99);fraction=float(inlier.mean()) if inlier is not None else 0.
     if a is None or fraction<.5:c['active']=False
     else:c['centre']=a[:,:2]@c['centre']+a[:,2];c['points']=q[inlier.ravel()>0];c['inliers'].append(fraction)
    c['previous']=gray.copy()
   if c['active']:
    c['positions'].append(c['centre'].copy()/resize);c['times'].append(ts);c['survival'].append(len(c['points'])/max(c['initial_count'],1))
   if len(c['images'])<6 and (len(c['times'])==1 or ts-e['start_us']>=len(c['images'])*250000):
    im=rgb.copy()
    if c['active']:
     for point in c['points'][:,0]:cv2.circle(im,tuple(np.rint(point).astype(int)),2,(50,255,100),-1)
     cv2.circle(im,tuple(np.rint(c['centre']).astype(int)),5,(255,90,40),2)
    c['images'].append((ts,im))
 metrics=[];review=[];raw_traces=[];traces=defaultdict(list);model=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text());base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text())['temporal']['options'];candidate=json.loads((path('reports')/'temporal_candidate.json').read_text())['options'];scale=np.asarray(model['feature_scale'])
 for uid,c in contexts.items():
  e=c['episode'];ts=np.array(c['times'],dtype=np.int64);positions=np.array(c['positions']);accepted=len(ts)>=20 and ts[-1]-ts[0]>=1000000 and c['survival'][-1]>=.25;review.append(dict(**e,feature_points_initial=c['initial_count'],reference_samples=len(ts),reference_duration_us=int(ts[-1]-ts[0]) if len(ts)>1 else 0,final_survival=c['survival'][-1] if c['survival'] else 0.,algorithmic_reference_accepted=bool(accepted),independent_manual_review='pending',reference_kind='RGB forward-backward checked feature flow and RANSAC similarity; not physical ground truth'))
  sheet=Image.new('RGB',(1440,600),(20,20,20));draw=ImageDraw.Draw(sheet)
  for j,(t,im) in enumerate(c['images']):
   panel=Image.fromarray(im);panel.thumbnail((480,270));sheet.paste(panel,(j%3*480,j//3*300));draw.text((j%3*480+4,j//3*300+274),f'{v} {t/1e6:.3f}s {e["entity_name"][:20]}',fill='white')
  episode_folder=path('reports')/('dense_episodes' if prefix else 'episodes');episode_folder.mkdir(exist_ok=True);sheet.save(episode_folder/f'{prefix}episode.{uid}.jpg',quality=88)
  for tt,pp in zip(ts,positions):raw_traces.append(dict(video_id=v,episode_uid=uid,timestamp_us=int(tt),position_native_x=float(pp[0]),position_native_y=float(pp[1]),algorithmic_reference_accepted=bool(accepted)))
  if not accepted:continue
  rs=tracks[e['entity_track_id']];rtime=np.array([r['timestamp_us'] for r in rs]);ids=np.array([r['row_index'] for r in rs]);prior=np.array([r['provenance_weight'] for r in rs]);quality=np.array([r['geometry_weight'] for r in rs]);tref=(ts-ts[0])/1e6
  # Independent RGB reference derivative is checked at three smoothing scales.
  for reference_window in [5,7,11]:
   # Smooth position on a uniform physical-time grid before differentiating.
   uniform=np.arange(tref[0],tref[-1]+1e-6,step);pos=np.stack([np.interp(uniform,tref,positions[:,k]) for k in range(2)],axis=1);refpos=savgol_filter(pos,reference_window,2,axis=0);refvel=savgol_filter(pos,reference_window,2,deriv=1,delta=step,axis=0)
   for j in range(reference_window//2,len(uniform)-reference_window//2,max(1,int(round(.1/step)))):
    cutoff=int(ts[0]+uniform[j]*1e6);keep=(rtime>=cutoff-1000000)&(rtime<=cutoff)
    if keep.sum()<15 or rtime[keep][-1]-rtime[keep][0]<650000:continue
    local=ids[keep];t=(rtime[keep]-(cutoff-1000000))/1e6;past_pos=s['centre_native'][local].astype(float);recent=t>=max(t[-1]-.2,t[0]);design=np.c_[np.ones(recent.sum()),t[recent]-t[-1]];cv=np.linalg.lstsq(design,past_pos[recent],rcond=None)[0];preds={'constant_position':(past_pos[-1],np.zeros(2),None),'constant_velocity':(cv[0],cv[1],None)}
    model_options=[('v2_cubic',base),('endpoint_cubic',candidate)]
    if prefix and (path('reports')/'derivative_candidate.json').exists():model_options.append(('slope_cubic',json.loads((path('reports')/'derivative_candidate.json').read_text())['candidate']['options']))
    for method,option in model_options:
     result=fit(t,s['h'][local].astype(float),s['variance'][local].astype(float)*model['kappa_cov'],prior[keep],quality[keep],option,scale);coeff=result['coefficients'];x=basis(1.)@coeff;vel=2*basis(1.,1)@coeff;preds[method]=(project(x[:2],camera),velocity_native(x[:2],vel[:2],camera),coeff)
    reference_position=refpos[j];reference_velocity=refvel[j];speed=np.linalg.norm(reference_velocity);ref_speed_scale=max(speed,30.)
    for method,(position,velocity,coeff) in preds.items():
     angle=float(np.degrees(np.arccos(np.clip(np.dot(velocity,reference_velocity)/max(np.linalg.norm(velocity)*speed,1e-20),-1,1)))) if speed>=30 and np.linalg.norm(velocity)>=1 else None
     metrics.append(dict(video_id=v,split=split(v),episode_uid=uid,entity_name=e['entity_name'],category=e['category'],timestamp_us=cutoff,reference_smoothing_samples=reference_window,method=method,position_error_frame_diagonal=float(np.linalg.norm(position-reference_position)/np.hypot(width,height)),velocity_error_native_px_s=float(np.linalg.norm(velocity-reference_velocity)),velocity_error_relative_motion_scale=float(np.linalg.norm(velocity-reference_velocity)/ref_speed_scale),velocity_direction_error_degrees=angle,reference_speed_native_px_s=float(speed),estimated_speed_native_px_s=float(np.linalg.norm(velocity)),horizon_seconds=0.))
     for horizon in [.1,.25,.5]:
      jj=int(round(j+horizon/step))
      if jj>=len(uniform):continue
      future_ref=refpos[jj]
      if coeff is not None:future=project((basis(1+2*horizon)@coeff)[:2],camera)
      else:future=position+horizon*velocity
      metrics.append(dict(video_id=v,split=split(v),episode_uid=uid,entity_name=e['entity_name'],category=e['category'],timestamp_us=cutoff,reference_smoothing_samples=reference_window,method=method,position_error_frame_diagonal=float(np.linalg.norm(future-future_ref)/np.hypot(width,height)),velocity_error_native_px_s=None,velocity_error_relative_motion_scale=None,velocity_direction_error_degrees=None,reference_speed_native_px_s=float(speed),estimated_speed_native_px_s=float(np.linalg.norm(velocity)),horizon_seconds=horizon))
   if reference_window==7:
    for t,p,vel in zip(uniform,refpos,refvel):traces['time'].append((ts[0]+t*1e6)/1e6);traces['position'].append(p);traces['velocity'].append(vel)
 save_index(path('reports')/f'{prefix}flow_raw.{v}.parquet',raw_traces);save_index(path('reports')/f'{prefix}motion.{v}.parquet',metrics);save_index(path('reports')/f'{prefix}episodes.{v}.parquet',review);save_shard(path('derived')/f'{prefix}flow_reference.{v}.pt',{k:np.asarray(a,np.float64) for k,a in traces.items()});return dict(video_id=v,episodes=len(episodes),algorithmic_reference_accepted=sum(r['algorithmic_reference_accepted'] for r in review),metric_rows=len(metrics))


def run(monitor,dense=False):
 cfg=config();prefix='dense_' if dense else '';step=.02 if dense else .05;results=pool(worker,[(v,step,prefix) for v in cfg['train']+cfg['validation']],monitor,workers=2 if dense else 3);records=[];review=[]
 for v in cfg['train']+cfg['validation']:
  records+=pq.read_table(path('reports')/f'{prefix}motion.{v}.parquet').to_pylist();review+=pq.read_table(path('reports')/f'{prefix}episodes.{v}.parquet').to_pylist()
 save_index(path('reports')/(prefix+'motion.parquet'),records);save_index(path('reports')/(prefix+'episodes.parquet'),review);summary={}
 for window in [5,7,11]:
  summary[str(window)]={}
  for method in ['v2_cubic','endpoint_cubic','constant_position','constant_velocity']+(['slope_cubic'] if dense else []):
   summary[str(window)][method]={}
   for horizon in [0.,.1,.25,.5]:
    byvideo={}
    for v in cfg['validation']:
     rs=[r for r in records if r['video_id']==v and r['method']==method and r['reference_smoothing_samples']==window and r['horizon_seconds']==horizon]
     if rs:byvideo[v]=dict(n=len(rs),position_nrmse=float(np.sqrt(np.mean([r['position_error_frame_diagonal']**2 for r in rs]))),velocity_error_px_s=float(np.mean([r['velocity_error_native_px_s'] for r in rs])) if horizon==0 else None)
    summary[str(window)][method][str(horizon)]=byvideo
 report(prefix+'motion_reference.json',dict(reference_sampling_hz=1/step,smoothing_support_seconds={str(w):(w-1)*step for w in [5,7,11]},videos=results,validation_metrics=summary,reference='RGB pyramidal Lucas-Kanade with forward-backward checks, foreground initialization, RANSAC similarity, independent 5/7/11-sample Savitzky-Golay reference derivatives.',limits=['Algorithmically independent of interpolated mask positions, but not independently verified physical ground truth.','Feature-motion-propagated initial centroid is not the centroid of a deforming silhouette.','Reference failures are retained in the episode ledger; acceptance is based on feature survival/coverage, never model error.','Manual review of episode sheets is pending; no certified onset/short-event recall or true velocity claim.','Future-image observations are used only as references after the model cutoff; no Track 2 model is trained.'],test_videos_used=False,state='experimental_not_finalized',**identity()))
