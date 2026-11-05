"""Train-only tuning of causal slope regularization against RGB feature-motion proxies."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_index
from track1.numerics import basis
from .common import config,path,allowed,split,report,LOG,identity
from .temporal import fit
from .motion import project,velocity_native
_SAMPLES=None;_SCALE=None

def initialize():
 global _SAMPLES,_SCALE
 cfg=config();model=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text());_SCALE=np.asarray(model['feature_scale']);samples=[]
 for v in allowed(cfg['train']+cfg['validation']):
  ref=load_shard(path('derived')/f'flow_reference.{v}.pt');episodes=pq.read_table(path('reports')/f'episodes.{v}.parquet').to_pylist();s=load_shard(ROOT/f'derived/track1_refined_v2/spatial/{v}.pt');rows=pq.read_table(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').to_pylist();tracks=defaultdict(list)
  for r in rows:
   if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
  camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());offset=0
  for e in episodes:
   if not e['algorithmic_reference_accepted']:continue
   n=len(np.arange(0,e['reference_duration_us']/1e6+1e-6,.05));assert abs(ref['time'][offset]-e['start_us']/1e6)<1e-6;pos=ref['position'][offset:offset+n];vel=ref['velocity'][offset:offset+n];rts=ref['time'][offset:offset+n];offset+=n;rs=tracks[e['entity_track_id']];times=np.array([r['timestamp_us'] for r in rs]);ids=np.array([r['row_index'] for r in rs])
   for j in range(3,n-3,2):
    cutoff=int(round(rts[j]*1e6));keep=(times>=cutoff-1000000)&(times<=cutoff)
    if keep.sum()<15 or times[keep][-1]-times[keep][0]<650000:continue
    local=ids[keep];t=(times[keep]-(cutoff-1000000))/1e6;samples.append(dict(video_id=v,split=split(v),episode_uid=e['episode_uid'],category=e['category'],timestamp_us=cutoff,t=t,h=s['h'][local].astype(float),variance=s['variance'][local].astype(float)*model['kappa_cov'],prior=np.array([r['provenance_weight'] for r in np.array(rs,dtype=object)[keep]]),quality=np.array([r['geometry_weight'] for r in np.array(rs,dtype=object)[keep]]),camera=camera,reference_position=pos[j],reference_velocity=vel[j],future={str(h):pos[j+int(round(h/.05))] for h in [.1,.25,.5] if j+int(round(h/.05))<n}))
  assert offset==len(ref['time']),(v,offset,len(ref['time']))
 _SAMPLES=samples


def evaluate(options,include_validation=False,details=False):
 if _SAMPLES is None:initialize()
 records=[]
 for s in _SAMPLES:
  if s['split']=='validation' and not include_validation:continue
  result=fit(s['t'],s['h'],s['variance'],s['prior'],s['quality'],options,_SCALE);c=result['coefficients'];x=(basis(1.)@c)[:2];v=(2*basis(1.,1)@c)[:2];position=project(x,s['camera']);velocity=velocity_native(x,v,s['camera']);error=float(np.linalg.norm(velocity-s['reference_velocity']));record=dict(video_id=s['video_id'],split=s['split'],episode_uid=s['episode_uid'],category=s['category'],timestamp_us=s['timestamp_us'],velocity_error_px_s=error,position_error_native_px=float(np.linalg.norm(position-s['reference_position'])),condition=result['condition'])
  for horizon,target in s['future'].items():record['future_'+horizon+'_position_error_native_px']=float(np.linalg.norm(project((basis(1+2*float(horizon))@c)[:2],s['camera'])-target))
  records.append(record)
 summary={}
 for part in ['train','validation']:
  byvideo={}
  for video in config()[part]:
   rs=[r for r in records if r['video_id']==video]
   if rs:byvideo[video]=dict(n=len(rs),velocity_error_px_s=float(np.sqrt(np.mean([r['velocity_error_px_s']**2 for r in rs]))),position_error_native_px=float(np.sqrt(np.mean([r['position_error_native_px']**2 for r in rs]))))
  if byvideo:summary[part]=dict(velocity_error_px_s=float(np.mean([r['velocity_error_px_s'] for r in byvideo.values()])),position_error_native_px=float(np.mean([r['position_error_native_px'] for r in byvideo.values()])),by_video=byvideo)
 result=dict(options=options,metrics=summary,objective=summary['train']['velocity_error_px_s'])
 if details:result['records']=records
 return result


def run(monitor):
 from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
 import multiprocessing
 base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text())['temporal']['options'];endpoint=json.loads((path('reports')/'temporal_candidate.json').read_text())['options'];options=[base,endpoint]
 for parent in [base,endpoint]:
  for strength in [.001,.01,.1,1.]:
   for width in [.15,.3]:
    for value in [0.,.1]:options.append(dict(parent,slope_strength=strength,slope_window=width,endpoint_value_strength=value,slope_groups=['position']))
 results=[];ex=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize);jobs={ex.submit(evaluate,o):i for i,o in enumerate(options)}
 try:
  while jobs:
   done,_=wait(jobs,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
   for future in done:
    i=jobs.pop(future);r=future.result();results.append(r);LOG.info('Derivative train trial %d velocity RMSE %.3f',i,r['objective'])
 except BaseException:
  for p in ex._processes.values():p.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 ex.shutdown();initial=evaluate(base);eligible=[r for r in results if r['metrics']['train']['position_error_native_px']<=1.1*initial['metrics']['train']['position_error_native_px']];best=min(eligible,key=lambda r:r['objective']);candidate=evaluate(best['options'],True,True);baseline=evaluate(base,True,True)
 for name,r in [('baseline',baseline),('candidate',candidate)]:save_index(path('reports')/f'derivative_{name}_cases.parquet',r.pop('records'))
 report('derivative_candidate.json',dict(candidate=candidate,baseline=baseline,train_trials=results,selection='Training-participant RGB-proxy velocity RMSE only, with 10% training position-error guard. Validation scores are descriptive research diagnostics; previously observed validation participants are not a fresh benchmark.',covariance_limit='Slope and endpoint targets are functions of already retained observations. They are regularizers, not independent extra measurements; returned inverse-Gram covariance needs fresh empirical calibration.',reference_limit='RGB foreground-affine motion, not independent physical ground truth. Reference smoothing is fixed at 7 samples here; original audit reports baseline sensitivity at 5/7/11 samples.',state='experimental_not_finalized',**identity()))
