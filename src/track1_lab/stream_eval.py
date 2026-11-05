"""Evaluate the actual stored 4 Hz states against dense RGB reference proxies."""
from collections import defaultdict
import json
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_index
from track1.numerics import basis
from .common import config,path,allowed,split,report,LOG,identity
from .motion import project,velocity_native
from .working import directory


def run(monitor):
 cfg=config();records=[];ledger=[];methods=['v2','working_fixed','half_second_cubic','one_second_degree5','constant_position','constant_velocity']
 for v in cfg['train']+cfg['validation']:
  allowed([v]);raw=pq.read_table(path('reports')/f'dense_flow_raw.{v}.parquet').to_pylist();byepisode=defaultdict(list)
  for r in raw:byepisode[r['episode_uid']].append(r)
  episodes=pq.read_table(path('reports')/f'dense_episodes.{v}.parquet').to_pylist();stores={};indices={}
  for name in methods[:4]:
   folder=ROOT/'derived/track1_refined_v2/functional' if name=='v2' else directory('functional_fixed') if name=='working_fixed' else path('derived')/name;stores[name]=load_shard(folder/f'{v}.pt');indices[name]={(r['entity_track_id'],r['window_end_us']):r for r in pq.read_table(folder/f'{v}.parquet').to_pylist()}
  source=load_shard(directory('spatial')/f'{v}.pt');tracks=defaultdict(list)
  for r in pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist():
   if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
  camera=json.loads((directory('spatial')/f'{v}.camera.json').read_text());diag=np.hypot(*camera['native_size'])
  for episode in episodes:
   if not episode['algorithmic_reference_accepted']:continue
   uid=episode['episode_uid'];rawref=byepisode[uid];ts=np.array([r['timestamp_us'] for r in rawref]);time=(ts-ts[0])/1e6;positions=np.array([[r['position_native_x'],r['position_native_y']] for r in rawref]);uniform=np.arange(0,time[-1]+1e-6,.02);position=np.stack([np.interp(uniform,time,positions[:,j]) for j in range(2)],axis=1);rs=tracks[episode['entity_track_id']];source_times=np.array([r['timestamp_us'] for r in rs]);source_ids=np.array([r['row_index'] for r in rs]);first=int(np.ceil((ts[0]+120000)/250000))*250000;last=int(np.floor((ts[-1]-120000)/250000))*250000
   for cutoff in range(first,last+1,250000):
    key=(episode['entity_track_id'],cutoff);rows={name:indices[name].get(key) for name in methods[:4]};supported=all(r is not None and bool(r.get('support_valid',r.get('valid'))) for r in rows.values());ledger.append(dict(video_id=v,episode_uid=uid,timestamp_us=cutoff,all_methods_present=all(r is not None for r in rows.values()),all_temporally_supported=supported))
    if not supported:continue
    past=(source_times<=cutoff)&(source_times>=cutoff-200000)
    if past.sum()<3:continue
    pp=source['centre_native'][source_ids[past]].astype(float);tt=(source_times[past]-cutoff)/1e6;line=np.linalg.lstsq(np.c_[np.ones(len(tt)),tt],pp,rcond=None)[0];relative=(cutoff-ts[0])/1e6
    for width in [5,7,11]:
     smooth=savgol_filter(position,width,2,axis=0);velocity=savgol_filter(position,width,2,deriv=1,delta=.02,axis=0);reference_velocity=np.array([np.interp(relative,uniform,velocity[:,j]) for j in range(2)])
     for name in methods:
      if name in stores:
       r=rows[name];c=stores[name]['coefficients'][r['row_index']].astype(float);window=r.get('window_seconds',1.);degree=len(c)-1;x=(basis(1.,degree=degree)@c)[:2];vv=(basis(1.,1,degree=degree)@c)[:2]*(2/window);pred_velocity=velocity_native(x,vv,camera)
      else:c=None;pred_velocity=np.zeros(2) if name=='constant_position' else line[1]
      for horizon in [0.,.1,.25,.5]:
       target_time=relative+horizon
       if target_time>uniform[-1]-.04:continue
       reference=np.array([np.interp(target_time,uniform,smooth[:,j]) for j in range(2)])
       if c is None:pred=pp[-1] if name=='constant_position' else line[0]+horizon*line[1]
       else:pred=project((basis(1+2*horizon/window,degree=degree)@c)[:2],camera)
       records.append(dict(video_id=v,split=split(v),episode_uid=uid,category=episode['category'],entity_name=episode['entity_name'],state_end_us=cutoff,method=name,horizon_seconds=horizon,reference_smoothing_samples=width,position_error_frame_diagonal=float(np.linalg.norm(pred-reference)/diag),velocity_error_native_px_s=float(np.linalg.norm(pred_velocity-reference_velocity)) if horizon==0 else None,reference_speed_native_px_s=float(np.linalg.norm(reference_velocity)),predicted_speed_native_px_s=float(np.linalg.norm(pred_velocity))))
  monitor.check();LOG.info('Stored-state stream evaluation %s',v)
 save_index(path('reports')/'stream_metrics.parquet',records);save_index(path('reports')/'stream_eligibility.parquet',ledger);summary={}
 for width in [5,7,11]:
  summary[str(width)]={}
  for name in methods:
   summary[str(width)][name]={}
   for horizon in [0.,.1,.25,.5]:
    byvideo={}
    for v in cfg['validation']:
     rs=[r for r in records if (r['video_id'],r['method'],r['horizon_seconds'],r['reference_smoothing_samples'])==(v,name,horizon,width)]
     if rs:byvideo[v]=dict(n=len(rs),episodes=len(set(r['episode_uid'] for r in rs)),position_nrmse=float(np.sqrt(np.mean([r['position_error_frame_diagonal']**2 for r in rs]))),velocity_rmse_native_px_s=float(np.sqrt(np.mean([r['velocity_error_native_px_s']**2 for r in rs]))) if horizon==0 else None)
    if byvideo:summary[str(width)][name][str(horizon)]=dict(position_nrmse=float(np.mean([r['position_nrmse'] for r in byvideo.values()])),velocity_rmse_native_px_s=float(np.mean([r['velocity_rmse_native_px_s'] for r in byvideo.values()])) if horizon==0 else None,by_video=byvideo)
 categories={}
 for part in ['train','validation']:
  for category in ['fast_proxy','moderate_proxy','stationary_proxy']:
   allrows=[]
   for v in cfg[part]:allrows+=pq.read_table(path('reports')/f'dense_episodes.{v}.parquet').to_pylist()
   es=[r for r in allrows if r['category']==category];categories[part+'/'+category]=dict(selected=len(es),accepted=sum(r['algorithmic_reference_accepted'] for r in es))
 report('stream_evaluation.json',dict(validation_metrics=summary,reference_selection=categories,eligible_state_cutoffs=sum(r['all_temporally_supported'] for r in ledger),attempted_reference_cutoffs=len(ledger),limits=['Actual cached 4Hz state timestamps, matched source cutoff and reference rows. Geometry-invalid states are retained in this position/velocity diagnostic; common temporal-support criteria are required.','Reference is RGB foreground-affine feature motion, not human physical ground truth.','Fast-motion reference acceptance is reported explicitly; favorable aggregate errors cannot establish fast-hand accuracy.','Future horizons extrapolate the encoder functions for a diagnostic comparison with persistence and constant velocity; no Track 2 model is trained.'],state='experimental_not_finalized',**identity()))
