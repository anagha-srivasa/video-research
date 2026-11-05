"""Build the separate causal stream and reconcile all proposed pilot probes."""
from collections import defaultdict,OrderedDict
import json
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,digest
from track1.artifacts import load_shard,save_shard,save_index
from refinement.data import holdout
from .common import config,path,allowed,split,pool,report,identity,audit,LOG
from .causal_kinematics import estimate,KinematicState
from .pilot32 import selected_cases

def source_eligibility(time,hidden):
 t=np.asarray(time)[(~hidden)&(np.asarray(time)>=.5)]
 if not np.any(hidden):return False,'no_holdout_target'
 if len(t)<8:return False,'too_few_source_observations'
 if t[-1]-t[0]<.3:return False,'insufficient_source_span'
 if np.diff(np.r_[.5,t,1.]).max()>.175:return False,'source_observation_gap'
 return True,None

class KinematicStore:
 """Full stream only: do not substitute this reader into holdout protocols."""
 def __init__(self):self.cache=OrderedDict()
 def close(self):self.cache.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def get_state(self,video,entity_track_id,timestamp_us,robust=False):
  allowed([video])
  if video not in self.cache:
   if len(self.cache)>=2:self.cache.popitem(last=False)
   folder=path('derived')/'causal_kinematics';rows=pq.read_table(folder/f'{video}.parquet').to_pylist();self.cache[video]=(load_shard(folder/f'{video}.pt'),{(r['entity_track_id'],r['state_time_us']):r for r in rows})
  self.cache.move_to_end(video);data,index=self.cache[video];r=index[(entity_track_id,timestamp_us)];i=r['row_index'];prefix='robust_' if robust else '';valid=r[prefix+'support_valid'];values=[data[prefix+k][i].astype(float) if valid else None for k in ['position','velocity','covariance']]
  return KinematicState(*values,'robust_local_linear' if robust else r['method'],valid,r['failure_reason'],r['observation_count'],r['latest_input_us']/1e6 if r['latest_input_us'] is not None else None,r['age_seconds'])

def reconcile():
 old=pq.read_table(path('reports')/'compound32_eligibility.parquet').to_pylist();bykey={(r['video_id'],r['entity_track_id'],r['window_end_us'],r['protocol']):r for r in old};rows=[]
 for v in config()['train']+config()['validation']:
  for c in selected_cases(v):
   for protocol in ['single','gap100','endpoint100']:
    eligible,reason=source_eligibility(c['time'],holdout(c['time'],protocol,.875));key=(v,c['entity_track_id'],c['window_end_us'],protocol);r=dict(bykey.get(key,dict(video_id=v,split=split(v),entity_track_id=c['entity_track_id'],window_end_us=c['window_end_us'],protocol=protocol,supported=False)));r.update(proposed=True,source_support_eligible=eligible,source_ineligibility_reason=reason,independently_eligible=None,outcome='emitted' if r['supported'] else 'source_ineligible' if not eligible else 'candidate_failed_or_unsupported');rows.append(r)
 # Keep the historical ledger for audit; future compound builds record all attempts.
 save_index(path('reports')/'compound32_attempts.parquet',rows);summary={}
 for part in ['train','validation']:
  rs=[r for r in rows if r['split']==part];eligible=[r for r in rs if r['source_support_eligible']];emitted=[r for r in rs if r['supported']];summary[part]=dict(proposed=len(rs),source_support_eligible=len(eligible),emitted=len(emitted),missing_or_invalid=len(rs)-len(emitted),failure_fraction_all_proposed=1-len(emitted)/len(rs),coverage_among_source_supported=sum(r['supported'] for r in eligible)/len(eligible),independently_eligible_coverage=None,outcomes={reason:sum(r['outcome']==reason for r in rs) for reason in sorted({r['outcome'] for r in rs})},source_ineligibility_reasons={reason:sum(r['source_ineligibility_reason']==reason for r in rs) for reason in sorted({r['source_ineligibility_reason'] for r in rs if r['source_ineligibility_reason']})})
 report('attempt_denominators.json',dict(summary=summary,policy='Source support is computed before candidate fitting from retained observation times, not candidate success. This selected pilot is not an independently preregistered full-dataset eligibility population. No-target probes remain recorded.',conditional_accuracy_source='research_report_fixes.json; conditional on emitted states; not silently extended to failed probes',**identity()));return summary

def video(v):
 allowed([v]);source=path('derived')/'spatial';s=load_shard(source/f'{v}.pt');rows=pq.read_table(source/f'{v}.parquet').to_pylist();functional=pq.read_table(path('derived')/'functional_fixed'/f'{v}.parquet').to_pylist();tracks=defaultdict(list)
 for r in rows:
  if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
 grouped={}
 for track,rs in tracks.items():
  ids=np.array([r['row_index'] for r in rs]);grouped[track]=(np.array([r['timestamp_us']/1e6 for r in rs]),s['h'][ids,:2].astype(float),s['variance'][ids,:2].astype(float))
 arrays={name:[] for name in ['position','velocity','covariance','robust_position','robust_velocity','robust_covariance']};meta=[]
 for r in functional:
  t,p,var=grouped[r['entity_track_id']];cutoff=r['window_end_us']/1e6;lo=np.searchsorted(t,cutoff-.2);hi=np.searchsorted(t,cutoff,side='right');states=[estimate(t[lo:hi],p[lo:hi],var[lo:hi],cutoff,robust=robust) for robust in [False,True]]
  for prefix,state in zip(['','robust_'],states):
   arrays[prefix+'position'].append(state.position if state.support_valid else np.zeros(2));arrays[prefix+'velocity'].append(state.velocity if state.support_valid else np.zeros(2));arrays[prefix+'covariance'].append(state.covariance if state.support_valid else np.zeros((4,4)))
  state=states[0];meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=r['entity_track_id'],state_time_us=r['window_end_us'],latest_input_us=int(round(state.latest_input_time*1e6)) if state.latest_input_time is not None else None,method=state.method,support_valid=state.support_valid,robust_support_valid=states[1].support_valid,failure_reason=state.failure_reason,observation_count=state.observation_count,age_seconds=state.age_seconds,covariance_calibrated=False,derivative_reliable=None,coordinate_semantics='intrinsic_normalized_2d; velocity_per_second',representation_version='track1-v3.0.0-dev.hybrid1',source_scope='full_available_past_spatial_valid_observations; no holdout exclusions',tuning_split=split(v)))
 out=path('derived')/'causal_kinematics';out.mkdir(exist_ok=True);save_shard(out/f'{v}.pt',{k:np.asarray(a,np.float32) for k,a in arrays.items()});save_index(out/f'{v}.parquet',meta)
 assert all(r['latest_input_us'] is None or r['latest_input_us']<=r['state_time_us'] for r in meta)
 return dict(video_id=v,attempted=len(meta),supported=sum(r['support_valid'] for r in meta),source_sha256=digest(source/f'{v}.pt'),tensor_sha256=digest(out/f'{v}.pt'),index_sha256=digest(out/f'{v}.parquet'))

def run(monitor):
 audit();denominators=reconcile();jobs=pool(video,config()['train']+config()['validation'],monitor,workers=7);report('hybrid_architecture.json',dict(state='open_not_finalized',architecture='Option B: functional history plus separate causal kinematics',primary_kinematic_method='constant_velocity_local_ols',alternative='robust_local_linear; no automatic promotion',velocity_source='past position observations; independent of polynomial/residual derivatives',history_derivatives='diagnostics only',covariance_scope='conditional joint position/velocity estimate; not independently calibrated',acceleration='not estimated without adequate evidence',videos=jobs,denominators=denominators,ingestion_ready=False,track2_started=False,**identity()));monitor.check()

def evaluate_stream(monitor):
 """Reuse the RGB reference cohort; compare on the same supported cutoffs."""
 from scipy.signal import savgol_filter
 from .motion import project,velocity_native
 from .scientific import CandidateStore
 baseline=pq.read_table(path('reports')/'stream_metrics.parquet').to_pylist();records=[];ledger=[];counts=0;minimum_eigenvalue=0.;leak_checks=0
 for v in config()['train']+config()['validation']:
  meta=pq.read_table(path('derived')/'causal_kinematics'/f'{v}.parquet').to_pylist();data=load_shard(path('derived')/'causal_kinematics'/f'{v}.pt')
  for name,a in data.items():
   assert np.isfinite(a).all()
   if name.endswith('covariance'):
    eig=np.linalg.eigvalsh(a.astype(float));minimum_eigenvalue=min(minimum_eigenvalue,float(eig.min()));assert eig.min()>=-1e-8
  assert all(r['latest_input_us'] is None or r['latest_input_us']<=r['state_time_us'] for r in meta);counts+=len(meta)
  camera=json.loads((path('derived')/'spatial'/f'{v}.camera.json').read_text());diag=np.hypot(*camera['native_size']);episodes={r['episode_uid']:r for r in pq.read_table(path('reports')/f'dense_episodes.{v}.parquet').to_pylist()};raw=defaultdict(list)
  for r in pq.read_table(path('reports')/f'dense_flow_raw.{v}.parquet').to_pylist():raw[r['episode_uid']].append(r)
  reference={}
  used_uids={r['episode_uid'] for r in baseline if r['video_id']==v}
  for uid in used_uids:
   rr=raw[uid]
   ts=np.array([r['timestamp_us'] for r in rr]);tt=(ts-ts[0])/1e6;uniform=np.arange(0,tt[-1]+1e-6,.02);position=np.array([[r['position_native_x'],r['position_native_y']] for r in rr]);position=np.stack([np.interp(uniform,tt,position[:,j]) for j in range(2)],axis=1)
   reference[uid]=(ts[0],uniform,{w:(savgol_filter(position,w,2,axis=0),savgol_filter(position,w,2,deriv=1,delta=.02,axis=0)) for w in [5,7,11]})
  bycutoff=defaultdict(list)
  for r in baseline:
   if r['video_id']==v:bycutoff[(r['episode_uid'],r['state_end_us'])].append(r)
  with KinematicStore() as store:
   for (uid,cutoff),old in bycutoff.items():
    track=episodes[uid]['entity_track_id'];states={name:store.get_state(v,track,cutoff,robust) for name,robust in [('causal_cv',False),('causal_robust',True)]};supported=all(s.support_valid for s in states.values());ledger.append(dict(video_id=v,episode_uid=uid,state_time_us=cutoff,all_kinematic_methods_supported=supported))
    if not supported:continue
    records+=old;t0,uniform,curves=reference[uid];relative=(cutoff-t0)/1e6
    for width,horizon in sorted({(r['reference_smoothing_samples'],r['horizon_seconds']) for r in old}):
     smooth,velocity=curves[width];position_reference=np.array([np.interp(relative+horizon,uniform,smooth[:,j]) for j in range(2)]);velocity_reference=np.array([np.interp(relative,uniform,velocity[:,j]) for j in range(2)])
     for name,state in states.items():
      pred=project(state.position+horizon*state.velocity,camera);vv=velocity_native(state.position,state.velocity,camera);records.append(dict(video_id=v,split=split(v),episode_uid=uid,category=episodes[uid]['category'],entity_name=episodes[uid]['entity_name'],state_end_us=cutoff,method=name,horizon_seconds=horizon,reference_smoothing_samples=width,position_error_frame_diagonal=float(np.linalg.norm(pred-position_reference)/diag),velocity_error_native_px_s=float(np.linalg.norm(vv-velocity_reference)) if horizon==0 else None,reference_speed_native_px_s=float(np.linalg.norm(velocity_reference)),predicted_speed_native_px_s=float(np.linalg.norm(vv))))
  # Verify that the hybrid pilot reader does not regain hidden observations.
  with CandidateStore() as reader:
   pilot=pq.read_table(path('derived')/'compound32_topology'/f'{v}.parquet').to_pylist()
   for i in np.linspace(0,len(pilot)-1,min(12,len(pilot)),dtype=int):
    state=reader.get_state(v,int(i));assert not hasattr(state,'endpoint_velocity');r=pilot[i];case=reader.cache[v][2][(r['entity_track_id'],r['window_end_us'])];hidden=holdout(case['time'],r['protocol'],.875);original=case['h'][hidden].copy()
    try:
     case['h'][hidden]=1e8;again=reader.get_state(v,int(i));assert state.kinematics.support_valid==again.kinematics.support_valid
     if state.kinematics.support_valid:np.testing.assert_array_equal(state.kinematics.velocity,again.kinematics.velocity)
    finally:case['h'][hidden]=original
    leak_checks+=1
  monitor.check();LOG.info('Hybrid reader and stream verified %s',v)
 save_index(path('reports')/'stream_hybrid_metrics.parquet',records);save_index(path('reports')/'stream_hybrid_support.parquet',ledger);summary={}
 for width in [5,7,11]:
  summary[str(width)]={}
  for name in sorted({r['method'] for r in records}):
   summary[str(width)][name]={}
   for horizon in [0.,.1,.25,.5]:
    videos={}
    for v in config()['validation']:
     rr=[r for r in records if (r['video_id'],r['method'],r['reference_smoothing_samples'],r['horizon_seconds'])==(v,name,width,horizon)]
     if rr:videos[v]=dict(n=len(rr),position_nrmse=float(np.sqrt(np.mean([r['position_error_frame_diagonal']**2 for r in rr]))),velocity_rmse_native_px_s=float(np.sqrt(np.mean([r['velocity_error_native_px_s']**2 for r in rr]))) if horizon==0 else None)
    if videos:summary[str(width)][name][str(horizon)]=dict(position_nrmse=float(np.mean([r['position_nrmse'] for r in videos.values()])),velocity_rmse_native_px_s=float(np.mean([r['velocity_rmse_native_px_s'] for r in videos.values()])) if horizon==0 else None,by_video=videos)
 report('hybrid_stream_evaluation.json',dict(validation_metrics=summary,attempted_reference_cutoffs=len(ledger),matched_supported_cutoffs=sum(r['all_kinematic_methods_supported'] for r in ledger),kinematic_states_checked=counts,minimum_covariance_eigenvalue=minimum_eigenvalue,hidden_target_independence_checks=leak_checks,velocity_calibrated=False,limits=['Matched existing RGB foreground-flow reference; no independent fast validation motion reference.','Covariance PSD/finiteness is not statistical calibration.','Future positions use an explicit constant-velocity diagnostic, not polynomial extrapolation and not Track 2 training.','Prototype compound-history reader preserves its holdout exclusions; the full kinematic stream must not be joined into a holdout evaluation.'],**identity()))
