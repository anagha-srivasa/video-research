"""Kalman comparison, independent-reference validation and evidence snapshot."""
from collections import defaultdict
import csv,json
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from pipeline.common import ROOT,digest
from track1.artifacts import load_shard,save_shard,save_index
from .common import config,path,allowed,split,report,identity,LOG
from .state_space import filter_states
from .motion import project,velocity_native

def reference_file(row):
 """Validate a submitted anchor file, never turn source interpolation into truth."""
 if row.get('reference_partition') in ['locked','locked_final','locked_audit']:return dict(status='declared_locked_not_opened',failure_code='requires_separate_locked_audit',samples=0,scientific_reference_ready=False)
 if not row.get('trajectory_path'):return dict(status='pending',failure_code='no_independent_trajectory',samples=0,scientific_reference_ready=False)
 candidate=ROOT/row['trajectory_path'];resolved=candidate.resolve()
 if not resolved.is_relative_to(ROOT.resolve()):return dict(status='rejected',failure_code='reference_outside_project',samples=0,scientific_reference_ready=False)
 if not resolved.is_file():return dict(status='rejected',failure_code='missing_trajectory_file',samples=0,scientific_reference_ready=False)
 if not row.get('independent_reviewer') or not row.get('trajectory_sha256') or digest(resolved)!=row['trajectory_sha256']:return dict(status='rejected',failure_code='missing_reviewer_or_hash_mismatch',samples=0,scientific_reference_ready=False)
 try:
  with resolved.open(newline='') as f:
   reader=csv.DictReader(f);required={'timestamp_us','x_native_px','y_native_px','visible','reviewer_id'}
   if not required<=set(reader.fieldnames or []):raise ValueError('missing_reference_columns')
   if len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('duplicate_reference_columns')
   rows=list(reader)
  time=np.asarray([int(r['timestamp_us']) for r in rows],dtype=np.int64);position=np.array([[float(r['x_native_px']),float(r['y_native_px'])] for r in rows]);visibility=[(r['visible'] or '').strip().lower() for r in rows]
  if not set(visibility)<= {'0','1','true','false'}:raise ValueError('invalid_visibility_encoding')
  visible=np.array([r in ['1','true'] for r in visibility]);reviewer=all(r['reviewer_id']==row['independent_reviewer'] for r in rows)
  if len(time)<7 or not np.all(np.diff(time)>0):raise ValueError('insufficient_or_unordered_anchors')
  duration=(time[-1]-time[0])/1e6
  if not .5<=duration<=2.:raise ValueError('reference_duration_outside_contract')
  if (len(time)-1)/duration<10 or np.diff(time).max()>150000:raise ValueError('reference_sampling_too_sparse')
  if visible.sum()<7 or not np.isfinite(position[visible]).all() or not reviewer:raise ValueError('invalid_visible_anchor_or_reviewer')
  if row.get('t_start_us') and time[0]<int(row['t_start_us']) or row.get('t_end_us') and time[-1]>int(row['t_end_us']):raise ValueError('anchors_outside_declared_episode')
  return dict(status='development_reference_submitted',failure_code=None,samples=len(time),duration_seconds=float(duration),visible_samples=int(visible.sum()),trajectory_sha256=digest(resolved),scientific_reference_ready=False,missing_final_evidence='Second annotator/disagreement, event/contact/identity geometry, locked assignment and threshold preregistration are not established by this position file')
 except (ValueError,KeyError,TypeError) as error:return dict(status='rejected',failure_code=str(error),samples=0,scientific_reference_ready=False)

def references():
 with (path('reports')/'independent_reference_queue.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 results=[dict(episode_uid=r['episode_uid'],video_id=r['video_id'],**reference_file(r)) for r in rows];save_index(path('reports')/'independent_reference_validation.parquet',results)
 report('independent_reference_validation.json',dict(proposals=len(rows),status_counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},independent_final_references_ready=False,locked_reference_created=False,locked_scope_warning='All current participants have informed development. No final holdout is manufactured from the iteratively used material.',required_final_fields=['participant_id','entity_type','hand_side','t_start_us','t_end_us','anchor_position','visible','occluded','mask_or_boundary','contact_object_id','contact_state','contact_onset_us','contact_release_us','identity_id','ambiguity','annotator_id','second_annotator','disagreement_position_px','disagreement_contact_frames','disagreement_identity'],velocity_reference_policy='Derive velocity from reviewed anchors; audit two smoothing scales and estimator families; never ask annotators to type velocity labels',uncertainty_bookkeeping=dict(observation='source provenance/annotation uncertainty',fit='conditional parameter uncertainty',representation='empirical misspecification allowance',association=None,motion='conditional causal state posterior',combination='Components can overlap/correlate. No independent additive decomposition or full calibrated total is claimed.'),**identity()))

def kalman_video(v,monitor):
 allowed([v]);old=[r for r in pq.read_table(path('reports')/'stream_hybrid_metrics.parquet').to_pylist() if r['video_id']==v];episodes={r['episode_uid']:r for r in pq.read_table(path('reports')/f'dense_episodes.{v}.parquet').to_pylist()};queries=defaultdict(set)
 for r in old:queries[episodes[r['episode_uid']]['entity_track_id']].add(r['state_end_us'])
 folder=path('derived')/'spatial';source=load_shard(folder/f'{v}.pt');tracks=defaultdict(list)
 for r in pq.read_table(folder/f'{v}.parquet').to_pylist():
  if r['spatial_valid'] and r['entity_track_id'] in queries:tracks[r['entity_track_id']].append(r)
 lookup={};saved=[];meta=[]
 for track,rows in tracks.items():
  ids=np.array([r['row_index'] for r in rows]);time=np.array([r['timestamp_us']/1e6 for r in rows]);cutoffs=sorted(queries[track]);outputs={}
  for name,robust in [('kalman_cv',False),('kalman_cv_robust',True)]:outputs[name]=filter_states(time,source['h'][ids,:2].astype(float),source['variance'][ids,:2].astype(float),np.asarray(cutoffs)/1e6,spectral_density=.01,robust=robust)
  for j,cutoff in enumerate(cutoffs):
   for name,states in outputs.items():
    state=states[j];assert state['latest_input_time']<=cutoff/1e6;lookup[(track,cutoff,name)]=state;saved.append((state['state'],state['covariance']));meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=track,state_time_us=cutoff,method=name,valid=state['valid'],latest_input_us=int(round(state['latest_input_time']*1e6)),observation_count=state['observation_count'],covariance_calibrated=False,model_probabilities=None,representation_version='track1-v3.0.0-dev.kalman-pilot1'))
 raw=defaultdict(list)
 for r in pq.read_table(path('reports')/f'dense_flow_raw.{v}.parquet').to_pylist():raw[r['episode_uid']].append(r)
 references={}
 for uid in {r['episode_uid'] for r in old}:
  rows=raw[uid];ts=np.array([r['timestamp_us'] for r in rows]);tt=(ts-ts[0])/1e6;uniform=np.arange(0,tt[-1]+1e-6,.02);p=np.array([[r['position_native_x'],r['position_native_y']] for r in rows]);p=np.stack([np.interp(uniform,tt,p[:,j]) for j in range(2)],axis=1);references[uid]=(ts[0],uniform,{w:(savgol_filter(p,w,2,axis=0),savgol_filter(p,w,2,deriv=1,delta=.02,axis=0)) for w in [5,7,11]})
 grouped=defaultdict(list)
 for r in old:grouped[(r['episode_uid'],r['state_end_us'])].append(r)
 camera=json.loads((folder/f'{v}.camera.json').read_text());diag=np.hypot(*camera['native_size']);metrics=[];ledger=[]
 for (uid,cutoff),baseline in grouped.items():
  track=episodes[uid]['entity_track_id'];states={name:lookup[(track,cutoff,name)] for name in ['kalman_cv','kalman_cv_robust']};valid=all(r['valid'] for r in states.values());ledger.append(dict(video_id=v,episode_uid=uid,state_time_us=cutoff,all_methods_supported=valid))
  if not valid:continue
  metrics+=baseline;t0,uniform,ref=references[uid];relative=(cutoff-t0)/1e6
  for width,horizon in sorted({(r['reference_smoothing_samples'],r['horizon_seconds']) for r in baseline}):
   p,vref=ref[width];truth=np.array([np.interp(relative+horizon,uniform,p[:,j]) for j in range(2)]);velocity=np.array([np.interp(relative,uniform,vref[:,j]) for j in range(2)])
   for name,state in states.items():
    x=state['state'][:2];vv=state['state'][2:];pred=project(x+horizon*vv,camera);native=velocity_native(x,vv,camera);metrics.append(dict(video_id=v,split=split(v),episode_uid=uid,category=episodes[uid]['category'],entity_name=episodes[uid]['entity_name'],state_end_us=cutoff,method=name,horizon_seconds=horizon,reference_smoothing_samples=width,position_error_frame_diagonal=float(np.linalg.norm(pred-truth)/diag),velocity_error_native_px_s=float(np.linalg.norm(native-velocity)) if horizon==0 else None,reference_speed_native_px_s=float(np.linalg.norm(velocity)),predicted_speed_native_px_s=float(np.linalg.norm(native))))
 out=path('derived')/'kalman_cv_pilot';out.mkdir(exist_ok=True);save_shard(out/f'{v}.pt',dict(state=np.array([s for s,_ in saved],np.float32),covariance=np.array([c for _,c in saved],np.float32)));save_index(out/f'{v}.parquet',meta)
 tensors=load_shard(out/f'{v}.pt');assert all(np.isfinite(a).all() for a in tensors.values());eigen=np.linalg.eigvalsh(tensors['covariance'].astype(float));assert eigen.min()>-1e-7;monitor.check();return metrics,ledger

def run(monitor):
 references();records=[];ledger=[]
 for v in config()['train']+config()['validation']:
  rows,attempts=kalman_video(v,monitor);records+=rows;ledger+=attempts;LOG.info('Kalman causal audit %s complete',v)
 save_index(path('reports')/'kalman_metrics.parquet',records);save_index(path('reports')/'kalman_support.parquet',ledger);summary={}
 for width in [5,7,11]:
  summary[str(width)]={}
  for name in sorted({r['method'] for r in records}):
   summary[str(width)][name]={}
   for horizon in [0.,.1,.25,.5]:
    byvideo={}
    for v in config()['validation']:
     rr=[r for r in records if (r['video_id'],r['method'],r['reference_smoothing_samples'],r['horizon_seconds'])==(v,name,width,horizon)]
     if rr:byvideo[v]=dict(n=len(rr),position_nrmse=float(np.sqrt(np.mean([r['position_error_frame_diagonal']**2 for r in rr]))),velocity_rmse_native_px_s=float(np.sqrt(np.mean([r['velocity_error_native_px_s']**2 for r in rr]))) if horizon==0 else None)
    if byvideo:summary[str(width)][name][str(horizon)]=dict(position_nrmse=float(np.mean([r['position_nrmse'] for r in byvideo.values()])),velocity_rmse_native_px_s=float(np.mean([r['velocity_rmse_native_px_s'] for r in byvideo.values()])) if horizon==0 else None,by_video=byvideo)
 report('kalman_comparison.json',dict(validation_metrics=summary,attempted_reference_cutoffs=len(ledger),matched_cutoffs=sum(r['all_methods_supported'] for r in ledger),spectral_density=.01,parameter_policy='Fixed normalized-coordinate engineering ablation; not selected using validation or a human velocity target',posterior_calibrated=False,promoted=False,limitations=['Existing RGB foreground-flow proxy; no new independent accepted fast-motion validation references.','Joseph covariance update and PSD checks do not establish empirical calibration.','IMM/CA model probabilities and association uncertainty remain unavailable.','CV continuation remains a diagnostic; Track 2 is not trained.'],**identity()))
 snapshot()

def snapshot():
 from .scientific import environment
 from .scientific import CandidateStore
 reader_checks=[]
 for v in config()['train']+config()['validation']:
  rows=pq.read_table(path('derived')/'compound32_topology'/f'{v}.parquet').to_pylist()
  with CandidateStore() as reader:
   for row in rows:
    state=reader.get_state(v,row['row_index']);allowed_geometry=row['certified_topology_accepted'];failure=None
    try:
     curves=state.normalized_geometry_curves(row['window_end_us']-250000)
     if not allowed_geometry:raise AssertionError('Uncertified/epoch-crossing state accepted by geometry reader')
     assert all(np.isfinite(curve).all() for curve in curves)
    except ValueError as error:
     if allowed_geometry:raise
     failure=str(error)
    reader_checks.append(dict(video_id=v,row_index=row['row_index'],certified_geometry_available=allowed_geometry,blocking_reason=failure))
 save_index(path('reports')/'certificate_reader_checks.parquet',reader_checks)
 evidence=[path('reports')/(name+'.json') for name in ['topology_certificate','neutral_detail','neutral_detail_store','neutral_detail_selection','kalman_comparison','independent_reference_validation']]
 evidence += [p for p in [path('reports')/'kinematics_kalman.json',path('reports')/'certificate_reader_checks.parquet'] if p.exists()]
 outputs=[]
 for name in ['endpoint_neutral_detail','kalman_cv_pilot','compound32_topology']:
  for extension in ['*.pt','*.parquet']:outputs+=sorted((path('derived')/name).glob(extension))
 report('remediation_manifest.json',dict(state='open_not_finalized',scientific_acceptance=False,ingestion_ready=False,track2_started=False,source_report_sha256=digest('/Users/anagha/Downloads/Finalising Track 1_ research-driven remediation plan for uncertainty-aware spatio-temporal functiona.pdf'),environment=environment(),dataset_manifest_sha256=digest(ROOT/'reports/freeze_manifest.json'),certificate_reader_checks=len(reader_checks),certified_epoch_safe_reader_outputs=sum(r['certified_geometry_available'] for r in reader_checks),evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},output_sha256={str(p.relative_to(ROOT)):digest(p) for p in outputs},clean_full_rebuild_performed=False,deferred=['Actual independent annotations and fresh locked participant audit','Contact-aware shape fitting and independently reviewed contact boundaries','Verified ring lineage/identity association','IMM/CA and calibrated velocity/contact/association uncertainty','Full-timeline compound rebuild and clean scientific reproduction'],**identity()))
