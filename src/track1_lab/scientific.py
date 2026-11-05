"""Evidence gates, candidate contract and independent-reference handoff."""
from collections import OrderedDict
from dataclasses import dataclass
import csv,json,platform,sys,subprocess,time
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,digest
from track1.artifacts import load_shard,save_shard,save_index,versions
from track1.numerics import basis
from .common import config,path,allowed,report,identity,LOG
from .pilot32 import selected_cases
from .residual import fit_residual,evaluate
from .causal_kinematics import KinematicState,estimate

QUALITY_KEYS=['spatial_fit_valid','simple_loop_valid','ring_topology_valid','contact_geometry_valid','small_object_warning','identity_continuous','temporal_support_valid','derivative_reliable','residual_active','covariance_psd','covariance_calibrated','camera_pose_valid','appearance_valid']

def training_eligible(flags,required):
 """Unknown evidence is never promoted to True by Python truthiness."""
 return all(flags.get(key) is True for key in required)

@dataclass(frozen=True)
class FunctionalEntityState:
 timestamp_us:int
 entity_track_id:str
 coarse_coefficients:np.ndarray
 functional_endpoint_state:np.ndarray
 functional_endpoint_velocity:np.ndarray
 geometry_descriptor:dict
 appearance_embedding:object
 covariance:object
 quality_flags:dict
 residual_descriptor:object
 representation_version:str
 kinematics:KinematicState|None=None

 def normalized_geometry_curves(self,timestamp_us,samples=512,require_certificate=True):
  """Use the declared certified decoder; shape coordinates, not metric 3D."""
  from .topology_certificate import evaluate_curves,input_hashes
  geometry=self.geometry_descriptor;certificate=geometry.get('topology_certificate') or {}
  if require_certificate and (not certificate.get('certified') or certificate.get('decoder_version')!='canonical_tensor_bezier1'):raise ValueError('No certificate for the declared geometry decoder')
  if require_certificate and geometry.get('topology_epoch_valid') is not True:raise ValueError('A stable source topology epoch is not established')
  if require_certificate and geometry.get('ring_correspondence_epoch_valid') is not True:raise ValueError('A stable retained ring-correspondence epoch is not established')
  if require_certificate:
   expected=input_hashes(self.coarse_coefficients,geometry['allocation'],geometry['roles'],geometry['parents'],geometry['window_seconds'])
   if any(certificate.get(k)!=value for k,value in expected.items()):raise ValueError('Geometry or decoder changed after certification')
  phase=1+(timestamp_us-self.timestamp_us)/(1e6*geometry['window_seconds'])
  return evaluate_curves(self.coarse_coefficients,geometry['allocation'],geometry['roles'],phase,samples)

class CandidateStore:
 """Explicit topology candidate reader; at most two mmap video shards."""
 def __init__(self):self.cache=OrderedDict()
 def close(self):self.cache.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def get_state(self,video,row_index):
  allowed([video])
  if video not in self.cache:
   if len(self.cache)>=2:self.cache.popitem(last=False)
   folder=path('derived')/'compound32_topology';cases={(c['entity_track_id'],c['window_end_us']):{k:a.copy() if isinstance(a,np.ndarray) else a for k,a in c.items()} for c in selected_cases(video)};self.cache[video]=(load_shard(folder/f'{video}.pt'),pq.read_table(folder/f'{video}.parquet').to_pylist(),cases)
  self.cache.move_to_end(video);data,meta,cases=self.cache[video];r=meta[row_index];c=data['coefficients'][row_index].astype(float);c.setflags(write=False);window=r['window_seconds'];flags=r['quality_flags'];assert set(QUALITY_KEYS)<=set(flags)
  case=cases[(r['entity_track_id'],r['window_end_us'])]
  # The pilot has hidden-target protocols. Never join an unrestricted full-stream
  # kinematic state here: that would reintroduce the withheld target observations.
  from refinement.data import holdout
  keep=(~holdout(case['time'],r['protocol'],.875))&(case['time']>=.5);origin=r['window_end_us']/1e6-1.;kinematic=estimate(case['time'][keep]+origin,case['h'][keep,:2],case['variance'][keep,:2],r['window_end_us']/1e6)
  return FunctionalEntityState(r['window_end_us'],r['entity_track_id'],c,basis(1.)@c,2/window*basis(1.,1)@c,dict(allocation=r['allocation'],roles=r['roles'],parents=r['ring_parent'],orientation=r['ring_orientation'],hole_traversal=r['hole_traversal'],window_seconds=window,topology_epoch_valid=r.get('topology_epoch_valid'),ring_correspondence_epoch_valid=r.get('ring_correspondence_epoch_valid'),topology_certificate=r.get('topology_certificate')),None,None,flags,None,'track1-v3.0.0-dev.hybrid1',kinematic)

def build_residual_states(monitor):
 from .common import split
 base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());parameters=json.loads((path('reports')/'residual_selection.json').read_text())['parameters'];folder=path('derived')/'causal_residual';folder.mkdir(exist_ok=True)
 for v in config()['train']+config()['validation']:
  means=[];residuals=[];projections=[];endpoints=[];velocities=[];meta=[]
  for c in selected_cases(v):
   m=fit_residual(c['time'],c['h'],c['variance'],c['prior'],c['quality'],base['temporal']['options'],np.asarray(base['feature_scale']),**parameters);means.append(m['coarse']);residuals.append(m['residual']);projections.append(m['projection']);endpoints.append(evaluate(m,np.array([1.]))[0]);velocities.append(evaluate(m,np.array([1.]),1)[0]);meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=c['entity_track_id'],window_end_us=c['window_end_us'],latest_input_us=int(c['window_end_us']-1000000+round(float(c['time'].max())*1e6)),split=split(v),representation_version='track1-v3.0.0-dev.residual1',residual_channels=[0,1],residual_count=parameters['count'],residual_gate=m['gate'].tolist(),orthogonality_error=m['orthogonality_error'],scientifically_accepted=False))
  for r in meta:assert r['latest_input_us']<=r['window_end_us']
  save_shard(folder/f'{v}.pt',dict(coarse_coefficients=np.asarray(means,np.float32),residual_coefficients=np.asarray(residuals,np.float32),residual_projection=np.asarray(projections,np.float32),endpoint_state=np.asarray(endpoints,np.float32),endpoint_velocity=np.asarray(velocities,np.float32)));save_index(folder/f'{v}.parquet',meta);monitor.check();LOG.info('Stored causal residual states %s %d',v,len(meta))

def reference_queue():
 old=pq.read_table(path('reports')/'dense_episodes.parquet').to_pylist();queue=[]
 for r in old:queue.append(dict(episode_uid=r['episode_uid'],video_id=r['video_id'],entity_track_id=r['entity_track_id'],proposal_source='existing_RGB_proxy_screen',proposed_event_unverified=r.get('category'),independent_reviewer='',trajectory_path='',trajectory_sha256='',identity_review='',contact_review=''))
 existing={r['episode_uid'] for r in queue}
 for v in config()['train']+config()['validation']:
  for c in selected_cases(v)[:10]:
   uid='manual_'+v+'_'+str(c['window_end_us'])+'_'+digest_text(c['entity_track_id'])[:10]
   if uid not in existing:queue.append(dict(episode_uid=uid,video_id=v,entity_track_id=c['entity_track_id'],proposal_source='source_window_proposal; event category unverified',proposed_event_unverified=None,independent_reviewer='',trajectory_path='',trajectory_sha256='',identity_review='',contact_review=''));existing.add(uid)
 target=path('reports')/'independent_reference_queue.csv'
 if target.exists():
  with target.open(newline='') as f:previous_rows=list(csv.DictReader(f))
  previous={r['episode_uid']:r for r in previous_rows}
  if len(previous)!=len(previous_rows):raise ValueError('Duplicate episode IDs in independent review queue; refusing to overwrite reviewer work')
  generated={r['episode_uid'] for r in queue}
  for r in queue:
   for key,value in previous.get(r['episode_uid'],{}).items():
    if key not in ['episode_uid','video_id','entity_track_id','proposal_source'] and value:r[key]=value
  queue += [r for uid,r in previous.items() if uid not in generated]
 fields=list(dict.fromkeys(key for r in queue for key in r))
 with target.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(queue)
 submitted=sum(bool(r.get('trajectory_path') and r.get('independent_reviewer')) for r in queue)
 report('independent_reference_contract.json',dict(proposals=len(queue),submitted_reference_paths=submitted,completed_independent_reviews=None if submitted else 0,required_trajectory_columns=['timestamp_us','x_native_px','y_native_px','visible','reviewer_id'],minimum_sample_hz=10,recommended_sample_hz=25,minimum_duration_seconds=.5,maximum_duration_seconds=2.,event_categories=['stationary','smooth_translation','pickup','put_down','reversal','out_and_back','rapid_reach','contact_onset','contact_release','occlusion_reappearance','small_manipulated_object'],event_balance_verified=False,required_estimators=['local polynomial differentiation','regularized spline differentiation'],independent_labels_present=None if submitted else False,warning='Proposals are not labels. Submitted paths require validation; existing algorithmic acceptance is not independent review. No scientific gate is satisfied by this queue.',queue_sha256=digest(target)))

def digest_text(value):
 import hashlib
 return hashlib.sha256(value.encode()).hexdigest()

def derivative_references(timestamp_us,position,noise_px):
 """Two reference families for future human trajectories, in native px/s."""
 from scipy.interpolate import UnivariateSpline
 time=(np.asarray(timestamp_us,dtype=np.int64)-int(timestamp_us[0]))/1e6;position=np.asarray(position,float)
 if len(time)<7 or not np.all(np.diff(time)>0) or position.shape!=(len(time),2) or not np.isfinite(position).all():raise ValueError('Reference requires >=7 finite, strictly ordered 2D observations')
 local=[]
 for t in time:
  ids=np.sort(np.argsort(abs(time-t))[:7]);design=np.vander(time[ids]-t,3,increasing=True);coefficients=np.linalg.lstsq(design,position[ids],rcond=None)[0];local.append(coefficients[1])
 spline=np.stack([UnivariateSpline(time,position[:,j],k=3,s=len(time)*float(noise_px)**2).derivative()(time) for j in range(2)],axis=1)
 return dict(local_polynomial=np.asarray(local),regularized_spline=spline,units='native_pixels_per_second',sensitivity_rmse=float(np.sqrt(np.mean((np.asarray(local)-spline)**2))))

def environment():
 hardware={}
 for key in ['hw.model','machdep.cpu.brand_string','hw.memsize','hw.logicalcpu']:
  try:hardware[key]=subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True,stderr=subprocess.DEVNULL).strip()
  except (OSError,subprocess.CalledProcessError):hardware[key]=None
 return dict(python=sys.version,platform=platform.platform(),hardware=hardware,packages=versions(),git_commit=None,git_reason='Workspace is not a Git checkout',seed=config()['seed'])

def resources():
 since=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime((ROOT/'work/track1_lab/report_review/report.txt').stat().st_mtime));rows=[json.loads(line) for line in (path('reports')/'resources.jsonl').read_text().splitlines()];rows=[r for r in rows if r['timestamp']>=since]
 return dict(scope='Monitored process trees since report extraction; sampled, not a peak-allocation profiler',since=since,samples=len(rows),peak_tree_rss_gib=max(r['tree_rss_gib'] for r in rows),minimum_available_gib=min(r['available_gib'] for r in rows),target_gib=16,warning_gib=18,hard_limit_gib=20,oom_observed=False)

def audit_candidate(monitor):
 counts=default_counts();psd=0;orth=0.;rebuild=0.
 for v in config()['train']+config()['validation']:
  folder=path('derived')/'causal_residual';data=load_shard(folder/f'{v}.pt');meta=pq.read_table(folder/f'{v}.parquet').to_pylist()
  for i,r in enumerate(meta):
   model=dict(coarse=data['coarse_coefficients'][i].astype(float),residual=data['residual_coefficients'][i].astype(float),projection=data['residual_projection'][i].astype(float),cutoff=1.,count=r['residual_count']);rebuild=max(rebuild,float(np.max(abs(evaluate(model,np.array([1.]))[0]-data['endpoint_state'][i]))),float(np.max(abs(evaluate(model,np.array([1.]),1)[0]-data['endpoint_velocity'][i]))));orth=max(orth,r['orthogonality_error']);assert r['latest_input_us']<=r['window_end_us']
  counts['residual_states']+=len(meta)
  joint=load_shard(path('derived')/'joint_kinematics'/f'{v}.pt')
  for name,a in joint.items():
   assert np.isfinite(a).all();eig=np.linalg.eigvalsh(a.astype(float));psd+=int(np.sum(eig[:,0]< -1e-6*np.maximum(1.,np.max(abs(eig),axis=1))))
  with CandidateStore() as store:
   rows=pq.read_table(path('derived')/'compound32_topology'/f'{v}.parquet').to_pylist()
   for i in range(len(rows)):
    s=store.get_state(v,i);assert s.coarse_coefficients.shape==(4,67);assert np.isfinite(s.functional_endpoint_state).all();assert s.covariance is None
    if s.quality_flags['ring_topology_valid']:counts['accepted_topology_states']+=1
    if s.quality_flags.get('topology_certified'):counts['curve_certified_states']+=1
    if rows[i].get('certified_topology_accepted'):counts['certified_epoch_safe_states']+=1
    assert not training_eligible(s.quality_flags,['ring_topology_valid','identity_continuous','covariance_calibrated'])
   counts['topology_states']+=len(rows)
  monitor.check()
 assert psd==0 and orth<1e-8 and rebuild<1e-4,(psd,orth,rebuild)
 return dict(passed=True,counts=counts,significant_psd_failures=psd,max_orthogonality_error=orth,max_float32_endpoint_rebuild_error=rebuild,unknown_evidence_rejected_for_training=True,count_scope='accepted_topology_states is the historical sampled-valid count; curve_certified_states and certified_epoch_safe_states are separate development checks, not scientific acceptance')

def default_counts():return dict(residual_states=0,topology_states=0,accepted_topology_states=0,curve_certified_states=0,certified_epoch_safe_states=0)

def stratify_geometry():
 """Train-derived size bins; preserve unavailable labels as unknown."""
 from .review_fixes import clustered_summary
 rows=pq.read_table(path('reports')/'joint_topology.parquet').to_pylist();training=[r['target_area_fraction'] for r in rows if r['split']=='train' and r['method']=='joint_guard'];cuts=np.quantile(training,[.25,.5,.75]);names={}
 for v in config()['train']+config()['validation']:
  names[v]={r['entity_track_id']:r['entity_name'] for r in pq.read_table(path('derived')/'spatial'/f'{v}.parquet',columns=['entity_track_id','entity_name']).to_pylist()};folder=path('derived')/'compound32_topology';meta=pq.read_table(folder/f'{v}.parquet').to_pylist();source_hash=digest(path('derived')/'compound32_pilot'/f'{v}.pt')
  for r in meta:r.update(state_time_us=r['window_end_us'],participant_id=v.split('_')[0],control_offsets=np.r_[0,np.cumsum(r['allocation'])].tolist(),source_hash=source_hash,ring_confidence=[None]*len(r['allocation']))
  save_index(folder/f'{v}.parquet',meta)
 for r in rows:
  name=names[r['video_id']][r['entity_track_id']];r.update(entity_type='hand' if 'hand' in name.lower() else 'object',area_quantile='Q'+str(1+int(np.searchsorted(cuts,r['target_area_fraction']))),ring_mode='single' if r['ring_count']==1 else 'multi',contact_label=None,occlusion_label=None,boundary_complexity_label=None,motion_reference_label=None,camera_pose_status='unavailable_intrinsic_2d')
 save_index(path('reports')/'joint_topology_stratified.parquet',rows);summary={}
 for key in ['entity_type','area_quantile','source_kind','ring_mode','camera_pose_status']:
  summary[key]={}
  for value in sorted({r[key] for r in rows}):
   selected=[r for r in rows if r['split']=='validation' and r['method']=='joint_guard' and r[key]==value]
   if selected:summary[key][value]=dict(probes=len(selected),iou=clustered_summary(selected,'iou'),boundary_fscore=clustered_summary(selected,'boundary_fscore_2px'))
 report('evaluation_strata.json',dict(size_quartile_cutpoints_training_only=cuts.tolist(),validation=summary,missing_reference_strata=['contact/non-contact','occlusion','independent stationary/moderate/fast motion','independently scored boundary complexity'],unknown_labels_remain_null=True,**identity()))

def run(monitor):
 build_residual_states(monitor);reference_queue();stratify_geometry();verification=audit_candidate(monitor);fixes=json.loads((path('reports')/'research_report_fixes.json').read_text());gates={
 'spatial':dict(status='unverified',required='Fresh locked manual masks: IoU>=.95, BF2>=.85, scaleBF>=.95; accepted topology failures=0'),
 'contact':dict(status='unverified',required='Independent contact boundary median<=2px, P95<=5px and sign error<2%'),
 'temporal':dict(status='unverified',required='Preregister regime-specific thresholds after annotation reliability: stationary false motion; steady translation versus robust CV; abrupt onset/reversal accuracy; joint event recall/FAR trade-off. Polynomial derivative is diagnostic, not authoritative velocity.'),
 'forecast_preservation':dict(status='unverified',required='Combined history/causal state must retain a usable CV baseline. Raw polynomial extrapolation is a diagnostic, not a universal encoder acceptance gate; no Track 2 model trained.'),
 'identity':dict(status='missing_reference',required='Independent IDF1>=.95; zero consequential contact-episode swaps'),
 'uncertainty':dict(status='unverified',required='Independent 95% coverage92-98%, joint consistency, fast/contact subsets; 2D source-position calibration is insufficient'),
 'robustness':dict(status='unverified',required='Missing-observation/occlusion/perturbation gates on locked independent references'),
 'eligibility':dict(status='unverified',required='>=99% usable among independently coverage-eligible windows, not conditional pilot support'),
 'reproducibility':dict(status='partial',required='Clean rebuild hash match; numerical checks passed; full clean rebuild not performed'),
 'resources':dict(status='measured_development_only',required='NoOOM, peak treeRSS<20GiB; steady target16/warning18; profiling is not a scientific gate'),
 'leakage':dict(status='partial',required='Train-only new selection; iteratively used validation is not fresh final evaluation')}
 if (path('reports')/'topology_certificate.json').exists():gates['continuous_topology']=dict(status='partial',required='Use the bound canonical decoder, certificate/input hashes and stable retained epochs. Unresolved states fail closed; source completeness, raster topology and independent lineage remain unverified.')
 if (path('reports')/'independent_reference_validation.json').exists():gates['independent_references']=dict(status='missing_reference',required='Reviewed anchors/contact boundaries/identity with second-annotator reliability and preregistered locked assignment; pending proposals are not ground truth.')
 report('scientific_readiness.json',dict(state='open_not_finalized',scientific_acceptance_gates_passed=False,ingestion_ready=False,finalization_authorized=False,track2_authorized=False,gates=gates,numerical_verification=verification,source_report_sha256=digest('/Users/anagha/Downloads/Track 1 Refinement and Scientific Freeze Plan.pdf'),environment=environment(),resources=resources(),dataset_manifest_sha256=digest(ROOT/'reports/freeze_manifest.json'),independent_reference_annotation_sha256=None,source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/track1_lab').glob('*.py'))},split_sha256=digest(ROOT/'configs/track1_lab.yaml'),evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in [path('reports')/'research_report_fixes.json',path('reports')/'joint_uncertainty.json',path('reports')/'residual_selection.json',path('reports')/'evaluation_strata.json',path('reports')/'independent_reference_queue.csv']},**identity()));LOG.info('Scientific readiness audited; missing references remain unverified')
