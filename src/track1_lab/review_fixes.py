"""Apply actionable research-review fixes without releasing a new baseline."""
from collections import defaultdict
import json,platform,sys,time
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,digest
from track1.artifacts import load_shard,save_shard,save_index,versions
from track1.numerics import basis
from refinement.data import holdout
from refinement.protocol import rank
from .common import config,path,allowed,split,pool,report,identity,audit,LOG
from .compound import render
from .pilot32 import selected_cases,dest
from .ring_topology import guard,check,render as render_hierarchy
from .residual import fit_residual,evaluate
from .joint_covariance import ensemble,jacobian,score

def topology_video(video):
 import cv2
 from pipeline import Dataset
 from track1.evaluation import geometry_metrics
 cv2.setNumThreads(1);allowed([video]);source=path('derived')/'compound32_pilot';data=load_shard(source/f'{video}.pt');meta=pq.read_table(source/f'{video}.parquet').to_pylist();pilot=pq.read_table(dest()/f'{video}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r for r in pilot};cases={(c['entity_track_id'],c['window_end_us']):c for c in selected_cases(video)};camera=json.loads((path('derived')/'spatial'/f'{video}.camera.json').read_text());out=path('derived')/'compound32_topology';out.mkdir(exist_ok=True);states=[];coefficients=[];metrics=[]
 with Dataset() as ds:
  for row in meta:
   c=data['guarded_coefficients'][row['row_index']].astype(float);allocation=row['allocation'];roles=row['roles'];original=check(c,allocation,roles);fixed=guard(c,allocation,roles);quality=fixed['quality'];i=len(states);coefficients.append(fixed['coefficients'].astype(np.float32));case=cases[(row['entity_track_id'],row['window_end_us'])];t=case['time'];hidden=holdout(t,row['protocol'],.875);j=np.flatnonzero(hidden)[-1];target_row=lookup[(row['entity_track_id'],int(row['window_end_us']-1000000+round(t[j]*1e6)))];target=ds.get_mask(target_row['mask_uid']);z=2*(t[j]-.5)/.5-1
   scores={}
   for name,cc in [('loop_guard',c),('joint_guard',fixed['coefficients'])]:
    prediction=basis(z)@cc
    if name=='joint_guard':mask=render_hierarchy(prediction,allocation,roles,quality['ring_parent'],camera,target.shape)
    else:mask,_,_=render(prediction,allocation,roles,camera,target.shape)
    scores[name]=geometry_metrics(target,mask)
   states.append(dict(**{k:v for k,v in row.items() if k not in ['row_index','guarded_geometry_valid','guard_factors','guard_spatial_samples']},row_index=i,representation_version='track1-v3.0.0-dev.topology1',ring_parent=quality['ring_parent'],ring_orientation=[1 if role=='outer' else -1 for role in roles],hole_traversal='reverse_decoded_curve; historical coefficient parameter orientation retained',quality_flags=dict(spatial_fit_valid=None,simple_loop_valid=quality['simple_loop_valid'],ring_topology_valid=quality['ring_topology_valid'],contact_geometry_valid=None,small_object_warning=bool(np.count_nonzero(mask)<256),identity_continuous=None,temporal_support_valid=True,derivative_reliable=None,residual_active=False,covariance_psd=None,covariance_calibrated=False,camera_pose_valid=False,appearance_valid=None),accepted=quality['ring_topology_valid'],joint_guard_alpha=fixed['alpha'],joint_guard_shape_mse=fixed['shape_mse'],covariance_available=False,guard_spatial_samples=2048 if quality['ring_topology_valid'] else quality['spatial_samples']))
   for method,gm in scores.items():metrics.append(dict(video_id=video,split=split(video),entity_track_id=row['entity_track_id'],window_end_us=row['window_end_us'],protocol=row['protocol'],method=method,accepted=quality['ring_topology_valid'],joint_topology_before=original['ring_topology_valid'],joint_topology_after=quality['ring_topology_valid'],source_kind=target_row['source_kind'],target_area_fraction=float(np.mean(target>0)),entity_type='hand' if 'hand' in case.get('entity_name','').lower() else 'unknown',ring_count=len(allocation),guard_alpha=fixed['alpha'],shape_mse=fixed['shape_mse'],**gm))
 out=path('derived')/'compound32_topology';save_shard(out/f'{video}.pt',dict(coefficients=np.stack(coefficients)));save_index(out/f'{video}.parquet',states);save_index(path('reports')/f'joint_topology.{video}.parquet',metrics);return dict(video_id=video,states=len(states),accepted=sum(r['accepted'] for r in states))

def temporal_video(video):
 allowed([video]);base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());scale=np.asarray(base['feature_scale']);options=base['temporal']['options'];records=[]
 candidates=[dict(count=count,strength=strength,threshold=threshold) for count in [2,3] for strength,threshold in [(1.,4.),(3.,9.),(10.,16.)]]
 if split(video)=='validation':candidates=[json.loads((path('reports')/'residual_selection.json').read_text())['parameters']]
 for case in selected_cases(video):
  t=case['time'];h=case['h'];variance=case['variance'];centre=.5
  for protocol in ['single','gap100','endpoint100']:
   hidden=holdout(t,protocol,centre);keep=~hidden
   if keep.sum()<8 or not hidden.any():continue
   for parameters in candidates:
    model=fit_residual(t[keep],h[keep],variance[keep],case['prior'][keep],case['quality'][keep],options,scale,**parameters);query=t[hidden];coarse=basis(2*query-1)@model['coarse'];prediction=evaluate(model,query);normalizer=scale[:2];record=dict(video_id=video,split=split(video),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,parameter_id=json.dumps(parameters,sort_keys=True),coarse_position_mse=float(np.mean(((coarse[:,:2]-h[hidden,:2])/normalizer)**2)),residual_position_mse=float(np.mean(((prediction[:,:2]-h[hidden,:2])/normalizer)**2)),gate_mean=float(model['gate'].mean()),orthogonality_error=model['orthogonality_error']);records.append(record)
 save_index(path('reports')/f'causal_residual.{video}.parquet',records);return dict(video_id=video,cases=len(records))

def select_residual():
 rows=[]
 for v in config()['train']:rows+=pq.read_table(path('reports')/f'causal_residual.{v}.parquet').to_pylist()
 options=[]
 for key in sorted({r['parameter_id'] for r in rows}):
  rs=[r for r in rows if r['parameter_id']==key];metrics={}
  for protocol in ['single','gap100','endpoint100']:
   rr=[r for r in rs if r['protocol']==protocol];metrics[protocol]={name:float(np.mean([np.sqrt(np.mean([r[name+'_position_mse'] for r in rr if r['video_id']==v])) for v in config()['train']])) for name in ['coarse','residual']}
  eligible=all(metrics[p]['residual']<=1.05*metrics[p]['coarse'] for p in ['single','gap100']);objective=sum(metrics[p]['residual'] for p in metrics);options.append(dict(parameters=json.loads(key),eligible=eligible,objective=objective,metrics=metrics))
 eligible=[r for r in options if r['eligible']];best=min(eligible or options,key=lambda r:r['objective']);report('residual_selection.json',dict(**best,candidates=options,selected_on='five training participants only; validation evaluated after selection',scientifically_accepted=False,**identity()));return best

def covariance_video(video):
 allowed([video]);base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());scale=np.asarray(base['feature_scale']);options=base['temporal']['options'];parameters=json.loads((path('reports')/'residual_selection.json').read_text())['parameters'];records=[];covariances=[];augmented_covariances=[];meta=[]
 for case in selected_cases(video)[:24]:
  t=case['time'];hidden=holdout(t,'endpoint100',.5);keep=~hidden
  if keep.sum()<8 or not hidden.any():continue
  args=(t[keep],case['h'][keep],case['variance'][keep],case['prior'][keep],case['quality'][keep],options,scale);model=fit_residual(*args,**parameters);cov=ensemble(*args,parameters,seed=int(rank(video+case['entity_track_id']+str(case['window_end_us']))[:8],16),members=24);j=np.flatnonzero(hidden)[-1];time=t[j];position=jacobian(time,parameters['count']);velocity=jacobian(time,parameters['count'],1);joint=np.vstack([position,velocity]);sigma=joint@cov@joint.T;error=evaluate(model,np.array([time]))[0,:2]-case['h'][j,:2];obs=np.diag(np.maximum(case['variance'][j,:2],1e-12));position_cov=sigma[:2,:2]+obs;sc=score(error,position_cov);covariances.append(sigma.astype(np.float32));augmented_covariances.append(cov.astype(np.float32));meta.append(dict(video_id=video,row_index=len(meta),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],reference_time_seconds=float(time),reference_kind='source_annotation_position_only',velocity_calibrated=False,augmented_coarse_residual_cross_terms=True));records.append(dict(video_id=video,split=split(video),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],error=error.tolist(),position_covariance=position_cov.tolist(),**sc))
  residual=case['h'][keep,:2]-evaluate(model,t[keep])[:,:2];records[-1]['observable_difficulty']=max(float(np.mean(residual**2)+np.mean(case['variance'][keep,:2])),1e-12)
 out=path('derived')/'joint_kinematics';out.mkdir(exist_ok=True);save_shard(out/f'{video}.pt',dict(endpoint_joint_covariance_uncalibrated=np.stack(covariances),augmented_coefficient_covariance_uncalibrated=np.stack(augmented_covariances)));save_index(out/f'{video}.parquet',meta);save_index(path('reports')/f'joint_uncertainty.{video}.parquet',records);return dict(video_id=video,windows=len(records))

def clustered_summary(rows,key):
 videos=sorted({r['video_id'] for r in rows});values=np.array([np.mean([r[key] for r in rows if r['video_id']==v]) for v in videos]);rng=np.random.default_rng(config()['seed']);draws=rng.choice(values,(2000,len(values))).mean(1)
 return dict(mean=float(values.mean()),participant_bootstrap95=[float(v) for v in np.quantile(draws,[.025,.975])],participants=len(videos),limited_independence_warning=len(videos)<5)

def calibration():
 rows=[]
 for v in config()['train']+config()['validation']:rows+=pq.read_table(path('reports')/f'joint_uncertainty.{v}.parquet').to_pylist()
 train=[r for r in rows if r['split']=='train'];multiplier=max(float(np.mean([r['nees'] for r in train]))/2,1e-6);result={};evidence=[]
 discrepancy=np.mean([(np.outer(r['error'],r['error'])-np.asarray(r['position_covariance']))/r['observable_difficulty'] for r in train],axis=0);eigen,vectors=np.linalg.eigh(discrepancy);discrepancy=(vectors*np.maximum(eigen,0))@vectors.T
 for part in ['train','validation']:
  result[part]={}
  for variant,factor in [('raw',1.),('train_calibrated_position',multiplier),('train_heteroscedastic_block',1.)]:
   scored=[dict(video_id=r['video_id'],**score(r['error'],np.asarray(r['position_covariance'])*factor+(r['observable_difficulty']*discrepancy if variant=='train_heteroscedastic_block' else 0))) for r in rows if r['split']==part];keys=['nees','nll','width95','joint_coverage_0.5','joint_coverage_0.8','joint_coverage_0.9','joint_coverage_0.95','marginal_coverage_0.95'];summary={k:clustered_summary(scored,k) for k in keys};ordered=sorted(scored,key=lambda r:r['variance']);summary['risk_coverage']=[dict(retained_fraction=f,rmse=float(np.sqrt(np.mean([r['squared_error'] for r in ordered[:max(1,round(len(ordered)*f))]])))) for f in [.25,.5,.75,1.]];result[part][variant]=summary;evidence.extend(dict(split=part,variant=variant,**r) for r in scored)
 save_index(path('reports')/'joint_uncertainty_scored.parquet',evidence);report('joint_uncertainty.json',dict(position_multiplier=multiplier,representation_discrepancy_matrix=discrepancy.tolist(),discrepancy_policy='PSD projection of training held-out position error outer products minus conditional covariance, scaled by retained-input residual/noise energy. This empirical representation-error allowance is not independently identified physical noise.',metrics=result,ensemble_members=24,blocks=['joint x,y,vx,vy','joint coarse/residual position coefficients'],cross_terms_retained=True,calibration_scope='2D source-annotation position only; velocity, physical motion, contact and full compound state are not calibrated.',components='Observation perturbations and Bayesian observation weights form one conditional ensemble. They do not identify separate observation/representation/epistemic terms; these must not be falsely reported as independent additive covariances.',nees_consistency_certified=False,confidence_intervals='Resampled participant/video means, never overlapping windows. Only two validation participants: interval is exploratory.',**identity()))

def run(monitor):
 audit();cfg=config();videos=allowed(cfg['train']+cfg['validation']);start=time.monotonic();pool(topology_video,videos,monitor,workers=7);pool(temporal_video,cfg['train'],monitor,workers=5);select_residual();pool(temporal_video,cfg['validation'],monitor,workers=2);pool(covariance_video,videos,monitor,workers=7);calibration();summarize(monitor);LOG.info('Research-report fixes complete %.1fs; no freeze or Track 2 action',time.monotonic()-start)

def summarize(monitor):
 topology=[];temporal=[]
 for v in config()['train']+config()['validation']:
  topology+=pq.read_table(path('reports')/f'joint_topology.{v}.parquet').to_pylist();temporal+=pq.read_table(path('reports')/f'causal_residual.{v}.parquet').to_pylist()
 save_index(path('reports')/'joint_topology.parquet',topology);save_index(path('reports')/'causal_residual.parquet',temporal);summary={}
 for part in ['train','validation']:
  summary[part]={}
  for method in ['loop_guard','joint_guard']:
   rr=[r for r in topology if r['split']==part and r['method']==method];summary[part][method]=dict(fits=len(rr),accepted=sum(r['accepted'] for r in rr),joint_topology_valid_fraction=float(np.mean([r['joint_topology_after' if method=='joint_guard' else 'joint_topology_before'] for r in rr])),iou_all=clustered_summary(rr,'iou'),boundary_fscore_all=clustered_summary(rr,'boundary_fscore_2px'),mean_shape_discrepancy=float(np.mean([r['shape_mse'] for r in rr])))
 selection=json.loads((path('reports')/'residual_selection.json').read_text());selected=json.dumps(selection['parameters'],sort_keys=True);rr=[r for r in temporal if r['split']=='validation' and r['parameter_id']==selected];temporal_summary={p:{key:float(np.sqrt(np.mean([r[key+'_position_mse'] for r in rr if r['protocol']==p]))) for key in ['coarse','residual']} for p in ['single','gap100','endpoint100']}
 document=ROOT/'work/track1_lab/report_review/report.txt';report('research_report_fixes.json',dict(state='open_not_finalized',ingestion_ready=False,scientific_acceptance=False,track2_started=False,source_report='/Users/anagha/Downloads/Track 1 Refinement and Scientific Freeze Plan.pdf',source_report_text_sha256=digest(document),geometry=summary,residual_validation=temporal_summary,representation='track1-v3.0.0-dev; parallel working candidates only',remaining=['Independent dense human trajectories and contact/identity labels','Fresh locked participant evaluation','Calibrated velocity and compound-geometry joint uncertainty','Continuous-time topology proof','End-to-end full-timeline compound identity continuity'],**identity()));monitor.check()
