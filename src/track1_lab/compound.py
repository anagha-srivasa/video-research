"""Matched-window compound contour pilot with explicit correspondence/support limits."""
from collections import defaultdict
from itertools import permutations
import json,time
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import LinearRing
from shapely import linearrings,is_simple
from pipeline.common import ROOT
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.numerics import wavelet_matrix,basis
from track1.spatial import resample
from refinement.kernels import periodic_design
from refinement.data import holdout
from .architecture import general_fit,polynomial_fit
from .common import config,path,allowed,split,report,pool,identity
from .pilot32 import selected_cases,dest as pilot_directory
from .working import directory
from .variants import collocation,synthesis
from .motion import project
from .geometry_contract import normalize as normalize_rings,select as select_geometry,render_curves,topology_signature


def select_rings(source,native_size):
 scale=np.asarray(native_size)/[source['annotation_width'],source['annotation_height']]
 all_rings=normalize_rings(source['contours_json']);chosen=select_geometry(all_rings,3)
 rings=[dict(index=r['source_ring_index'],role=r['role'],parent_ring=r['parent_ring'],depth=r['depth'],area=abs(r['signed_area'])*float(scale.prod()),points=np.asarray(r['points'],float)*scale) for r in chosen]
 if not rings:raise ValueError('No representable rings')
 allocation=[32] if len(rings)==1 else [24,8] if len(rings)==2 else [16,8,8]
 return rings,tuple(allocation),tuple(r['role'] for r in rings),len(all_rings)


def match_rings(previous,current,centre,scale):
 if previous is None:return current,False
 cost=np.full((len(previous),len(current)),np.inf)
 for i,a in enumerate(previous):
  for j,b in enumerate(current):
   if a['role']==b['role'] and a.get('depth',0)==b.get('depth',0):
    relative=(b['points'].mean(0)-centre)/scale;cost[i,j]=np.linalg.norm(a['relative_centre']-relative)+.15*abs(np.log(max(b['area'],1)/max(a['area'],1)))
 choices=sorted((sum(cost[i,j] for i,j in enumerate(order)),order) for order in permutations(range(len(current))))
 if not np.isfinite(choices[0][0]):return None,True
 # ponytail: <=3 rings, enumerate <=6 assignments; use Hungarian assignment if the ring budget grows.
 ambiguous=len(choices)>1 and choices[1][0]-choices[0][0]<.05
 return [current[j] for j in choices[0][1]],ambiguous


def encode_sequence(frame_ids,mode,frames,fit_ring):
 previous=None;features=[];kept=[];excluded=defaultdict(int);phases=0
 allocation,roles=mode;source_signature=frames[frame_ids[-1]].get('source_topology') if len(frame_ids) else None
 for i in frame_ids:
  frame=frames[i]
  if (frame['allocation'],frame['roles'])!=mode or source_signature is not None and frame.get('source_topology')!=source_signature:
   excluded['topology_mode']+=1;excluded['discarded_prior_epoch_observations']+=len(kept);features=[];kept=[];previous=None;continue
  ordered,ambiguous=match_rings(previous,frame['rings'],frame['centre'],frame['scale'])
  if ambiguous:
   excluded['ambiguous_matching']+=1;excluded['discarded_ambiguous_epoch_observations']+=len(kept);features=[];kept=[];previous=None;continue
  controls=[];descriptors=[];valid=True
  for slot,(ring,m) in enumerate(zip(ordered,allocation)):
   result=fit_ring(frame,ring,m)
   if not result['valid']:valid=False;break
   p=result['controls'].copy()
   if previous is not None:
    old=previous[slot]['controls'];old_shape=(old-old.mean(0))/max(float(np.linalg.norm(old-old.mean(0))),1e-12);shape=(p-p.mean(0))/max(float(np.linalg.norm(p-p.mean(0))),1e-12);shift=int(np.argmin([np.sum((np.roll(shape,-q,axis=0)-old_shape)**2) for q in range(m)]));p=np.roll(p,-shift,axis=0);phases+=int(shift!=0)
   controls.append(p);descriptors.append(dict(role=ring['role'],depth=ring.get('depth',0),area=ring['area'],relative_centre=(ring['points'].mean(0)-frame['centre'])/frame['scale'],controls=p))
  if not valid:excluded['invalid_ring_fit']+=1;continue
  shape=np.concatenate([(wavelet_matrix(m)@((p-frame['centre'])/frame['scale'])).reshape(-1) for p,m in zip(controls,allocation)]);h=np.r_[frame['position'],shape,frame['log_scale']];features.append(h);kept.append(i);previous=descriptors
 return np.asarray(kept,dtype=int),np.asarray(features),dict(excluded),phases


def loop_validity(coefficients,m):
 q=len(coefficients);time=np.unique(np.r_[np.linspace(-1,1,65),np.polynomial.legendre.leggauss(q)[0]]);shape=(basis(time,degree=q-1)@coefficients).reshape(-1,m,2);curves=np.einsum('qn,tnc->tqc',synthesis(m,512),shape);area=np.sum(curves[:,:,0]*np.roll(curves[:,:,1],-1,axis=1)-curves[:,:,1]*np.roll(curves[:,:,0],-1,axis=1),axis=1)
 return np.asarray(is_simple(linearrings(curves)),bool)&(area>1e-8)


def guard_loops(coefficients,covariance,observed,allocation):
 c=coefficients.copy();cov=covariance.copy();offset=2;factors=[];valid=True
 for m in allocation:
  sl=slice(offset,offset+2*m);original=c[:,sl].copy();offset+=2*m
  if loop_validity(original.astype(np.float32).astype(float),m).all():factors.append(1.);continue
  references=np.argsort(np.sum((observed[:,sl]-original[0])**2,axis=1))[:min(3,len(observed))];best=None;phi=basis(np.linspace(-1,1,17),degree=len(c)-1)
  for index in references:
   for alpha in [.9,.75,.5,.25,0.]:
    candidate=original*alpha;candidate[0]+=(1-alpha)*observed[index,sl]
    if loop_validity(candidate.astype(np.float32).astype(float),m).all():
     loss=float(np.mean((phi@(candidate-original))**2))
     if best is None or loss<best[0]:best=(loss,alpha,candidate)
     break
  if best is None:factors.append(None);valid=False;continue
  _,alpha,candidate=best;c[:,sl]=candidate;delta=candidate-original;cov[sl]*=alpha**2;cov[sl]+=np.einsum('id,jd->dij',delta,delta);factors.append(alpha)
 return dict(coefficients=c,covariance=cov,factors=factors,valid=valid,applied=any(a!=1. for a in factors))


def render(h,allocation,roles,camera,target_shape):
 import cv2
 centre=project(h[:2],camera);k=np.asarray(camera['intrinsic_native']);clipped=bool(h[-1]<-15 or h[-1]>2);scale=np.sqrt(k[0,0]*k[1,1])*np.exp(np.clip(h[-1],-15,2));height,width=target_shape;native=np.asarray(camera['native_size']);curves=[];offset=2;simple=True
 for m,role in zip(allocation,roles):
  p=wavelet_matrix(m).T@h[offset:offset+2*m].reshape(m,2)*scale+centre;offset+=2*m;curve=periodic_design(np.arange(512)/512,m=m)@p;simple&=bool(LinearRing(curve).is_simple);curves.append((role,curve*[width/native[0],height/native[1]]))
 try:mask=render_curves([p for _,p in curves],[role for role,_ in curves],target_shape)
 except ValueError:mask=np.zeros(target_shape,np.uint8);simple=False
 return mask,simple,clipped


def video(v):
 import cv2
 from pipeline import Dataset
 from track1.evaluation import geometry_metrics
 cv2.setNumThreads(1);allowed([v]);started=time.monotonic();sp=load_shard(pilot_directory()/f'{v}.pt');rows=pq.read_table(pilot_directory()/f'{v}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r['row_index'] for r in rows};wanted={r['mask_uid'] for r in rows};source={r['mask_uid']:r for r in pq.read_table(ROOT/f'canonical/v1/masks.{v}.parquet').to_pylist() if r['mask_uid'] in wanted};base=load_shard(directory('spatial')/f'{v}.pt');camera=json.loads((directory('spatial')/f'{v}.camera.json').read_text());frames={};cache={}
 for r in rows:
  i=r['row_index'];original=r['original_row_index'];rings,allocation,roles,total=select_rings(source[r['mask_uid']],camera['native_size']);frames[i]=dict(uid=r['mask_uid'],rings=rings,allocation=allocation,roles=roles,total_rings=total,centre=base['centre_native'][original].astype(float),scale=float(base['scale_native'][original]),position=sp['h'][i,:2].astype(float),log_scale=float(sp['h'][i,-1]))
  frames[i]['source_topology']=topology_signature(source[r['mask_uid']]['contours_json'])

 def fit_ring(frame,ring,m):
  key=(frame['uid'],ring['index'],m)
  if key not in cache:cache[key]=general_fit(resample(ring['points'],128),m=m)
  return cache[key]
 metrics=[];ledger=[];coefficients=[];covariances=[];collocations=[];guarded_coefficients=[];guarded_covariances=[];guarded_collocations=[];states=[]
 with Dataset() as ds:
  for case_number,case in enumerate(selected_cases(v)):
   t=case['time'];ids=np.array([lookup[(case['entity_track_id'],int(case['window_end_us']-1000000+round(tt*1e6)))] for tt in t]);slot={i:j for j,i in enumerate(ids)}
   for protocol in ['single','gap100','endpoint100']:
    hidden=holdout(t,protocol,.875);available=(~hidden)&(t>=.5)
    from .hybrid import source_eligibility
    source_eligible,source_reason=source_eligibility(t,hidden)
    if not hidden.any() or available.sum()<8:
     ledger.append(dict(video_id=v,split=split(v),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,eligible_input_observations=int(available.sum()),used_input_observations=0,supported=False,allocation=None,roles=None,target_mode_changed=None,excluded_topology=0,excluded_matching=0,excluded_invalid_fit=0,phase_corrections=0,source_support_eligible=source_eligible,source_ineligibility_reason=source_reason,independently_eligible=None));continue
    last=frames[ids[np.flatnonzero(available)[-1]]];mode=(last['allocation'],last['roles']);kept,compound,excluded,phases=encode_sequence(ids[available],mode,frames,fit_ring);positions=np.array([slot[i] for i in kept]);tt=t[positions];span=float(tt[-1]-tt[0]) if len(tt) else 0.;gap=float(np.max(np.diff(np.r_[.5,tt,1.]))) if len(tt) else .5;stable_epoch=not excluded.get('topology_mode',0) and not excluded.get('ambiguous_matching',0);supported=len(kept)>=8 and span>=.3 and gap<=.175 and stable_epoch
    target_i=np.flatnonzero(hidden)[-1];target_frame=frames[ids[target_i]];target=ds.get_mask(target_frame['uid']);target_changed=(target_frame['allocation'],target_frame['roles'])!=mode;ledger.append(dict(video_id=v,split=split(v),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,eligible_input_observations=int(available.sum()),used_input_observations=len(kept),supported=bool(supported),allocation=list(mode[0]),roles=list(mode[1]),target_mode_changed=target_changed,excluded_topology=excluded.get('topology_mode',0),excluded_matching=excluded.get('ambiguous_matching',0),excluded_invalid_fit=excluded.get('invalid_ring_fit',0),phase_corrections=phases))
    ledger[-1].update(source_support_eligible=source_eligible,source_ineligibility_reason=source_reason,independently_eligible=None)
    methods=[('single_all',np.flatnonzero(available),sp['h'][ids[available]].astype(float),(32,),('outer',))]
    if supported:methods += [('single_paired',positions,sp['h'][kept].astype(float),(32,),('outer',)),('compound',positions,compound,mode[0],mode[1])]
    for method,index,h,allocation,roles in methods:
     f=polynomial_fit(t[index],h,np.zeros_like(h),case['prior'][index],case['quality'][index],.5,3,3.,np.ones(67));guarded=guard_loops(f['coefficients'],f['covariance'],h,allocation) if method=='compound' else None;outputs=[(method,f['coefficients'],[1.]*len(allocation))]
     if guarded is not None:outputs.append(('compound_guarded',guarded['coefficients'],guarded['factors']))
     for output_method,c,factors in outputs:
      prediction=basis(2*(t[target_i]-.5)/.5-1)@c;mask,simple,clipped=render(prediction,allocation,roles,camera,target.shape);gm=geometry_metrics(target,mask);offset=2;window_valid=True
      for mm in allocation:window_valid&=bool(loop_validity(c[:,offset:offset+2*mm],mm).all());offset+=2*mm
      metrics.append(dict(video_id=v,split=split(v),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,method=output_method,compound_supported=bool(supported),target_mode_changed=target_changed,target_source_kind=rows[ids[target_i]]['source_kind'],target_rings=target_frame['total_rings'],represented_rings=len(allocation),allocation=list(allocation),roles=list(roles),input_observations=len(index),geometry_valid=bool(simple),sampled_window_geometry_valid=window_valid,guard_applied=any(a!=1. for a in factors),guard_factors=factors,scale_clipped=clipped,**gm))
     if method=='compound':
      nodes=collocation(f['coefficients'],.5);gnodes=collocation(guarded['coefficients'],.5);coefficients.append(f['coefficients'].astype(np.float32));covariances.append(f['covariance'].astype(np.float32));collocations.append(nodes['xi'].astype(np.float32));guarded_coefficients.append(guarded['coefficients'].astype(np.float32));guarded_covariances.append(guarded['covariance'].astype(np.float32));guarded_collocations.append(gnodes['xi'].astype(np.float32));states.append(dict(video_id=v,row_index=len(states),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],window_seconds=.5,protocol=protocol,allocation=list(allocation),roles=list(roles),guard_factors=guarded['factors'],guarded_geometry_valid=guarded['valid'],guard_spatial_samples=512,guard_temporal_samples=69,guard_precision='float32',correspondence_scope='within_retained_window_only',cross_window_correspondence_verified=False,covariance_calibrated=False))
   if case_number%8==0:memory_guard(config())
 out=path('derived')/'compound32_pilot';out.mkdir(exist_ok=True);save_index(out/f'{v}.parquet',states)
 if coefficients:save_shard(out/f'{v}.pt',dict(coefficients=np.stack(coefficients),coefficient_covariance_uncalibrated=np.stack(covariances),xi=np.stack(collocations),guarded_coefficients=np.stack(guarded_coefficients),guarded_coefficient_covariance_uncalibrated=np.stack(guarded_covariances),guarded_xi=np.stack(guarded_collocations)))
 save_index(path('reports')/f'compound32.{v}.parquet',metrics);save_index(path('reports')/f'compound32_eligibility.{v}.parquet',ledger);return dict(video_id=v,proposed_windows=len(ledger),supported=sum(r['supported'] for r in ledger),fitted_states=len(states),seconds=time.monotonic()-started)


def run(monitor):
 cfg=config();jobs=pool(video,cfg['train']+cfg['validation'],monitor,workers=7);metrics=[];ledger=[]
 for v in cfg['train']+cfg['validation']:metrics+=pq.read_table(path('reports')/f'compound32.{v}.parquet').to_pylist();ledger+=pq.read_table(path('reports')/f'compound32_eligibility.{v}.parquet').to_pylist()
 save_index(path('reports')/'compound32.parquet',metrics);save_index(path('reports')/'compound32_eligibility.parquet',ledger);summary={}
 for protocol in ['single','gap100','endpoint100']:
  summary[protocol]={}
  for method in ['single_all','single_paired','compound','compound_guarded']:
   byvideo={}
   for v in cfg['validation']:
    rs=[r for r in metrics if r['video_id']==v and r['protocol']==protocol and r['method']==method and r['compound_supported']]
    if rs:byvideo[v]=dict(n=len(rs),iou=float(np.mean([r['iou'] for r in rs])),boundary_fscore_2px=float(np.mean([r['boundary_fscore_2px'] for r in rs])),geometry_valid_fraction=float(np.mean([r['geometry_valid'] for r in rs])),sampled_window_geometry_valid_fraction=float(np.mean([r['sampled_window_geometry_valid'] for r in rs])))
   if byvideo:summary[protocol][method]=dict(**{k:float(np.mean([r[k] for r in byvideo.values()])) for k in ['iou','boundary_fscore_2px','geometry_valid_fraction','sampled_window_geometry_valid_fraction']},by_video=byvideo)
 report('compound32.json',dict(videos=jobs,paired_validation_metrics=summary,limits=['Same total 32 controls; per-window allocations 32,24+8,16+8+8 and explicit outer/hole roles.','Matching uses retained observations only; hidden target topology never selects input eligibility.','single_paired and compound use identical retained timestamps and uniform variance weights. single_all is the full-input baseline on the same supported target cohort.','Unsupported windows and target topology changes are retained in a separate ledger; paired scores are conditional on declared support.','Per-ring guard checks 69 sampled times at the renderer’s 512 spatial samples and float32 output precision; preserves position/scale and unaffected loops, and can suppress shape motion; raw and guarded scores are both retained.',
        'Loop-only validity does not certify inter-loop hole containment or disjointness, or validity at every continuous time.',
        'Correspondence is within-window only. No cross-window sequence index, calibrated compound covariance, default ingestion interface or Track 2 model is exported.'],state='pilot_not_finalized',**identity()))
