"""Bounded remediation pilots; never freezes Track 1 or starts Track 2."""
from collections import defaultdict
import json,time
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,digest
from track1.artifacts import load_shard,save_shard,save_index
from track1.numerics import basis
from refinement.data import holdout
from .common import config,path,allowed,split,pool,report,identity,audit,LOG
from .pilot32 import selected_cases
from .topology_certificate import certify
from .history_detail import fit_detail,evaluate,multiscale_summary,detail_basis

def certificate_video(v):
 allowed([v]);folder=path('derived')/'compound32_topology';data=load_shard(folder/f'{v}.pt');rows=pq.read_table(folder/f'{v}.parquet').to_pylist();results=[];start=time.monotonic();cases={(c['entity_track_id'],c['window_end_us']):c for c in selected_cases(v)};pilot=pq.read_table(path('derived')/'pilot32'/f'{v}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r for r in pilot};events=[];old=pq.read_table(path('reports')/f'compound32_eligibility.{v}.parquet').to_pylist();ambiguity={(r['entity_track_id'],r['window_end_us'],r['protocol']):(r.get('excluded_matching') or 0)>0 for r in old}
 for r in rows:
  case=cases[(r['entity_track_id'],r['window_end_us'])];keep=(~holdout(case['time'],r['protocol'],.875))&(case['time']>=.5);observed=[lookup[(r['entity_track_id'],r['window_end_us']-1000000+int(round(t*1e6)))] for t in case['time'][keep]];changed=[]
  for a,b in zip(observed,observed[1:]):
   before=(a['component_count'],a['hole_count']);after=(b['component_count'],b['hole_count'])
   if before!=after:changed.append(dict(video_id=v,entity_track_id=r['entity_track_id'],start_us=a['timestamp_us'],end_us=b['timestamp_us'],components_before=before[0],holes_before=before[1],components_after=after[0],holes_after=after[1],event_kind='annotation_topology_transition',lineage_verified=False,physical_split_or_merge_verified=False))
  events+=changed;r['topology_epoch_valid']=not changed;r['quality_flags']['topology_epoch_valid']=not changed;r['ring_correspondence_epoch_valid']=not ambiguity[(r['entity_track_id'],r['window_end_us'],r['protocol'])];r['quality_flags']['ring_correspondence_epoch_valid']=r['ring_correspondence_epoch_valid']
  result=certify(data['coefficients'][r['row_index']],r['allocation'],r['roles'],r['ring_parent'],r['window_seconds']);results.append(dict(video_id=v,split=split(v),row_index=r['row_index'],entity_track_id=r['entity_track_id'],window_end_us=r['window_end_us'],protocol=r['protocol'],sampled_topology_valid=r['quality_flags']['ring_topology_valid'],topology_epoch_valid=not changed,ring_correspondence_epoch_valid=r['ring_correspondence_epoch_valid'],known_annotation_transitions=len(changed),**result))
  if len(results)%20==0:LOG.info('Certificate %s %d/%d',v,len(results),len(rows))
 for r,result in zip(rows,results):
  r['topology_certificate']={k:result[k] for k in ['certified','valid','min_clearance','clearance_semantics','temporal_resolution','minimum_leaf_seconds','failure_code','tolerance','precision','decoder_version','scope','coefficient_sha256','geometry_contract_sha256','decoder_source_sha256']};r['quality_flags']['topology_certified']=result['certified'];r['sampled_topology_accepted']=r['accepted'];r['certified_topology_accepted']=bool(r['accepted'] and result['certified'] and r['topology_epoch_valid'] and r['ring_correspondence_epoch_valid'])
 save_index(folder/f'{v}.parquet',rows)
 events=list({(r['entity_track_id'],r['start_us'],r['end_us']):r for r in events}.values());save_index(path('reports')/f'topology_events.{v}.parquet',events)
 save_index(path('reports')/f'topology_certificate.{v}.parquet',results);return dict(video_id=v,states=len(results),certified=sum(r['certified'] for r in results),seconds=time.monotonic()-start)

def detail_video(v):
 allowed([v]);base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());scale=np.asarray(base['feature_scale']);trials=[dict(count=n,strength=s) for n in [2,3] for s in [.01,.1]]
 if split(v)=='validation':trials=[json.loads((path('reports')/'neutral_detail_selection.json').read_text())['parameters']]
 records=[]
 for c in selected_cases(v):
  for protocol in ['single','gap100','endpoint100']:
   hidden=holdout(c['time'],protocol,.5);keep=~hidden
   if not hidden.any() or keep.sum()<8:continue
   for parameters in trials:
    model=fit_detail(c['time'][keep],c['h'][keep],c['variance'][keep],c['prior'][keep],c['quality'][keep],base['temporal']['options'],scale,**parameters);query=c['time'][hidden];truth=c['h'][hidden,:2];coarse=(basis(2*query-1)@model['coarse'])[:,:2];pred=evaluate(model,query)[:,:2];records.append(dict(video_id=v,split=split(v),entity_track_id=c['entity_track_id'],window_end_us=c['window_end_us'],protocol=protocol,parameter_id=json.dumps(parameters,sort_keys=True),coarse_mse=float(np.mean(((coarse-truth)/scale[:2])**2)),detail_mse=float(np.mean(((pred-truth)/scale[:2])**2)),orthogonality_error=model['orthogonality_error']))
 save_index(path('reports')/f'neutral_detail.{v}.parquet',records);return dict(video_id=v,probes=len(records))

def select_detail():
 rows=[]
 for v in config()['train']:rows+=pq.read_table(path('reports')/f'neutral_detail.{v}.parquet').to_pylist()
 trials=[]
 for key in sorted({r['parameter_id'] for r in rows}):
  rs=[r for r in rows if r['parameter_id']==key];metrics={p:{name:float(np.mean([np.sqrt(np.mean([r[name+'_mse'] for r in rs if r['video_id']==v and r['protocol']==p])) for v in config()['train']])) for name in ['coarse','detail']} for p in ['single','gap100','endpoint100']};trials.append(dict(parameters=json.loads(key),metrics=metrics,objective=sum(r['detail'] for r in metrics.values()),interior_guard=all(metrics[p]['detail']<=1.05*metrics[p]['coarse'] for p in ['single','gap100'])))
 best=min([r for r in trials if r['interior_guard']] or trials,key=lambda r:r['objective']);report('neutral_detail_selection.json',dict(**best,trials=trials,selection_scope='training participants only; validation exploratory',promoted=False,**identity()));return best

def store_detail(monitor):
 base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());scale=np.asarray(base['feature_scale']);parameters=json.loads((path('reports')/'neutral_detail_selection.json').read_text())['parameters'];folder=path('derived')/'endpoint_neutral_detail';folder.mkdir(exist_ok=True);counts=0;max_rebuild=0.;max_orth=0.;max_stored_orth=0.
 for v in config()['train']+config()['validation']:
  arrays=defaultdict(list);meta=[]
  for c in selected_cases(v):
   model=fit_detail(c['time'],c['h'],c['variance'],c['prior'],c['quality'],base['temporal']['options'],scale,**parameters);features,valid=multiscale_summary(c,model['coarse'],scale)
   t=c['time'][(c['time']>=0)&(c['time']<=1)];stored_transform=model['transform'].astype(np.float32).astype(float);stored_orth=float(np.max(abs(basis(2*t-1).T@(model['weights'][:,None]*(detail_basis(t)@stored_transform)))));max_stored_orth=max(max_stored_orth,stored_orth)
   for k,a in [('coarse',model['coarse']),('detail',model['detail']),('transform',model['transform']),('multiscale',features),('multiscale_valid',valid)]:arrays[k].append(a)
   meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=c['entity_track_id'],window_end_us=c['window_end_us'],count=parameters['count'],cutoff=1.,endpoint_neutral=True,detail_role='reconstruction_only',orthogonality_error=model['orthogonality_error'],latest_input_us=int(c['window_end_us']-1000000+round(float(c['time'][c['time']<=1].max())*1e6)),representation_version='track1-v3.0.0-dev.neutral-detail1'))
  tensors={k:np.asarray(a,bool if k=='multiscale_valid' else np.float32) for k,a in arrays.items()};save_shard(folder/f'{v}.pt',tensors);save_index(folder/f'{v}.parquet',meta)
  for r in meta:
   i=r['row_index'];m=dict(coarse=tensors['coarse'][i].astype(float),detail=tensors['detail'][i].astype(float),transform=tensors['transform'][i].astype(float),cutoff=1.)
   for derivative in [0,1]:
    error=float(np.max(abs(evaluate(m,[1.],derivative)-(basis(np.array([1.]),derivative)*2**derivative)@m['coarse'])));max_rebuild=max(max_rebuild,error)
   assert r['latest_input_us']<=r['window_end_us'];max_orth=max(max_orth,r['orthogonality_error'])
  counts+=len(meta);monitor.check()
 assert max_rebuild==0. and max_stored_orth<1e-5,(max_rebuild,max_stored_orth)
 report('neutral_detail_store.json',dict(states=counts,exact_serialized_endpoint_difference=max_rebuild,max_fit_orthogonality_error=max_orth,max_serialized_weighted_orthogonality_error=max_stored_orth,detail_in_core_collocation=False,optional_multiscale_feature_names=['delta_velocity_0.5_x','delta_velocity_0.5_y','delta_velocity_0.25_x','delta_velocity_0.25_y','short_to_long_history_error_energy','recent_observation_quality'],detail_is_authoritative_velocity=False,promoted=False,**identity()))

def summarize():
 certificates=[];details=[]
 for v in config()['train']+config()['validation']:
  certificates+=pq.read_table(path('reports')/f'topology_certificate.{v}.parquet').to_pylist();details+=pq.read_table(path('reports')/f'neutral_detail.{v}.parquet').to_pylist()
 save_index(path('reports')/'topology_certificate.parquet',certificates);save_index(path('reports')/'neutral_detail.parquet',details);key=json.dumps(json.loads((path('reports')/'neutral_detail_selection.json').read_text())['parameters'],sort_keys=True);validation={}
 for protocol in ['single','gap100','endpoint100']:
  rs=[r for r in details if r['split']=='validation' and r['protocol']==protocol and r['parameter_id']==key];validation[protocol]={name:float(np.mean([np.sqrt(np.mean([r[name+'_mse'] for r in rs if r['video_id']==v])) for v in config()['validation']])) for name in ['coarse','detail']}
 groups={}
 for part in ['train','validation']:
  rs=[r for r in certificates if r['split']==part];groups[part]=dict(attempted=len(rs),sampled_valid=sum(r['sampled_topology_valid'] for r in rs),certified=sum(r['certified'] for r in rs),certified_and_epoch_valid=sum(r['certified'] and r['topology_epoch_valid'] and r['ring_correspondence_epoch_valid'] for r in rs),known_epoch_crossing_probes=sum(not r['topology_epoch_valid'] for r in rs),known_ambiguous_correspondence_probes=sum(not r['ring_correspondence_epoch_valid'] for r in rs),unresolved=sum(not r['certified'] for r in rs),failure_codes={code:sum(r['failure_code']==code for r in rs) for code in sorted({r['failure_code'] for r in rs if r['failure_code']})},largest_certified_temporal_leaf_seconds=max((r['temporal_resolution'] for r in rs if r['certified']),default=None))
 report('topology_certificate.json',dict(metrics=groups,tolerance=1e-8,arithmetic='Outward-rounded interval transforms and de Casteljau subdivision; Bernstein convex hulls; local injectivity and adjacent tangent cones; fixed hierarchy anchor',fail_closed=True,limits=['Continuous normalized cubic curves of serialized float32 coefficients; not raster topology, hidden source rings, identity or contact truth.','A failed or exhausted certificate is unresolved, not automatically an intersection.','No accepted/default store is silently replaced; sampled-valid states without a certificate remain flagged separately.'],**identity()));report('neutral_detail.json',dict(validation=validation,endpoint_neutral=True,coarse_unchanged=True,promoted=False,**identity()))

def run(monitor):
 audit();cfg=config();pool(certificate_video,cfg['train']+cfg['validation'],monitor,workers=7);pool(detail_video,cfg['train'],monitor,workers=5);select_detail();pool(detail_video,cfg['validation'],monitor,workers=2);store_detail(monitor);summarize()
