"""Matched source-contour experiments; every attempted holdout stays in the ledger."""
import copy
import os
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from track1.numerics import basis,wavelet_matrix
from track1_retrain.temporal import fit,fit_detail,target_h
from track1_retrain.evaluation import curves,matched_masks,source_rows
from track1_lab.history_detail import evaluate as detail_evaluate
from track1_lab.topology_certificate import certify
from track1_finetune.data import ROOT,BASES,VIDEOS,OLD_CFG,split,save,array,rank,cases_for
from track1_finetune.fit import probe_data
from .phase import align,phase_discontinuity
from .geometry import repair,rollback

D=Path(os.environ.get('TRACK1_REVISION_DATA',str(ROOT/'derived/track1_encoder_revision'))).resolve();R=Path(os.environ.get('TRACK1_REVISION_REPORT',str(ROOT/'reports/track1_encoder_revision'))).resolve();W=Path(os.environ.get('TRACK1_REVISION_WORK',str(ROOT/'work/track1_encoder_revision'))).resolve()
for path in [D,R,W]:
 if not path.is_relative_to(ROOT):raise ValueError('All outputs must remain inside project root')
CFG=json.loads((ROOT/'configs/track1_encoder_revision.json').read_text())
VARIANTS=['A_discrete','B_circular','C_deformation','D_detail']

def worker(video):
 model=json.loads((ROOT/'derived/track1_finetuned/v2/calibration/model.json').read_text())['32'];scale=np.array(model['feature_scale']);probes,attempts=probe_data(video,32,CFG['phase_cases_per_video']);_,inputs=cases_for(video,32,CFG['phase_cases_per_video']);wanted={p['target_uid'] for p in probes}
 for p in probes:
  rows,_=inputs[p['identity']['source']];wanted.update(rows[i]['mask_uid'] for i in p['p']['ids'])
 source=source_rows(video,wanted);names={r['mask_uid']:str(r['entity_name']) for r in pq.read_table(ROOT/f'canonical/v2/spatial_metadata.{video}.parquet',columns=['mask_uid','entity_name']).to_pylist() if r['mask_uid'] in wanted};camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{video}.camera.json').read_text());native=np.array(camera['native_size']);records=[];metadata=[];coefficients=[];geometry_coefficients=[];repairs=[];static_cache={};repair_count=0
 for probe in probes:
  p=probe['p'];rows,data=inputs[probe['identity']['source']];scales=[rows[i]['scale_native'] for i in p['ids']]
  # The guard compares each retained source mask before/after fractional reparameterization.
  def static_guard(i,candidate,original):
   uid=rows[p['ids'][i]]['mask_uid'];key=(uid,tuple(candidate[2:-1]))
   if key not in static_cache:
    predictions=[(curves(h,p['allocation'],camera),p['roles'],p['parents']) for h in [original,candidate]];a,b=matched_masks(source[uid],predictions,native);static_cache[key]=b['iou']>=a['iou']-1e-12 and b['boundary_fscore_2px']>=a['boundary_fscore_2px']-1e-12
   return static_cache[key]
  aligned,phase=align(p,scales,limit=CFG['fractional_phase_limit_controls'],smoothness=CFG['phase_smoothness'],max_displacement_px=CFG['maximum_sampled_static_displacement_px'],geometry_guard=static_guard)
  options=copy.deepcopy(model['options']);options['families']['shape']['regularization']+=CFG['weak_shape_regularization_increment'];fits={};predictions={}
  for variant in VARIANTS:
   inp=p if variant=='A_discrete' else aligned;opt=model['options'] if variant in ['A_discrete','B_circular'] else options
   f=fit(inp['time'],inp['h'],inp['variance'],inp['prior'],inp['quality'],opt,scale);pred=basis(2*probe['times']-1)@f['coefficients'];detail=None
   if variant=='D_detail':
    detail=fit_detail(inp['time'],inp['h'],inp['variance'],inp['prior'],inp['quality'],opt,scale,count=model['detail']['count'],strength=model['detail']['strength']) if model['detail']['count'] else None
    if detail is not None:pred=detail_evaluate(detail,probe['times'])
   predictions[variant]=(curves(pred[probe['target_at']],p['allocation'],camera),p['roles'],p['parents']);fits[variant]=(f,pred,inp)
  metrics=matched_masks(source[probe['target_uid']],list(predictions.values()),native)
  target_rows,target_data=inputs[probe['identity']['source']]
  target_index=next(i for i,r in enumerate(target_rows) if r['mask_uid']==probe['target_uid']);target_row=target_rows[target_index]
  spatial_oracle=matched_masks(source[probe['target_uid']],[(curves(np.array(target_data['h'][target_index]),target_row['allocation'],camera),target_row['roles'],target_row['parents'])],native)[0]
  base_target=probe['targets'][probe['target_at']];base_h=fits['A_discrete'][1][probe['target_at']].copy();base_h[:2]=base_target[:2]
  oracle_pred=(curves(base_h,p['allocation'],camera),p['roles'],p['parents']);oracle_metric=matched_masks(source[probe['target_uid']],[oracle_pred],native)[0]
  # A target-informed phase oracle changes retained gauge, never considered for selection.
  oracle=copy.deepcopy(p);offset=2
  for ring,m in enumerate(p['allocation']):
   from .phase import circular_matrix
   from scipy.optimize import minimize_scalar
   reference=wavelet_matrix(m).T@base_target[offset:offset+2*m].reshape(m,2)
   for j in range(len(oracle['h'])):
    ctrl=wavelet_matrix(m).T@p['h'][j,offset:offset+2*m].reshape(m,2);result=minimize_scalar(lambda x:float(np.mean((circular_matrix(m,x)@ctrl-reference)**2)),bounds=(-.5,.5),method='bounded');oracle['h'][j,offset:offset+2*m]=(wavelet_matrix(m)@circular_matrix(m,result.x)@ctrl).ravel()
   offset+=2*m
  of=fit(oracle['time'],oracle['h'],oracle['variance'],oracle['prior'],oracle['quality'],model['options'],scale);oh=basis(2*probe['times'][probe['target_at']]-1)@of['coefficients'];om=matched_masks(source[probe['target_uid']],[(curves(oh,p['allocation'],camera),p['roles'],p['parents'])],native)[0]
  identity=probe['identity'];target_kind=probe['target_kind']
  for variant,gm in zip(VARIANTS,metrics):
   f,pred,inp=fits[variant];c=f['coefficients'].astype(np.float32);cert=certify(c,p['allocation'],p['roles'],p['parents'],window_seconds=1.,max_depth=CFG['certificate_max_depth'],max_nodes=CFG['certificate_max_nodes']);error=(pred-probe['targets'])/scale
   record=dict(identity,variant=variant,iou=gm['iou'],bf2=gm['boundary_fscore_2px'],position_nrmse=float(np.sqrt(np.mean(error[:,:2]**2))),certificate_pass=cert['certified'],target_kind=target_kind,target_topology_compatible=bool(probe['compatible'][probe['target_at']]),rings=len(p['allocation']),entity_name=names.get(probe['target_uid'],'unknown'),entity_type='hand' if 'hand' in names.get(probe['target_uid'],'').lower() else 'object',phase_discontinuity=phase_discontinuity(phase['offsets_controls']) if variant!='A_discrete' else None,static_guard_rejections=phase['rejected_shifts_or_frames'] if variant!='A_discrete' else 0,static_guard_verified=True,retained_only=True,phase_uncertainty_calibrated=False,centroid_oracle_iou=oracle_metric['iou'],centroid_oracle_bf2=oracle_metric['boundary_fscore_2px'],target_informed_phase_oracle_iou=om['iou'] if probe['compatible'][probe['target_at']] else None,target_informed_phase_oracle_bf2=om['boundary_fscore_2px'] if probe['compatible'][probe['target_at']] else None,spatial_target_oracle_iou=spatial_oracle['iou'],spatial_target_oracle_bf2=spatial_oracle['boundary_fscore_2px'],spatial_oracle_changes_shape_and_topology=True,oracle_uses_hidden_target=True)
   records.append(record);index=len(metadata);metadata.append(dict(identity,variant=variant,row_index=index,window_start_us=identity['window_end_us']-1000000,window_seconds=1.,allocation=p['allocation'],roles=p['roles'],parents=p['parents'],certificate_json=json.dumps(cert),certificate_valid=cert['certified'],source_complete=all(rows[i]['source_complete'] for i in p['ids']),latest_input_us=max(rows[i]['timestamp_us'] for i in p['ids']),retained_mask_uids=[rows[i]['mask_uid'] for i in p['ids']],pilot_only=True,independent_geometry_verified=False,geometry_covariance_available=False,phase_metadata_json=json.dumps(phase) if variant!='A_discrete' else None));coefficients.append(c);geometry_coefficients.append(c)
  if repair_count<CFG['repair_cases_per_video']:
   repair_count+=1;f,pred,inp=fits['C_deformation'];result=repair(f['coefficients'],inp['h'][-1],p['allocation'],p['roles'],p['parents'],inp['time'],inp['h'],weights=inp['prior']**.5*inp['quality']);baseline,bcert,amount=rollback(f['coefficients'],inp['h'][-1],p['allocation'],p['roles'],p['parents'],dict(window_seconds=1.,max_depth=8,max_nodes=255));stats=[]
   for name,c,cc in [('global_rollback',baseline,bcert),('local_projection',result['coefficients'],result['certificate'])]:
    if c is None:stats.append(dict(method=name,certificate_pass=False,iou=None,bf2=None));continue
    h=basis(2*probe['times'][probe['target_at']]-1)@c;gm=matched_masks(source[probe['target_uid']],[(curves(h,p['allocation'],camera),p['roles'],p['parents'])],native)[0];stats.append(dict(method=name,certificate_pass=cc['certified'],iou=gm['iou'],bf2=gm['boundary_fscore_2px']))
   repairs.append(dict(identity,selected_method=result['method'],initial_failure_json=json.dumps(result['localization']),attempts_json=json.dumps(result['attempts']),comparison_json=json.dumps(stats),geometry_covariance_available=False))
   if result['coefficients'] is not None and result['method']=='local_projection':
    # Publish a distinct certificate-bound optional view, leaving historical coefficients intact.
    metadata.append(dict(identity,variant='local_projection',row_index=len(metadata),window_start_us=identity['window_end_us']-1000000,window_seconds=1.,allocation=p['allocation'],roles=p['roles'],parents=p['parents'],certificate_json=json.dumps(result['certificate']),certificate_valid=True,source_complete=all(rows[i]['source_complete'] for i in p['ids']),latest_input_us=max(rows[i]['timestamp_us'] for i in p['ids']),retained_mask_uids=[rows[i]['mask_uid'] for i in p['ids']],pilot_only=True,independent_geometry_verified=False,geometry_covariance_available=False,phase_metadata_json=None));coefficients.append(f['coefficients']);geometry_coefficients.append(result['coefficients'])
 save(R/f'phase.{video}.parquet',records);save(R/f'attempts.{video}.parquet',attempts);save(R/f'repair.{video}.parquet',repairs);save(D/f'{video}.parquet',metadata);array(D/f'{video}.coefficients.npy',coefficients);array(D/f'{video}.geometry.npy',geometry_coefficients)
 return dict(records=records,attempts=attempts,repairs=repairs,states=len(metadata),static_decisions=len(static_cache))
