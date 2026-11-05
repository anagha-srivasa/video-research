"""Matched reconstruction diagnostics, not independent dynamics acceptance."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from pipeline.geometry import _raster
from track1.artifacts import load_shard
from track1.numerics import basis
from track1.evaluation import geometry_metrics
from .temporal import fit
from .temporal import fit_detail
from track1_lab.history_detail import evaluate as evaluate_detail
from track1_lab.motion import project
from track1_lab.variants import synthesis
from data_v2.geometry import render
from .temporal import load,proposals,holdout,prepare,target_h,timing_eligible
from .common import *

def source_rows(v,wanted):
 result={}
 for batch in pq.ParquetFile(ROOT/f'canonical/v2/masks.{v}.parquet').iter_batches(batch_size=256,columns=['mask_uid','segments_raw','annotation_width','annotation_height','hole_count','component_count']):
  for r in batch.to_pylist():
   if r['mask_uid'] in wanted:result[r['mask_uid']]=r
 return result

def curves(h,allocation,camera):
 centre=project(h[:2],camera);k=np.asarray(camera['intrinsic_native']);scale=np.sqrt(k[0,0]*k[1,1])*np.exp(np.clip(h[-1],-15,2));out=[];at=2
 for m in allocation:out.append((synthesis(m,512)@h[at:at+2*m].reshape(m,2))*scale+centre);at+=2*m
 return out

def matched_masks(source,predictions,native_size):
 """Union ROI includes all target/prediction pixels, so cropping cannot hide FP."""
 target,tx,ty,_,_=_raster(json.loads(source['segments_raw']),source['annotation_width'],source['annotation_height'])
 if target is None:raise ValueError('Unexpected quarantined evaluation target')
 factor=np.array([source['annotation_width'],source['annotation_height']])/native_size
 points=[p*factor for item in predictions for p in item[0]];allpts=np.concatenate(points+[np.array([[tx,ty],[tx+target.shape[1],ty+target.shape[0]]])]);lo=np.maximum(np.floor(allpts.min(0)).astype(int)-3,0);hi=np.minimum(np.ceil(allpts.max(0)).astype(int)+4,[source['annotation_width'],source['annotation_height']]);w,h=hi-lo
 truth=np.zeros((h,w),np.uint8);truth[ty-lo[1]:ty-lo[1]+target.shape[0],tx-lo[0]:tx-lo[0]+target.shape[1]]=target
 metrics=[]
 for ps,roles,parents in predictions:
  rings=[dict(points=(p*factor-lo).tolist(),role=role,parent_ring=parent) for p,role,parent in zip(ps,roles,parents)];mask=render(rings,truth.shape);metrics.append(geometry_metrics(truth,mask))
 return metrics

def spatial_worker(v):
 rows,data=load(v,16);_,other=load(v,32);manual=[r for r in rows if r['source_kind']=='manual_sparse'];selected={r['mask_uid']:r for r in sorted(manual,key=lambda r:rank(r['mask_uid']))[:64]}
 for r in sorted([r for r in manual if r['hole_count']>0],key=lambda r:rank(r['mask_uid']))[:16]:selected[r['mask_uid']]=r
 source=source_rows(v,set(selected));old=load_shard(ROOT/f'derived/track1_refined_v2/spatial/{v}.pt');records=[]
 for uid,r in selected.items():
  index=r['row_index'];p16=np.asarray(data['controls'][index],float);p32=np.asarray(other['controls'][index],float);from refinement.kernels import periodic_design
  oldp=old['controls'][r['baseline_row_index']].astype(float);ps32=[];at=0
  for m in r['allocation']:ps32.append(periodic_design(np.arange(512)/512,m=m)@p32[at:at+m]);at+=m
  predictions=[([periodic_design(np.arange(512)/512)@oldp],['outer'],[-1]),([periodic_design(np.arange(512)/512)@p16],['outer'],[-1]),(ps32,r['roles'],r['parents'])]
  metrics=matched_masks(source[uid],predictions,np.array([r['width'],r['height']]))
  for method,gm in zip(['frozen_v2_16','refit16','refit32'],metrics):records.append(dict(video_id=v,split=split(v),mask_uid=uid,method=method,source_holes=r['hole_count'],source_components=r['component_count'],**gm))
 save(REPORT/f'spatial_quality.{v}.parquet',records);return records

def temporal_worker(job):
 v,n=job;rows,data=load(v,n);selected=json.loads((folder('calibration')/'model.json').read_text())[str(n)];scale=np.array(selected['feature_scale']);base=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text())['temporal']['options'];cases=sorted([c for c in proposals(v,rows) if timing_eligible(c,rows)],key=lambda c:rank(c['entity_track_id']+str(c['window_end_us'])))[:CFG['max_search_cases_per_video']];records=[];attempts=[]
 camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());wanted={rows[i]['mask_uid'] for c in cases for i in c['indices']};source=source_rows(v,wanted)
 for case in cases:
  for protocol in ['single','gap100','endpoint100']:
   t,hidden=holdout(case,rows,protocol);ledger=dict(video_id=v,split=split(v),model=n,entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,hidden_observations=int(hidden.sum()))
   if not hidden.any():attempts.append(dict(ledger,supported=False,reason='no_target'));continue
   try:p=prepare(case,rows,data,n,hidden)
   except ValueError as e:attempts.append(dict(ledger,supported=False,reason=str(e)));continue
   attempts.append(dict(ledger,supported=True,reason=None));ids=np.asarray(case['indices'])[hidden];targets,compatible=target_h(rows,data,ids,n,p);phi=basis(2*t[hidden]-1);f0=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],base,scale);f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],selected['options'],scale)
   predictions={'locked_v2_fitter':phi@f0['coefficients'],'refit_cubic':phi@f['coefficients']}
   if selected['detail']['count']:
    detail=fit_detail(p['time'],p['h'],p['variance'],p['prior'],p['quality'],selected['options'],scale,count=selected['detail']['count'],strength=selected['detail']['strength']);predictions['refit_with_neutral_detail']=evaluate_detail(detail,t[hidden])
   for method in ['nearest','linear']:predictions[method]=interp1d(p['time'],p['h'],axis=0,kind=method,fill_value='extrapolate')(t[hidden])
   # Mask reconstruction uses a deterministic target per probe, including target
   # topology changes. All targets still contribute to position/scale scores.
   target_at=int(rank(case['entity_track_id']+str(case['window_end_us'])+protocol)[:8],16)%len(ids);target_index=ids[target_at];r=rows[target_index];mask_predictions=[(curves(pred[target_at],p['allocation'],camera),p['roles'],p['parents']) for pred in predictions.values()];gm=matched_masks(source[r['mask_uid']],mask_predictions,np.array(camera['native_size']))
   for (method,pred),mask_metrics in zip(predictions.items(),gm):
    error=(pred-targets)/scale;record=dict(ledger,method=method,position_nrmse=float(np.sqrt(np.mean(error[:,:2]**2))),shape_nrmse=float(np.sqrt(np.mean(error[compatible,2:-1]**2))) if compatible.any() else None,scale_nrmse=float(np.sqrt(np.mean(error[:,-1]**2))),shape_compatible_targets=int(compatible.sum()),targets=len(ids),target_source_kind=r['source_kind'],target_mode_changed=not bool(compatible[target_at]),mask_iou=mask_metrics['iou'],mask_boundary_fscore_2_annotation_px=mask_metrics['boundary_fscore_2px'],observation_mode='retrospective_source_contours',**{k:None for k in ['independent_velocity_error','contact_accuracy']})
    if method=='refit_cubic':
     variance=np.maximum(np.einsum('ni,dij,nj->nd',phi,f['covariance'],phi)*np.asarray(selected['multipliers']),1e-15);raw_error=pred-targets;record['source_position_marginal95_coverage']=float(np.mean(abs(raw_error[:,:2])<=1.9599639845*np.sqrt(variance[:,:2])));record['source_position_diagonal_nll']=float(np.mean(.5*(np.log(2*np.pi*variance[:,:2])+raw_error[:,:2]**2/variance[:,:2])))
    else:record.update(source_position_marginal95_coverage=None,source_position_diagonal_nll=None)
    records.append(record)
 save(REPORT/f'temporal_quality.{v}.m{n}.parquet',records);save(REPORT/f'temporal_attempts.{v}.m{n}.parquet',attempts);return dict(records=records,attempts=attempts)

def run(monitor):
 spatial=[r for records in pool(spatial_worker,VIDEOS,monitor,workers=4) for r in records];save(REPORT/'spatial_quality.parquet',spatial);jobs=pool(temporal_worker,[(v,n) for v in VIDEOS for n in [16,32]],monitor,workers=4);temporal=[r for job in jobs for r in job['records']];attempts=[r for job in jobs for r in job['attempts']];save(REPORT/'temporal_quality.parquet',temporal);save(REPORT/'temporal_attempts.parquet',attempts)
 summary=[]
 for part in ['train','validation']:
  for n in [16,32]:
   for protocol in ['single','gap100','endpoint100']:
    methods=sorted({r['method'] for r in temporal})
    for method in methods:
     rs=[r for r in temporal if (r['split'],r['model'],r['protocol'],r['method'])==(part,n,protocol,method)];byvideo=[]
     for v in sorted({r['video_id'] for r in rs}):
      local=[r for r in rs if r['video_id']==v];byvideo.append({key:float(np.mean([r[key] for r in local])) for key in ['position_nrmse','scale_nrmse','mask_iou']})
     if byvideo:summary.append(dict(split=part,model=n,protocol=protocol,method=method,probes=len(rs),**{k:float(np.mean([r[k] for r in byvideo])) for k in byvideo[0]}))
 spatial_summary=[]
 for part in ['train','validation']:
  for method in ['frozen_v2_16','refit16','refit32']:
   rs=[r for r in spatial if r['split']==part and r['method']==method]
   spatial_summary.append(dict(split=part,method=method,masks=len(rs),iou=float(np.mean([r['iou'] for r in rs])),boundary_fscore_2px=float(np.mean([r['boundary_fscore_2px'] for r in rs]))))
 uncertainty=[]
 for part in ['train','validation']:
  for n in [16,32]:
   rs=[r for r in temporal if r['split']==part and r['model']==n and r['method']=='refit_cubic']
   if rs:uncertainty.append(dict(split=part,model=n,source_position_marginal95_coverage=float(np.mean([r['source_position_marginal95_coverage'] for r in rs])),scope='empirical source-position marginals; not joint or independent human uncertainty'))
 atomic_json(REPORT/'evaluation.json',dict(spatial=spatial_summary,temporal=summary,uncertainty=uncertainty,aggregation='temporal means of per-video probe means; spatial pooled paired manual masks',train_only_parameter_selection=True,validation_exploratory=True,independent_references=False,code=code()))
