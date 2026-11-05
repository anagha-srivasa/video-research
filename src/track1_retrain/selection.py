"""Training-only family safeguards and mask-space geometry refinement."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from track1.numerics import basis
from .common import *
from .temporal import load,proposals,timing_eligible,holdout,prepare,fit,search_worker,summarize
from .evaluation import curves,matched_masks

def geometry_worker(job):
 v,n,options,scale=job;rows,data=load(v,n);cases=sorted([c for c in proposals(v,rows) if timing_eligible(c,rows)],key=lambda c:rank(c['entity_track_id']+str(c['window_end_us'])))[:64];source={r['mask_uid']:r for r in pq.read_table(WORK/f'inputs.{v}.parquet').to_pylist()};camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());values=[]
 for case in cases:
  for protocol in ['single','gap100','endpoint100']:
   t,hidden=holdout(case,rows,protocol)
   if not hidden.any():continue
   try:p=prepare(case,rows,data,n,hidden)
   except ValueError:continue
   f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,np.asarray(scale));targets=np.flatnonzero(hidden);at=int(rank(case['entity_track_id']+str(case['window_end_us'])+protocol)[:8],16)%len(targets);target=targets[at];r=rows[case['indices'][target]];h=basis(2*t[target]-1)@f['coefficients'];prediction=(curves(h,p['allocation'],camera),p['roles'],p['parents']);metric=matched_masks(source[r['mask_uid']],[prediction],np.array(camera['native_size']))[0];values.append(dict(protocol=protocol,iou=metric['iou'],video_id=v,model=n,entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us']))
 return dict(video_id=v,model=n,options=options,probes=len(values),mean_iou=float(np.mean([r['iou'] for r in values])) if values else None,records=values)

def component_means(job,key):
 out={}
 for protocol in ['single','gap100','endpoint100']:
  rs=[r[key] for r in job['records'] if r['protocol']==protocol and r[key] is not None]
  if rs:out[protocol]=float(np.mean(rs))
 return out

def run(monitor):
 from .temporal import search_worker
 model=json.loads((folder('calibration')/'model.json').read_text());baseline=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text())['temporal']['options'];ledger=[];safeguards=[]
 for n in [16,32]:
  selected=model[str(n)];scale=selected['feature_scale'];options=dict(selected['options']);options['families']={k:dict(v) for k,v in options['families'].items()}
  old=pool(search_worker,[(v,n,baseline,scale,None) for v in CFG['train']],monitor,workers=5);current=pool(search_worker,[(v,n,options,scale,None) for v in CFG['train']],monitor,workers=5);old={r['video_id']:r for r in old};current={r['video_id']:r for r in current}
  for family,key in [('position','position_mse'),('scale','scale_mse')]:
   violations=[]
   for v in CFG['train']:
    a=component_means(old[v],key);b=component_means(current[v],key)
    for protocol in a.keys()&b.keys():
     if b[protocol]>a[protocol]*1.02+1e-12:violations.append(dict(video_id=v,protocol=protocol,baseline=a[protocol],candidate=b[protocol]))
   if violations:options['families'][family]=dict(regularization=baseline['regularization'],loss=baseline['loss'])
   safeguards.append(dict(model=n,family=family,selected=options['families'][family],train_regret_limit_mse_fraction=.02,violations=violations))
  trials=[]
  for strength in [.003,.03,1/3,3.,10.]:
   for loss in ['l2','huber']:
    candidate=dict(options,families={k:dict(v) for k,v in options['families'].items()});candidate['families']['shape']=dict(regularization=strength,loss=loss);jobs=pool(geometry_worker,[(v,n,candidate,scale) for v in CFG['train']],monitor,workers=5);means=[j['mean_iou'] for j in jobs if j['mean_iou'] is not None];score=float(np.mean(means)) if means else -1.;trials.append((score,candidate,jobs));ledger.append(dict(model=n,shape_strength=strength,shape_loss=loss,train_mean_video_mask_iou=score,probes=sum(j['probes'] for j in jobs)));LOG.info('Training mask objective M%d shape=%s/%s IoU=%.6f',n,strength,loss,score)
  score,options,jobs=max(trials,key=lambda x:x[0]);selected['options']=options;selected['geometry_train_iou']=score
  # Recalibrate only the raw functional posterior for its new mean/family fit.
  raw=pool(search_worker,[(v,n,options,scale,None) for v in CFG['train']],monitor,workers=5);qs=[]
  for protocol in ['all','endpoint100']:
   a=[r for j in raw for p,r in j['ratios'] if protocol=='all' or p==protocol];ratio=np.concatenate(a);qs.append(np.maximum(np.nanquantile(ratio,.95,axis=0)/1.9599639845**2,1e-4))
  multiplier=np.nan_to_num(np.maximum(*qs),nan=1.,posinf=1e6);selected['multipliers']=multiplier.tolist();array(folder('calibration')/f'm{n}.multipliers.npy',multiplier)
  # Refine the optional position-only detail using the safeguarded coarse fit.
  def position_score(values,protocol=None):
   means=[]
   for j in values:
    rs=[r['position_mse'] for r in j['records'] if protocol is None or r['protocol']==protocol]
    if rs:means.append(float(np.mean(rs)))
   return float(np.mean(means))
  base=position_score(raw);endpoint=position_score(raw,'endpoint100');detail_trials=[(base,dict(count=0,strength=0.))]
  for count in [2,3]:
   for strength in [.01,.1]:
    detail=dict(count=count,strength=strength);values=pool(search_worker,[(v,n,options,scale,detail) for v in CFG['train']],monitor,workers=5)
    if position_score(values,'endpoint100')<=endpoint*1.02:detail_trials.append((position_score(values),detail))
  best,detail=min(detail_trials,key=lambda x:x[0]);selected['detail']=detail if best<=base*.998 else dict(count=0,strength=0.);selected['position_train_objective']=base;selected['detail_position_train_objective']=best;selected['selection_objective']='training mask-space IoU for shape; position/scale per-participant/protocol guard; position-only neutral detail';LOG.info('Selected physical reconstruction M%d %s detail=%s',n,options['families'],selected['detail'])
 atomic_json(folder('calibration')/'model.json',model);atomic_json(REPORT/'geometry_selection.json',dict(trials=ledger,safeguards=safeguards,selected=model,numeric_parameter_selection='training participants only',validation_exposed_before_objective_revision=True,validation_is_exploratory=True,shape_sampling='64 timing-eligible hashed cases per training video, same across candidate options',code=code()))
