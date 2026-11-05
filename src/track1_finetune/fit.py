"""Participant-balanced training search and matched withheld-observation evidence."""
import collections
import copy
import json
import numpy as np
from scipy.interpolate import interp1d
from track1.numerics import basis
from track1_retrain.temporal import prepare,holdout,target_h,fit,fit_detail
from track1_lab.history_detail import evaluate as detail_evaluate
from track1_retrain.evaluation import curves,matched_masks,source_rows
from .data import *

FAMILIES={'position':slice(0,2),'shape':slice(2,-1),'scale':slice(-1,None)}

def candidates(base):
 out=[copy.deepcopy(base)]
 for loss in ['l2','huber']:
  for strength in [.1,1/3,1.,3.]:
   c=copy.deepcopy(base);c['families']={name:dict(loss=loss,regularization=strength) for name in FAMILIES};out.append(c)
 for field,value in [('variance_power',0.),('variance_power',.5),('provenance_power',1.),('delta',1.5),('delta',6.)]:
  c=copy.deepcopy(base);c[field]=value;out.append(c)
 return out

def probe_data(v,n,limit):
 cases,inputs=cases_for(v,n,limit);probes=[];attempts=[]
 for case in cases:
  rows,data=inputs[case['source']]
  for protocol in ['single','gap100','endpoint100']:
   times,hidden=holdout(case,rows,protocol);identity=dict(video_id=v,split=split(v),source=case['source'],model=n,entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol)
   if not hidden.any():attempts.append(dict(identity,supported=False,reason='no_target'));continue
   try:p=prepare(case,rows,data,n,hidden)
   except ValueError as e:attempts.append(dict(identity,supported=False,reason=str(e)));continue
   target_ids=np.asarray(case['indices'])[hidden];targets,compatible=target_h(rows,data,target_ids,n,p)
   at=int(rank((case['entity_track_id'],case['window_end_us'],protocol))[:8],16)%len(target_ids)
   target=rows[target_ids[at]]
   probes.append(dict(identity=identity,p=p,times=times[hidden],targets=targets,compatible=compatible,target_at=at,target_uid=target['mask_uid'],target_kind=target['source_kind'],target_width=target['width'],target_height=target['height'],source_dependencies=[rows[i]['dependency_group'] for i in target_ids]))
   attempts.append(dict(identity,supported=True,reason=None))
 return probes,attempts

def predict(probe,options,scale,detail=None):
 p=probe['p'];f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,np.asarray(scale));phi=basis(2*probe['times']-1)
 prediction=phi@f['coefficients']
 if detail and detail['count']:
  dm=fit_detail(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,np.asarray(scale),count=detail['count'],strength=detail['strength']);prediction=detail_evaluate(dm,probe['times'])
 if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
 return prediction,f

def errors(prediction,probe,scale):
 e=(prediction-probe['targets'])/np.asarray(scale);result={}
 for name,sl in FAMILIES.items():
  selected=e[probe['compatible'],sl] if name=='shape' else e[:,sl]
  result[name+'_mse']=float(np.mean(selected**2)) if selected.size else None
 return result

def search_worker(job):
 v,n,options,scale=job;probes,attempts=probe_data(v,n,CFG['cases_per_video']);records=[]
 for index,choice in enumerate(options):
  for probe in probes:
   try:pred,f=predict(probe,choice,scale);metrics=errors(pred,probe,scale)
   except (ValueError,np.linalg.LinAlgError):metrics={k+'_mse':float('inf') for k in FAMILIES}
   records.append(dict(probe['identity'],trial=index,**metrics))
 return dict(video_id=v,model=n,records=records,attempts=attempts)

def grouped_mean(records,key,trial=None):
 byvideo=collections.defaultdict(list)
 for r in records:
  if (trial is None or r.get('trial')==trial) and r.get(key) is not None:byvideo[r['video_id']].append(r[key])
 return float(np.mean([np.mean(x) for x in byvideo.values()])) if byvideo else float('inf')

def choose_family(records,family):
 key=family+'_mse';trials=sorted({r['trial'] for r in records});groups=collections.defaultdict(dict)
 for trial in trials:
  grouped=collections.defaultdict(list)
  for r in records:
   if r['trial']==trial and r[key] is not None:grouped[(r['video_id'],r['protocol'])].append(r[key])
  groups[trial]={k:float(np.mean(x)) for k,x in grouped.items()}
 valid=[0]
 for trial in trials[1:]:
  if groups[trial].keys()!=groups[0].keys():continue
  if all(np.isfinite(b) and b<=groups[0][k]*(1+CFG['maximum_training_family_regret'])+1e-12 for k,b in groups[trial].items()):valid.append(trial)
 best=min(valid,key=lambda i:grouped_mean(records,key,i));base=grouped_mean(records,key,0)
 if grouped_mean(records,key,best)>base*(1-CFG['minimum_relative_improvement']):best=0
 return best,dict(family=family,trial=best,baseline_mse=base,selected_mse=grouped_mean(records,key,best),admissible_trials=valid)

def mask_worker(job):
 v,n,choices,scale=job;probes,_=probe_data(v,n,CFG['mask_probes_per_video']);wanted={p['target_uid'] for p in probes};source=source_rows(v,wanted);camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());records=[]
 for probe in probes:
  predictions=[];failed=[]
  for index,choice in enumerate(choices):
   try:
    pred,_=predict(probe,choice,scale);ps=curves(pred[probe['target_at']],probe['p']['allocation'],camera)
    if not all(np.isfinite(a).all() for a in ps):raise ValueError('Nonfinite mask geometry')
    predictions.append((ps,probe['p']['roles'],probe['p']['parents']));failed.append(False)
   except (ValueError,np.linalg.LinAlgError):predictions.append(([np.zeros((3,2))],['outer'],[-1]));failed.append(True)
  metrics=matched_masks(source[probe['target_uid']],predictions,np.array(camera['native_size']))
  for index,metric in enumerate(metrics):records.append(dict(probe['identity'],trial=index,iou=0. if failed[index] else metric['iou'],failed=failed[index]))
 return records

def detail_worker(job):
 v,n,options,scale,details=job;probes,attempts=probe_data(v,n,CFG['cases_per_video']);records=[]
 for index,detail in enumerate(details):
  for p in probes:
   pred,_=predict(p,options,scale,detail);records.append(dict(p['identity'],trial=index,position_mse=errors(pred,p,scale)['position_mse']))
 return dict(video_id=v,model=n,records=records,attempts=attempts)

def calibration_worker(job):
 v,n,options,scale=job;probes,_=probe_data(v,n,CFG['cases_per_video']);ratios=collections.defaultdict(list)
 for p in probes:
  pred,f=predict(p,options,scale);phi=basis(2*p['times']-1);var=np.maximum(np.einsum('ni,dij,nj->nd',phi,f['covariance'],phi),1e-15);ratio=(pred-p['targets'])**2/var;ratio[~p['compatible'],2:-1]=np.nan;ratios[p['identity']['protocol']].append(ratio)
 # Keep equal-size deterministic samples per participant/protocol, avoiding domination by dense targets.
 return dict(video_id=v,ratios={pr:np.concatenate(values) for pr,values in ratios.items()})

def evaluate_worker(job):
 v,n,selected=job;probes,attempts=probe_data(v,n,CFG['evaluation_cases_per_video']);base=baseline()[str(n)];scale=np.array(base['feature_scale']);records=[]
 camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());source=source_rows(v,{p['target_uid'] for p in probes})
 for p in probes:
  predictions={};posteriors={}
  for name,choice,detail in [('prepared_baseline',base['options'],base['detail']),('finetuned_core',selected['options'],None),('finetuned_detail',selected['options'],selected['detail'])]:
   pred,f=predict(p,choice,scale,detail);predictions[name]=pred;posteriors[name]=f
  for method in ['nearest','linear']:predictions[method]=interp1d(p['p']['time'],p['p']['h'],axis=0,kind=method,fill_value='extrapolate')(p['times'])
  masks=[(curves(pred[p['target_at']],p['p']['allocation'],camera),p['p']['roles'],p['p']['parents']) for pred in predictions.values()]
  metrics=matched_masks(source[p['target_uid']],masks,np.array(camera['native_size']))
  for (name,pred),gm in zip(predictions.items(),metrics):
   r=dict(p['identity'],method=name,**errors(pred,p,scale),mask_iou=gm['iou'],boundary_fscore=gm['boundary_fscore_2px'],target_kind=p['target_kind'],targets=len(p['targets']),compatible_shape_targets=int(p['compatible'].sum()),source_dependency_groups=len(set(p['source_dependencies'])),causal_rgb_forecast=False,physical_velocity_reference=False)
   if name in ['prepared_baseline','finetuned_core']:
    f=posteriors[name];multiplier=base['multipliers'] if name=='prepared_baseline' else selected['multipliers'];phi=basis(2*p['times']-1);var=np.maximum(np.einsum('ni,dij,nj->nd',phi,f['covariance'],phi)*np.array(multiplier),1e-15);e=pred-p['targets'];r['position_marginal95_coverage']=float(np.mean(abs(e[:,:2])<=1.9599639845*np.sqrt(var[:,:2])));r['position_diagonal_nll']=float(np.mean(.5*(np.log(2*np.pi*var[:,:2])+e[:,:2]**2/var[:,:2])))
   else:r.update(position_marginal95_coverage=None,position_diagonal_nll=None)
   records.append(r)
 save(R/f'evaluation.{v}.m{n}.parquet',records);save(R/f'attempts.{v}.m{n}.parquet',attempts)
 return dict(records=records,attempts=attempts)
