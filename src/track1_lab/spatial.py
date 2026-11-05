"""Symmetric contour fitting experiments at the unchanged 16-control budget."""
from collections import defaultdict
import json
import numpy as np
from shapely.geometry import LinearRing
from pipeline.common import ROOT
from track1.artifacts import save_shard,save_index,load_shard
from track1.numerics import weighted_solve,huber_weights
from refinement.kernels import spatial_fit,spatial_errors,nearest_curve,periodic_design
from refinement.data import spatial_cases
from refinement.protocol import rank
from .common import config,path,allowed,report,pool,LOG,identity

_BASE=None;_CASES=None

def closest_polyline(query,vertices):
 a=np.asarray(vertices,float);ab=np.roll(a,-1,axis=0)-a;delta=query[:,None]-a[None];u=np.clip(np.sum(delta*ab,axis=2)/np.maximum(np.sum(ab*ab,axis=1),1e-20),0,1);p=a[None]+u[:,:,None]*ab[None];d=np.sum((query[:,None]-p)**2,axis=2);j=d.argmin(1);return p[np.arange(len(query)),j],np.sqrt(d[np.arange(len(query)),j])

def fit(y,options,sample_weights=None,reliability=1.):
 global _BASE
 if _BASE is None:_BASE=json.loads((ROOT/'reports/track1_refinement/spatial_selection.json').read_text())['options']
 y=np.asarray(y,float);weights=np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights);initial=spatial_fit(y,_BASE,reliability,weights)
 if options.get('baseline'):return initial
 p=initial['controls'].copy();p0=p.copy();base=weights*reliability;observed=y[weights>0];u_grid=np.arange(128)/128;design_reverse=periodic_design(u_grid);d2=np.roll(np.eye(16),1,axis=0)-2*np.eye(16)+np.roll(np.eye(16),-1,axis=0);best=initial;bestscore=np.inf
 def score(c):
  f,_=nearest_curve(observed,c);closest,r=closest_polyline(design_reverse@c,observed);return float(np.mean(f*f)+options['reverse']*np.mean(r*r)),closest
 bestscore,reverse_target=score(p)
 for step in range(options.get('steps',12)):
  _,u=nearest_curve(y,p);forward=periodic_design(u);design=np.vstack([forward,design_reverse,np.eye(16)]);targets=np.vstack([y,reverse_target,p0]);residual=np.linalg.norm(forward@p-y,axis=1);robust=np.ones(len(y)) if options.get('loss','huber')=='l2' else huber_weights(residual/max(float(np.median(residual[weights>0])/.67449),1.),2.)
  forward_w=base*robust;reverse_w=np.full(128,options['reverse']*forward_w.sum()/128);anchor=np.full(16,options.get('anchor',.0001)*forward_w.sum()/16);w=np.r_[forward_w,reverse_w,anchor];gram=design.T@(w[:,None]*design);prior=options.get('curvature',.0001)*np.trace(gram)/np.trace(d2.T@d2)*(d2.T@d2)+np.eye(16)*1e-6
  pp,cov,condition,solver=weighted_solve(design,targets,w,prior);simple=bool(LinearRing(periodic_design(np.arange(512)/512)@pp).is_simple);newscore,next_reverse_target=score(pp)
  if not simple or newscore>bestscore*(1+1e-8):break
  error=targets-design@pp;sigma2=max(float(np.sum(w*np.sum(error**2,axis=1))/(2*max(w.sum()-16,1))),1.);best=dict(controls=pp,covariance=cov*sigma2,valid=True,score=newscore,parameters=u,condition=condition,solver=solver,iterations=step+1);bestscore=newscore
  if np.max(abs(pp-p))<1e-4:break
  p=pp;reverse_target=next_reverse_target
 return best


def cases():
 global _CASES
 if _CASES is None:
  cfg=config();_CASES=[r for r in spatial_cases(allowed(cfg['train']+cfg['validation'])) if r['source_kind']=='manual_sparse']
 return _CASES

def partition(c):return 'train' if c['split']=='train' else ('validation_dev' if int(rank(c['mask_uid'])[:8],16)%2==0 else 'validation_confirmation')

def evaluate(options,details=False):
 records=[]
 for c in cases():
  if partition(c)=='validation_confirmation' and not details:continue
  if options.get('screen') and int(rank(c['mask_uid'])[:8],16)%3:continue
  y=c['y'];mask=np.ones(128);mask[::4]=0;result=fit(y,options,mask);error,var,d=spatial_errors(y[::8],result);curve=periodic_design(np.arange(128)/128)@result['controls'];_,reverse=closest_polyline(curve,y);scale=max(float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1)))),1.)
  records.append(dict(video_id=c['video_id'],partition=partition(c),mask_uid=c['mask_uid'],rmse=float(np.sqrt(np.mean(error**2))),normalised_rmse=float(np.sqrt(np.mean(error**2))/scale),reverse_rmse=float(np.sqrt(np.mean(reverse**2))),normalised_reverse_rmse=float(np.sqrt(np.mean(reverse**2))/scale),valid=result['valid']))
 summary={}
 for part in ['train','validation_dev','validation_confirmation']:
  byvideo={}
  for v in config()['train']+config()['validation']:
   rs=[r for r in records if r['video_id']==v and r['partition']==part]
   if rs:byvideo[v]={k:float(np.mean([r[k] for r in rs])) for k in ['rmse','normalised_rmse','reverse_rmse','normalised_reverse_rmse','valid']}
  if byvideo:summary[part]={**{k:float(np.mean([r[k] for r in byvideo.values()])) for k in ['rmse','normalised_rmse','reverse_rmse','normalised_reverse_rmse','valid']},'by_video':byvideo}
 v=summary['validation_dev'];objective=(v['normalised_rmse']+.35*v['normalised_reverse_rmse'])*(1+10*(1-v['valid']))
 result=dict(options=options,objective=objective,metrics=summary)
 if details:result['records']=records
 return result

def trial(options):return evaluate(options)

def search(monitor):
 options=[dict(baseline=True,screen=True)]
 for reverse in [0.,.25,1.]:
  for steps in [6,16]:
   for curvature in [.00001,.0001]:
    for loss in ['l2','huber']:options.append(dict(reverse=reverse,steps=steps,curvature=curvature,loss=loss,screen=True))
 results=pool(trial,options,monitor);best=sorted(results,key=lambda r:r['objective'])[:4];full=[evaluate(dict(baseline=True))]+[evaluate({k:v for k,v in r['options'].items() if k!='screen'}) for r in best if not r['options'].get('baseline')];chosen=min(full,key=lambda r:r['objective']);baseline=evaluate(dict(baseline=True),True);candidate=evaluate(chosen['options'],True)
 for name,r in [('baseline',baseline),('candidate',candidate)]:save_index(path('reports')/f'spatial_{name}_cases.parquet',r.pop('records'))
 report('spatial_search.json',dict(screen=results,development=full,baseline=baseline,candidate=candidate,selection_policy='Symmetric distance objective and invalid-fit penalty, manual train/dev masks only. Confirmation masks exposed once after selection.',**identity()))
 report('spatial_candidate.json',dict(options=chosen['options'],state='experimental_not_finalized',**identity()))
 # Full-source candidate previews retain source UIDs and original row indices.
 for v in config()['train']+config()['validation']:
  rows=[];arrays=defaultdict(list);old=load_shard(ROOT/f'derived/track1_refined_v2/spatial/{v}.pt');meta={r['mask_uid']:r for r in __import__('pyarrow.parquet',fromlist=['read_table']).read_table(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').to_pylist()}
  for c in [r for r in cases() if r['video_id']==v]:
   result=fit(c['y'],chosen['options'],reliability=c['geometry_weight']*c['provenance_weight']);r=meta[c['mask_uid']];i=r['row_index'];a,_=nearest_curve(c['y'],old['controls'][i]);b,_=nearest_curve(c['y'],result['controls']);fallback=bool(r['spatial_valid'] and (not result['valid'] or np.mean(b*b)>np.mean(a*a)))
   if fallback:result['controls']=old['controls'][i];result['covariance']=old['covariance'][i]
   arrays['controls'].append(result['controls']);arrays['covariance'].append(result['covariance']);rows.append(dict(video_id=v,mask_uid=c['mask_uid'],original_row_index=i,row_index=len(rows),valid=bool(r['spatial_valid'] or result['valid']),fallback=fallback))
  out=path('derived');save_shard(out/f'spatial_preview.{v}.pt',{k:np.stack(a).astype(np.float32) for k,a in arrays.items()});save_index(out/f'spatial_preview.{v}.parquet',rows);monitor.check()
 return candidate
