"""Fixed-cubic channel-family refinement with separate validation confirmation tracks."""
from collections import defaultdict
import json,time
import numpy as np
from pipeline.common import ROOT,atomic_json
from track1.artifacts import save_index,save_shard,load_shard
from track1.numerics import basis,huber_weights
from refinement.data import temporal_cases,holdout
from refinement.protocol import rank
from .common import config,path,allowed,split,report,LOG,pool,identity

GROUPS={'position':slice(0,2),'coarse_shape':slice(2,18),'fine_shape':slice(18,34),'scale':slice(34,35)}
IMPORTANCE={'position':.35,'coarse_shape':.30,'fine_shape':.20,'scale':.15}
_CASES=None;_SCALE=None;_BASE=None;_KAPPA=None

def local_linear(t,h,weights,variance,delta=None):
 """Endpoint-centred slope prior; optional Huber weights use observed noise."""
 design=np.c_[np.ones(len(t)),np.asarray(t)-1.];weights=np.asarray(weights,float);robust=np.ones(len(t))
 for _ in range(1 if delta is None else 4):
  root=np.sqrt(weights*robust);local=np.linalg.lstsq(design*root[:,None],h*root[:,None],rcond=None)[0]
  if delta is not None:
   residual=(h-design@local)[:,:2];sigma=np.maximum(1.4826*np.median(abs(residual-np.median(residual,axis=0)),axis=0),np.maximum(np.sqrt(np.median(variance[:,:2],axis=0)),1e-6));distance=np.linalg.norm(residual/sigma,axis=1);robust=np.minimum(1.,delta/np.maximum(distance,1e-12))
 return local

def fit(t,h,variance,prior,quality,options,feature_scale):
 t=np.asarray(t,float);h=np.asarray(h,float);n,d=h.shape;phi=basis(2*t-1)
 qtime=np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]]) if n>1 else np.ones(1)
 effective=np.maximum(variance,1e-12)+options.get('floor',.01)*feature_scale[None]**2
 clip=options.get('variance_clip',100)
 if clip:
  middle=np.median(effective,axis=0);effective=np.clip(effective,middle/clip,middle*clip)
 power=np.full(d,options.get('variance_power',.25));regscale=np.full(d,options.get('regularization',1/3));recency=np.full(d,options.get('recency',0.))
 for group,override in options.get('groups',{}).items():
  sl=GROUPS[group];power[sl]=override.get('variance_power',power[sl]);regscale[sl]=override.get('regularization',regscale[sl]);recency[sl]=override.get('recency',recency[sl])
 base=qtime[:,None]*prior[:,None]**options.get('provenance_power',.5)*quality[:,None]/effective**power[None]
 base*=np.exp(np.clip((t[:,None]-1)*recency[None],-40,40));base*=n/np.maximum(base.sum(0),1e-20)
 w=base.copy();mode=np.asarray(options.get('mode_penalty',[0,0,1,9]));reg=np.eye(4)[None]*(regscale[:,None,None]*mode[None,None,:]+1e-8)
 prior_rhs=np.zeros((d,4));slope_strength=options.get('slope_strength',0.);value_strength=options.get('endpoint_value_strength',0.)
 if slope_strength or value_strength:
  recent=t>=t[-1]-options.get('slope_window',.25)
  if recent.sum()<min(4,n):recent=np.arange(n)>=max(0,n-4)
  local=local_linear(t[recent],h[recent],np.median(base[recent,:2],axis=1),variance[recent],options.get('slope_robust_delta'));d1=basis(1.,1)*2;value=basis(1.);mask=np.zeros(d)
  for group in options.get('slope_groups',['position']):mask[GROUPS[group]]=1.
  reg+=mask[:,None,None]*n*(slope_strength*np.outer(d1,d1)+value_strength*np.outer(value,value));prior_rhs+=mask[:,None]*n*(slope_strength*local[1,:,None]*d1[None]+value_strength*local[0,:,None]*value[None])

 for _ in range(options.get('irls',3)):
  gram=np.einsum('ni,nd,nj->dij',phi,w,phi)+reg;rhs=np.einsum('ni,nd,nd->di',phi,w,h)+prior_rhs;chol=np.linalg.cholesky(gram);c=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,rhs[...,None]))[...,0].T;residual=h-phi@c
  if options.get('loss','l2')=='l2':break
  r=residual/np.sqrt(effective+options.get('robust_floor',.1)*feature_scale[None]**2)
  robust=huber_weights(r,options.get('delta',3)) if options['loss']=='huber' else 1/(1+r*r);w=base*robust
 cov=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,np.broadcast_to(np.eye(4),(d,4,4))))
 sigma=np.maximum(np.sum(w*residual**2,axis=0)/max(n-4,1),1e-10);cov*=sigma[:,None,None]
 return dict(coefficients=c,covariance=cov,weights=w,condition=float(np.linalg.cond(gram).max()),neff=float(np.min(w.sum(0)**2/np.maximum(np.sum(w*w,axis=0),1e-30))))

def partition(case):
 if case['split']=='train':return 'train'
 return 'validation_dev' if int(rank(case['entity_track_id'])[:8],16)%2==0 else 'validation_confirmation'

def init_worker():
 global _CASES,_SCALE,_BASE,_KAPPA
 cfg=config();_CASES=temporal_cases(allowed(cfg['train']+cfg['validation']));sel=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());_SCALE=np.asarray(sel['feature_scale']);_BASE=sel['temporal']['options'];_KAPPA=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text())['kappa_cov']

def evaluate(options,cases=None,details=False):
 global _CASES
 if _CASES is None:init_worker()
 cases=_CASES if cases is None else cases;records=[]
 for case in cases:
  if options.get('screen') and int(rank(case['entity_track_id'])[:8],16)%3:continue
  if partition(case)=='validation_confirmation' and not details:continue
  t=case['time'];h=case['h'];centre=[.25,.5,.75][int(rank(f"{case['entity_track_id']}:{case['window_end_us']}")[:8],16)%3]
  for protocol in ['single','gap100','endpoint100']:
   hidden=holdout(t,protocol,centre);keep=~hidden
   if not hidden.any() or keep.sum()<8:continue
   result=fit(t[keep],h[keep],case['variance'][keep]*_KAPPA,case['prior'][keep],case['quality'][keep],options,_SCALE);pred=basis(2*t[hidden]-1)@result['coefficients'];error=pred-h[hidden]
   for group,sl in GROUPS.items():
    records.append(dict(video_id=case['video_id'],partition=partition(case),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],protocol=protocol,group=group,mse=float(np.mean(error[:,sl]**2)),normalised_mse=float(np.mean((error[:,sl]/_SCALE[sl])**2)),neff=result['neff'],condition=result['condition']))
 grouped=defaultdict(list)
 for r in records:grouped[(r['partition'],r['video_id'],r['protocol'],r['group'])].append(r)
 summaries={}
 for part in ['train','validation_dev','validation_confirmation']:
  summaries[part]={}
  for protocol in ['single','gap100','endpoint100']:
   summaries[part][protocol]={}
   for group in GROUPS:
    byvideo={v:dict(n=len(rs),normalised_rmse=float(np.sqrt(np.mean([r['normalised_mse'] for r in rs]))),rmse=float(np.sqrt(np.mean([r['mse'] for r in rs])))) for (p,v,pr,g),rs in grouped.items() if (p,pr,g)==(part,protocol,group)}
    if byvideo:summaries[part][protocol][group]=dict(normalised_rmse=float(np.mean([r['normalised_rmse'] for r in byvideo.values()])),rmse=float(np.mean([r['rmse'] for r in byvideo.values()])),by_video=byvideo)
 score=sum(IMPORTANCE[g]*sum(summaries['validation_dev'][p][g]['normalised_rmse']*(2 if p=='endpoint100' else 1) for p in ['single','gap100','endpoint100'])/4 for g in GROUPS)
 out=dict(options=options,objective=score,metrics=summaries)
 if details:out['records']=records
 return out

def trial(options):
 if _CASES is None:init_worker()
 return evaluate(options)

def search(monitor):
 from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
 import multiprocessing
 selected=json.loads((ROOT/'reports/track1_refinement/selection_locked.json').read_text());base=selected['temporal']['options'];candidates=[dict(base,screen=True)]
 # Focus on family-specific regularisation and endpoint emphasis; exact budget preserved.
 for strength in [.003,.03,.1,1/3,1.,3.]:
  for recent in [0.,1.,3.,6.]:
   for vp in [0.,.25]:
    groups={g:dict(regularization=strength,recency=recent,variance_power=vp) for g in ['position','coarse_shape','scale']}
    candidates.append(dict(base,groups=groups,screen=True))
 for loss in ['huber','cauchy']:
  for delta in [2.,5.]:
   candidates.append(dict(base,loss=loss,delta=delta,robust_floor=.1,irls=3,screen=True))
 results=[];ex=ProcessPoolExecutor(max_workers=config()['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init_worker);jobs={ex.submit(trial,o):i for i,o in enumerate(candidates)}
 try:
  while jobs:
   done,_=wait(jobs,timeout=5,return_when=FIRST_COMPLETED);monitor.check()
   for future in done:
    i=jobs.pop(future);r=future.result();results.append(r);LOG.info('Temporal screen %d objective %.6f',i,r['objective'])
 except BaseException:
  for proc in ex._processes.values():proc.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 ex.shutdown();init_worker()
 # Rank on development tracks only, then resolve per-family choices on the full dev pool.
 top=sorted(results,key=lambda r:r['objective'])[:5];options=[dict(base)]
 for r in top:options.append({k:v for k,v in r['options'].items() if k!='screen'})
 evaluated=[evaluate(o) for o in options];family_best={}
 for g in GROUPS:
  best=min(evaluated,key=lambda r:sum(r['metrics']['validation_dev'][p][g]['normalised_rmse']*(2 if p=='endpoint100' else 1) for p in ['single','gap100','endpoint100']))
  family_best[g]=best['options'].get('groups',{}).get(g,dict(regularization=best['options']['regularization'],recency=best['options'].get('recency',0.),variance_power=best['options']['variance_power']))
 mixed=dict(base,groups=family_best);evaluated.append(evaluate(mixed));best=min(evaluated,key=lambda r:r['objective'])
 # Confirmation is exposed once, after the candidate is fixed; it does not select among options.
 baseline=evaluate(base,details=True);candidate=evaluate(best['options'],details=True)
 for name,r in [('baseline',baseline),('candidate',candidate)]:save_index(path('reports')/f'temporal_{name}_cases.parquet',r.pop('records'))
 report('temporal_search.json',dict(screen=results,development=evaluated,baseline=baseline,candidate=candidate,confirmation_policy='Track-disjoint halves within the two validation participants; candidate locked on dev before confirmation. This is not an external unseen-participant test.',channel_groups={k:[v.start,v.stop] for k,v in GROUPS.items()},objective_importance=IMPORTANCE,**identity()))
 report('temporal_candidate.json',dict(options=best['options'],state='experimental_not_finalized',objective=best['objective'],**identity()))
 return candidate

def breakdown(monitor):
 init_worker();records=[]
 for case in _CASES:
  t=case['time'];h=case['h'];res=fit(t,h,case['variance']*_KAPPA,case['prior'],case['quality'],_BASE,_SCALE);phi=basis(2*t-1);error=phi@res['coefficients']-h;ss=np.sum((h-h.mean(0))**2,axis=0);speed=np.sqrt(np.mean(np.sum((np.diff(h[:,:2],axis=0)/np.diff(t)[:,None])**2,axis=1)));regime='stationary_proxy' if speed<.02 else 'fast_proxy' if speed>.5 else 'moderate_proxy'
  for group,sl in GROUPS.items():
   mask=ss[sl]>len(t)*_SCALE[sl]**2*1e-4;r2=1-np.sum(error[:,sl]**2,axis=0)[mask]/ss[sl][mask];ls=np.linalg.lstsq(phi,h[:,sl],rcond=None)[0]
   records.append(dict(video_id=case['video_id'],partition=partition(case),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],group=group,motion_regime=regime,normalised_rmse=float(np.sqrt(np.mean((error[:,sl]/_SCALE[sl])**2))),median_r2=float(np.median(r2)) if len(r2) else None,nonconstant_channels=int(mask.sum()),cubic_lower_bound_normalised_rmse=float(np.sqrt(np.mean(((phi@ls-h[:,sl])/_SCALE[sl])**2)))))
 save_index(path('reports')/'channel_breakdown.parquet',records);summary={}
 for g in GROUPS:
  rs=[r for r in records if r['group']==g and r['median_r2'] is not None];summary[g]=dict(median_r2=float(np.median([r['median_r2'] for r in rs])),normalised_rmse=float(np.sqrt(np.mean([r['normalised_rmse']**2 for r in rs]))))
 report('channel_breakdown.json',dict(groups=summary,motion_regime_note='Observed intrinsic-2D speed thresholds are diagnostic proxies, not annotated motion or static-world labels.',test_videos_used=False));monitor.check();return summary


def balanced_search(monitor):
 init_worker();previous=json.loads((path('reports')/'temporal_search.json').read_text());baseline=evaluate(_BASE);screen_base=next(r for r in previous['screen'] if not r['options'].get('groups') and r['options'].get('loss')=='l2');possible=[]
 for r in previous['screen']:
  if all(r['metrics']['validation_dev'][p][g]['normalised_rmse']<=1.15*screen_base['metrics']['validation_dev'][p][g]['normalised_rmse'] for p in ['single','gap100'] for g in GROUPS):possible.append(r)
 options=[{k:v for k,v in r['options'].items() if k!='screen'} for r in sorted(possible,key=lambda r:r['objective'])[:10]];trials=[baseline]
 for o in options:
  r=evaluate(o);trials.append(r);LOG.info('Balanced temporal dev objective %.6f',r['objective']);monitor.check()
 groups={}
 for g in GROUPS:
  eligible=[r for r in trials if all(r['metrics']['validation_dev'][p][g]['normalised_rmse']<=1.05*baseline['metrics']['validation_dev'][p][g]['normalised_rmse'] for p in ['single','gap100'])]
  best=min(eligible,key=lambda r:r['metrics']['validation_dev']['endpoint100'][g]['normalised_rmse']);groups[g]=best['options'].get('groups',{}).get(g,dict(regularization=best['options']['regularization'],variance_power=best['options']['variance_power'],recency=best['options'].get('recency',0.)))
 option=dict(_BASE,groups=groups);candidate=evaluate(option)
 report('temporal_balanced.json',dict(candidate=candidate,baseline=baseline,trials=trials,guard='At most 5% relative dev interior error regression per channel family; endpoint objective selects among eligible fits.',confirmation_policy='This additional exploratory variant uses validation-dev only; confirmation has already been exposed in the endpoint experiment and is not repeatedly tested here.',state='experimental_not_finalized',**identity()));return candidate
