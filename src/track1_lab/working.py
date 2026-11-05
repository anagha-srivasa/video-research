"""Fit a complete fixed-format working candidate without publishing a release."""
from collections import defaultdict
import copy,json,time
import numpy as np
import pyarrow.parquet as pq
import duckdb
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.numerics import basis
from track1.temporal import collocation
from refinement.data import temporal_cases,holdout
from refinement.protocol import rank
from refinement.diagnostics import reliability
from .common import config,path,allowed,split,report,pool,LOG,identity
from .temporal import fit
from .topology import validity,guard


def directory(stage):
 p=path('derived')/stage;p.mkdir(exist_ok=True);return p


def working_cases(videos):
 result=[]
 for v in allowed(videos):
  s=load_shard(directory('spatial')/f'{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet',columns=['entity_track_id','timestamp_us','row_index']).to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r['row_index'] for r in rows}
  for c in temporal_cases([v]):
   ids=[lookup[(c['entity_track_id'],int(c['window_end_us']-1000000+round(t*1e6)))] for t in c['time']];c['h']=s['h'][ids].astype(float);c['variance']=s['variance'][ids].astype(float);result.append(c)
 return result


def calibrate(monitor):
 cfg=config();values=[]
 for v in cfg['train']:
  s=load_shard(directory('spatial')/f'{v}.pt');values.append(s['h'][::3].astype(float))
 x=np.concatenate(values);centre=np.median(x,axis=0);scale=np.maximum(np.median(abs(x-centre),axis=0)*1.4826,.01);del x,values
 options=copy.deepcopy(json.loads((path('reports')/'temporal_balanced.json').read_text())['candidate']['options']);derivative=json.loads((path('reports')/'derivative_candidate.json').read_text())['candidate']['options'];options['groups']['position']=derivative['groups']['position']
 for key in ['slope_strength','slope_window','endpoint_value_strength','slope_groups']:options[key]=derivative[key]
 kappa=json.loads((path('reports')/'uncertainty_candidate.json').read_text())['global_kappa'];records=[]
 for v in cfg['train']+cfg['validation']:
  for c in working_cases([v]):
   t=c['time'];h=c['h'];ctr=[.25,.5,.75][int(rank(c['entity_track_id']+str(c['window_end_us']))[:8],16)%3]
   for protocol in ['single','gap100','endpoint100']:
    hidden=holdout(t,protocol,ctr);keep=~hidden
    if not hidden.any() or keep.sum()<8:continue
    f=fit(t[keep],h[keep],c['variance'][keep]*kappa,c['prior'][keep],c['quality'][keep],options,scale);guarded=guard(f['coefficients'],f['covariance'],h[keep]);phi=basis(2*t[hidden]-1)
    for name,result in [('un_guarded',f),('guarded',guarded)]:
     error=phi@result['coefficients']-h[hidden];variance=np.maximum(np.einsum('ni,dij,nj->nd',phi,result['covariance'],phi),1e-15);records.append(dict(split=c['split'],protocol=protocol,method=name,error=error,variance=variance))
  monitor.check();LOG.info('Working calibration cases %s',v)
 multipliers={};evaluations={}
 for method in ['un_guarded','guarded']:
  ratios=[]
  for protocol in ['all','endpoint100']:
   rs=[r for r in records if r['split']=='train' and r['method']==method and (protocol=='all' or r['protocol']==protocol)];e=np.concatenate([r['error'] for r in rs]);var=np.concatenate([r['variance'] for r in rs]);ratios.append(np.maximum(np.quantile(e*e/var,.95,axis=0)/1.9599639845**2,1e-4))
  multiplier=np.maximum(*ratios);multipliers[method]=multiplier.tolist()
  for part in ['train','validation']:
   for protocol in ['single','gap100','endpoint100']:
    rs=[r for r in records if (r['split'],r['method'],r['protocol'])==(part,method,protocol)];e=np.concatenate([r['error'] for r in rs]);var=np.concatenate([r['variance'] for r in rs])*multiplier;evaluations[part+'/'+method+'/'+protocol]=reliability(e,var)
 model=dict(options=options,feature_scale=scale.tolist(),centre=centre.tolist(),kappa_cov=kappa,temporal_multipliers=multipliers,evaluations=evaluations,training_videos=cfg['train'],state='working_not_finalized',note='Hybrid keeps balanced coarse/scale fitting and the train-selected position slope prior. Both unguarded and guarded forms are retained; neither is scientifically accepted.',**identity());atomic_json(directory('calibration')/'model.json',model);report('working_calibration.json',model);return model


def video(v):
 cfg=config();allowed([v]);tuning=split(v);memory_guard(cfg);sp=directory('spatial');s=load_shard(sp/f'{v}.pt');rows=pq.read_table(sp/f'{v}.parquet').to_pylist();model=json.loads((directory('calibration')/'model.json').read_text());options=model['options'];scale=np.asarray(model['feature_scale']);appearance=load_shard(ROOT/f'derived/track1_refined_v2/appearance/{v}.pt');tracks=defaultdict(list)
 for r in rows:
  if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
 with duckdb.connect() as db:duration=db.execute('SELECT duration_us FROM read_parquet(?)',[str(ROOT/f'canonical/v1/videos.{v}.parquet')]).fetchone()[0]
 arrays=defaultdict(list);meta=[];start=time.monotonic()
 for number,(track,rs) in enumerate(sorted(tracks.items())):
  if number%100==0:memory_guard(cfg)
  ids=np.array([r['row_index'] for r in rs]);times=np.array([r['timestamp_us'] for r in rs]);prior=np.array([r['provenance_weight'] for r in rs]);quality=np.array([r['geometry_weight'] for r in rs])
  for end in range(max(1000000,((int(times[0])+249999)//250000)*250000),min(int(times[-1])+1000000,duration)//250000*250000+1,250000):
   begin=end-1000000;lo=np.searchsorted(times,begin);hi=np.searchsorted(times,end,side='right')
   if hi==lo:continue
   local=ids[lo:hi];t=(times[lo:hi]-begin)/1e6;h=s['h'][local].astype(float);f=fit(t,h,s['variance'][local].astype(float)*model['kappa_cov'],prior[lo:hi],quality[lo:hi],options,scale);c=f['coefficients'];cov=f['covariance']*np.asarray(model['temporal_multipliers']['un_guarded'])[:,None,None];g=guard(c,f['covariance'],h);gcov=g['covariance']*np.asarray(model['temporal_multipliers']['guarded'])[:,None,None];nodes=collocation(c);gnodes=collocation(g['coefficients']);ell=np.stack([basis(1.,d)*2**d for d in range(3)]);endpoint=ell@c;gendpoint=ell@g['coefficients'];ac=appearance['valid'][local].astype(bool);app=appearance['embedding'][local][ac].astype(float).mean(0) if ac.any() else np.zeros(128);predicted=(basis(2*np.linspace(0,1,21)-1)@c)[:,34];scale_valid=bool(np.all((predicted>-15)&(predicted<1.5)))
   vals=dict(coefficients=c,coefficient_cov=cov,guarded_coefficients=g['coefficients'],guarded_coefficient_cov=gcov,endpoint=endpoint[0],endpoint_velocity=endpoint[1],endpoint_accel=endpoint[2],guarded_endpoint=gendpoint[0],guarded_endpoint_velocity=gendpoint[1],guarded_endpoint_accel=gendpoint[2],appearance=app,xi=nodes['xi'],xi_dot=nodes['xi_dot'],guarded_xi=gnodes['xi'],guarded_xi_dot=gnodes['xi_dot'])
   for key,value in vals.items():arrays[key].append(np.asarray(value,np.float16 if key=='appearance' else np.float32))
   maxgap=float(np.max(np.diff(np.r_[0,t,1])));span=float(t[-1]-t[0]);support=len(t)>=4 and f['neff']>=6 and span>=.6 and maxgap<=.35 and f['condition']<1e10;geometric=bool(validity(c).all());meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=track,window_start_us=begin,window_end_us=end,window_seconds=1.,hop_seconds=.25,valid=bool(support and geometric and scale_valid),support_valid=bool(support),unguarded_geometry_valid=geometric,guarded_valid=bool(support and g['valid'] and scale_valid),guard_applied=g['guard_applied'],guard_alpha=g['alpha'],scale_valid=scale_valid,condition_score=f['condition'],effective_samples=f['neff'],span_seconds=span,max_gap_seconds=maxgap,appearance_valid=bool(ac.any()),appearance_source='frozen_v2_for_controlled_geometry_temporal_comparison',tuning_split=tuning,action_split=rs[0]['action_split'],visor_split=rs[0]['visor_split'],quality_flags=([] if support else ['insufficient_temporal_support'])+([] if geometric else ['functional_curve_self_intersection_or_orientation'])+([] if scale_valid else ['functional_apparent_scale_unstable'])))
 out=directory('functional_fixed');save_shard(out/f'{v}.pt',{k:np.stack(a) for k,a in arrays.items()});save_index(out/f'{v}.parquet',meta);result=dict(video_id=v,windows=len(meta),support_valid=sum(r['support_valid'] for r in meta),valid_un_guarded=sum(r['valid'] for r in meta),valid_guarded=sum(r['guarded_valid'] for r in meta),guarded_windows=sum(r['guard_applied'] for r in meta),seconds=time.monotonic()-start);atomic_json(out/f'{v}.manifest.json',dict(state='working_not_finalized',inputs=dict(spatial_sha256=digest(sp/f'{v}.pt'),model_sha256=digest(directory('calibration')/'model.json')),tensor_sha256=digest(out/f'{v}.pt'),index_sha256=digest(out/f'{v}.parquet'),metrics=result));return result


def run(monitor):
 cfg=config();calibrate(monitor);results=pool(video,cfg['train']+cfg['validation'],monitor,workers=7);report('working_functional.json',dict(videos=results,state='open_not_finalized',ingestion_ready=False,track2_started=False,note='Explicitly choose the unguarded or guarded research variant. Preserved v2 appearance is controlled context; appearance ablations are separate. No default training release has been changed.',**identity()))
