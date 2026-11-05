"""Complete working temporal architecture variants; no final publication interface."""
from collections import defaultdict
import json,time
import numpy as np
import pyarrow.parquet as pq
import duckdb
from shapely import linearrings,is_simple
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.numerics import basis,wavelet_matrix
from refinement.kernels import periodic_design
from refinement.data import holdout
from refinement.diagnostics import reliability
from .common import config,path,allowed,split,report,pool,LOG,identity
from .architecture import polynomial_fit
from .working import working_cases,directory

VARIANTS=[dict(name='half_second_cubic',window=.5,degree=3,strength=3.),dict(name='one_second_degree5',window=1.,degree=5,strength=1/3)]
_SYNTHESIS={}

def synthesis(m,samples=128):
 key=(m,samples)
 if key not in _SYNTHESIS:_SYNTHESIS[key]=periodic_design(np.arange(samples)/samples,m=m)@wavelet_matrix(m).T
 return _SYNTHESIS[key]

def validity(c):
 degree=len(c)-1;m=(c.shape[1]-3)//2;t=np.unique(np.r_[np.linspace(-1,1,17),np.polynomial.legendre.leggauss(degree+1)[0]]);shape=(basis(t,degree=degree)@c)[:,2:-1].reshape(-1,m,2);curves=np.einsum('qn,tnc->tqc',synthesis(m),shape);area=np.sum(curves[:,:,0]*np.roll(curves[:,:,1],-1,axis=1)-curves[:,:,1]*np.roll(curves[:,:,0],-1,axis=1),axis=1);return np.asarray(is_simple(linearrings(curves)),bool)&(area>1e-8)

def collocation(c,window):
 degree=len(c)-1;q=degree+1;nodes,weights=np.polynomial.legendre.leggauss(q);h,v,a=[basis(nodes,d,degree=degree)@c*(2/window)**d for d in range(3)];xi=np.c_[v[:,:2],h[:,2:-1],v[:,2:-1],h[:,-1:],v[:,-1:]];xidot=np.c_[a[:,:2],v[:,2:-1],a[:,2:-1],v[:,-1:],a[:,-1:]];return dict(nodes=nodes,weights=weights*window/2,h=h,velocity=v,acceleration=a,xi=xi,xi_dot=xidot)

def guard(c,cov,observed):
 if validity(c).all():return dict(coefficients=c.copy(),covariance=cov.copy(),alpha=1.,valid=True)
 order=np.argsort(np.sum((observed[:,2:-1]-c[0,2:-1])**2,axis=1));best=None;phi=basis(np.linspace(-1,1,17),degree=len(c)-1)
 for index in order[:min(6,len(order))]:
  for alpha in [.9,.75,.5,.25,0.]:
   cc=c.copy();cc[:,2:-1]*=alpha;cc[0,2:-1]+=(1-alpha)*observed[index,2:-1]
   if validity(cc).all():
    error=float(np.mean((phi@(cc-c))**2))
    if best is None or error<best['distortion']:best=dict(coefficients=cc,alpha=alpha,distortion=error,valid=True)
    break
 if best is None:return dict(coefficients=c.copy(),covariance=cov.copy(),alpha=None,valid=False)
 delta=best['coefficients']-c;vc=cov.copy();vc[2:-1]*=best['alpha']**2;vc[2:-1]+=np.einsum('id,jd->dij',delta[:,2:-1],delta[:,2:-1]);best['covariance']=vc;return best


def calibrate(variant,monitor):
 cfg=config();base=json.loads((directory('calibration')/'model.json').read_text());scale=np.asarray(base['feature_scale']);records=[];window=variant['window'];degree=variant['degree'];strength=variant['strength']
 for v in cfg['train']+cfg['validation']:
  for case in working_cases([v]):
   t=case['time'];h=case['h']
   for protocol in ['single','gap100','endpoint100']:
    hidden=holdout(t,protocol,.875);keep=(~hidden)&(t>=1-window)
    if not hidden.any() or keep.sum()<max(8,degree+2):continue
    f=polynomial_fit(t[keep],h[keep],case['variance'][keep]*base['kappa_cov'],case['prior'][keep],case['quality'][keep],window,degree,strength,scale);g=guard(f['coefficients'],f['covariance'],h[keep]);phi=basis(2*(t[hidden]-(1-window))/window-1,degree=degree)
    for method,result in [('raw',f),('guarded',g)]:records.append(dict(split=case['split'],protocol=protocol,method=method,error=phi@result['coefficients']-h[hidden],variance=np.maximum(np.einsum('ni,dij,nj->nd',phi,result['covariance'],phi),1e-15)))
  monitor.check()
 multipliers={};evaluations={}
 for method in ['raw','guarded']:
  qs=[]
  for protocol in ['all','endpoint100']:
   rs=[r for r in records if r['split']=='train' and r['method']==method and (protocol=='all' or r['protocol']==protocol)];e=np.concatenate([r['error'] for r in rs]);var=np.concatenate([r['variance'] for r in rs]);qs.append(np.maximum(np.quantile(e*e/var,.95,axis=0)/1.9599639845**2,1e-4))
  mult=np.maximum(*qs);multipliers[method]=mult.tolist()
  for part in ['train','validation']:
   for protocol in ['single','gap100','endpoint100']:
    rs=[r for r in records if (r['split'],r['method'],r['protocol'])==(part,method,protocol)];e=np.concatenate([r['error'] for r in rs]);var=np.concatenate([r['variance'] for r in rs])*mult;evaluations[part+'/'+method+'/'+protocol]=reliability(e,var)
 model=dict(**variant,feature_scale=scale.tolist(),kappa_cov=base['kappa_cov'],multipliers=multipliers,evaluations=evaluations,training_videos=cfg['train'],state='working_not_finalized',**identity());dest=path('derived')/variant['name'];dest.mkdir(exist_ok=True);atomic_json(dest/'model.json',model);report(variant['name']+'_calibration.json',model);return model


def video(job):
 v,name=job;cfg=config();allowed([v]);tuning=split(v);memory_guard(cfg);dest=path('derived')/name;model=json.loads((dest/'model.json').read_text());window=model['window'];degree=model['degree'];q=degree+1;scale=np.asarray(model['feature_scale']);s=load_shard(directory('spatial')/f'{v}.pt');rows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();tracks=defaultdict(list)
 for r in rows:
  if r['spatial_valid']:tracks[r['entity_track_id']].append(r)
 with duckdb.connect() as db:duration=db.execute('SELECT duration_us FROM read_parquet(?)',[str(ROOT/f'canonical/v1/videos.{v}.parquet')]).fetchone()[0]
 arrays=defaultdict(list);meta=[];window_us=int(window*1e6);started=time.monotonic()
 for number,(track,rs) in enumerate(sorted(tracks.items())):
  if number%100==0:memory_guard(cfg)
  ids=np.array([r['row_index'] for r in rs]);times=np.array([r['timestamp_us'] for r in rs]);prior=np.array([r['provenance_weight'] for r in rs]);quality=np.array([r['geometry_weight'] for r in rs])
  for end in range(max(window_us,((int(times[0])+249999)//250000)*250000),min(int(times[-1])+window_us,duration)//250000*250000+1,250000):
   begin=end-window_us;lo=np.searchsorted(times,begin);hi=np.searchsorted(times,end,side='right')
   if lo==hi:continue
   local=ids[lo:hi];t=(times[lo:hi]-(end-1000000))/1e6;h=s['h'][local].astype(float);f=polynomial_fit(t,h,s['variance'][local].astype(float)*model['kappa_cov'],prior[lo:hi],quality[lo:hi],window,degree,model['strength'],scale);c=f['coefficients'];g=guard(c,f['covariance'],h);cov=f['covariance']*np.asarray(model['multipliers']['raw'])[:,None,None];gcov=g['covariance']*np.asarray(model['multipliers']['guarded'])[:,None,None];nodes=collocation(c,window);gnodes=collocation(g['coefficients'],window);ell=np.stack([basis(1.,d,degree=degree)*(2/window)**d for d in range(3)]);endpoints=ell@c;gendpoints=ell@g['coefficients'];values=dict(coefficients=c,coefficient_cov=cov,guarded_coefficients=g['coefficients'],guarded_coefficient_cov=gcov,endpoint=endpoints[0],endpoint_velocity=endpoints[1],endpoint_accel=endpoints[2],guarded_endpoint=gendpoints[0],guarded_endpoint_velocity=gendpoints[1],guarded_endpoint_accel=gendpoints[2],xi=nodes['xi'],xi_dot=nodes['xi_dot'],guarded_xi=gnodes['xi'],guarded_xi_dot=gnodes['xi_dot'])
   for key,value in values.items():arrays[key].append(value.astype(np.float32))
   physical_t=(times[lo:hi]-begin)/1e6;span=float(physical_t[-1]-physical_t[0]);maxgap=float(np.max(np.diff(np.r_[0,physical_t,window])));support=len(t)>=q and f['neff']>=6 and span>=.6*window and maxgap<=.35*window and f['condition']<1e10;simple=bool(validity(c).all());meta.append(dict(video_id=v,row_index=len(meta),entity_track_id=track,window_start_us=begin,window_end_us=end,window_seconds=window,hop_seconds=.25,degree=degree,coefficients_per_channel=q,collocation_nodes=q,state_dimension=35,physical_dimension=68,support_valid=bool(support),geometry_valid=simple,valid=bool(support and simple),guarded_valid=bool(support and g['valid']),guard_alpha=g['alpha'],effective_samples=f['neff'],condition=f['condition'],span_seconds=span,max_gap_seconds=maxgap,tuning_split=tuning,source_kind='source_annotated_functional_architecture_variant'))
 save_shard(dest/f'{v}.pt',{k:np.stack(a) for k,a in arrays.items()});save_index(dest/f'{v}.parquet',meta);metrics=dict(video_id=v,variant=name,windows=len(meta),support_valid=sum(r['support_valid'] for r in meta),valid=sum(r['valid'] for r in meta),guarded_valid=sum(r['guarded_valid'] for r in meta),seconds=time.monotonic()-started);atomic_json(dest/f'{v}.manifest.json',dict(state='working_not_finalized',tensor_sha256=digest(dest/f'{v}.pt'),index_sha256=digest(dest/f'{v}.parquet'),model_sha256=digest(dest/'model.json'),metrics=metrics));return metrics


def run(monitor):
 for variant in VARIANTS:calibrate(variant,monitor)
 cfg=config();results=pool(video,[(v,variant['name']) for variant in VARIANTS for v in cfg['train']+cfg['validation']],monitor,workers=7);report('working_temporal_architectures.json',dict(videos=results,state='open_not_finalized',ingestion_ready=False,track2_started=False,**identity()))
