"""Train-only fitting search, holdout-safe gauges and full partition state build."""
import collections,itertools,json,hashlib
import numpy as np
import pyarrow.parquet as pq
from track1.numerics import basis,wavelet_matrix
from track1_lab.temporal import fit as base_fit
from track1_lab.history_detail import fit_detail as base_detail,evaluate as detail_evaluate
from track1_lab.causal_kinematics import estimate
from track1_lab.variants import collocation
from track1_lab.topology_certificate import certify
from .common import *

def fit(time,h,variance,prior,quality,options,scale):
 if not options.get('families'):return base_fit(time,h,variance,prior,quality,options,scale)
 parts=[]
 for name,sl in [('position',slice(0,2)),('shape',slice(2,-1)),('scale',slice(-1,None))]:
  local={k:v for k,v in options.items() if k!='families'};local.update(options['families'][name]);parts.append(base_fit(time,h[:,sl],variance[:,sl],prior,quality,local,np.asarray(scale)[sl]))
 return dict(coefficients=np.concatenate([p['coefficients'] for p in parts],axis=1),covariance=np.concatenate([p['covariance'] for p in parts]),weights=np.concatenate([p['weights'] for p in parts],axis=1),condition=max(p['condition'] for p in parts),neff=min(p['neff'] for p in parts))

def fit_detail(time,h,variance,prior,quality,options,scale,**kwargs):
 coarse=fit(time,h,variance,prior,quality,options,scale)
 return base_detail(time,h,variance,prior,quality,options,scale,coarse_model=coarse,**kwargs)


def load(v,n,base=None):
 base=folder('spatial') if base is None else Path(base);rows=pq.read_table(base/f'{v}.parquet').to_pylist()
 return rows,{key:np.load(base/f'{v}.{key}{n}.npy',mmap_mode='r') for key in ['h','variance','controls']}

def proposals(v,rows,episodes=None):
 episodes=episodes if episodes is not None else [r for r in pq.read_table(ROOT/'dataset/partition_v2/episodes.parquet').to_pylist() if r['video_id']==v];tracks=collections.defaultdict(list)
 for r in rows:tracks[r['entity_track_id']].append(r['row_index'])
 seen=set();result=[]
 for track,ids in sorted(tracks.items()):
  times=np.array([rows[i]['timestamp_us'] for i in ids])
  for episode in episodes:
   if not np.any((times>=episode['start_us'])&(times<episode['end_us'])):continue
   for end in range(((episode['start_us']+1000000+249999)//250000)*250000,episode['end_us']+1,250000):
    key=(track,end)
    if key in seen:continue
    seen.add(key);local=np.asarray(ids)[(times>=end-1000000)&(times<=end)]
    result.append(dict(video_id=v,split=split(v),entity_track_id=track,window_end_us=int(end),indices=local.tolist(),episode_id=episode['episode_id']))
 return result

def timing_eligible(case,rows):
 ids=case['indices']
 if len(ids)<8:return False
 t=(np.array([rows[i]['timestamp_us'] for i in ids])-case['window_end_us']+1000000)/1e6
 return bool(t[-1]-t[0]>=.6 and max(np.diff(np.r_[0,t,1]))<=.35)


def phase(p,previous=None):
 a=(p-p.mean(0))/max(float(np.linalg.norm(p-p.mean(0))),1e-12)
 if previous is None:shift=int(np.argmin(abs(np.angle(np.exp(1j*(np.arctan2(a[:,1],a[:,0])-np.pi/2))))))
 else:
  b=(previous-previous.mean(0))/max(float(np.linalg.norm(previous-previous.mean(0))),1e-12);shift=int(np.argmin([np.sum((np.roll(a,-j,axis=0)-b)**2) for j in range(len(p))]))
 return np.roll(p,-shift,axis=0)

def regauge(rows,data,ids,n):
 """Gauge/match only supplied observations. Hidden targets never enter this path."""
 if not len(ids):raise ValueError('empty_retained')
 reference=None;hs=[];vars=[];allocation=None;roles=None;parents=None;signature=None
 for index in ids:
  r=rows[index]
  if not r[f'fit{n}_valid']:raise ValueError('invalid_spatial_fit')
  aa=[16] if n==16 else r['allocation'];rr=['outer'] if n==16 else r['roles'];pp=[-1] if n==16 else r['parents'];ss='dominant_outer' if n==16 else r['source_signature']
  if signature is not None and (ss!=signature or aa!=allocation):raise ValueError('retained_annotation_topology_change')
  signature=ss;allocation=list(aa);p=np.asarray(data['controls'][index],float);pieces=[];at=0
  for m in aa:pieces.append(p[at:at+m]);at+=m
  centre=np.asarray(r['centre_native']);scale=r['scale_native'];choices=[]
  if reference is None:
   # Canonical first-row order within interchangeable slots, independent of stored
   # global track ordering or any target used before this held-out protocol.
   orders=list(itertools.permutations(range(len(pieces))))
   for order in orders:
    remap={j:i for i,j in enumerate(order)}
    if any(aa[j]!=aa[i] or rr[j]!=rr[i] or remap.get(pp[j],-1)!=pp[i] for i,j in enumerate(order)):continue
    key=tuple(float(x) for j in order for x in (pieces[j].mean(0)-centre)/scale);choices.append((key,order))
  else:
   for order in itertools.permutations(range(len(pieces))):
    remap={j:i for i,j in enumerate(order)}
    if any(len(pieces[j])!=len(reference[i]) or rr[j]!=roles[i] or remap.get(pp[j],-1)!=parents[i] for i,j in enumerate(order)):continue
    cost=sum(np.linalg.norm((pieces[j].mean(0)-centre)/scale-reference_relative[i]) for i,j in enumerate(order));choices.append((cost,order))
  choices.sort()
  if not choices:raise ValueError('retained_ring_assignment_unsupported')
  if reference is not None and len(choices)>1 and choices[1][0]-choices[0][0]<.05:raise ValueError('retained_ring_assignment_ambiguous')
  order=choices[0][1];remap={j:i for i,j in enumerate(order)};roles=[rr[j] for j in order];parents=[remap.get(pp[j],-1) for j in order];current=[phase(pieces[j],None if reference is None else reference[i]) for i,j in enumerate(order)]
  starts=np.cumsum([2]+[2*m for m in aa]);oldvar=np.asarray(data['variance'][index],float);var=np.r_[oldvar[:2],np.concatenate([oldvar[starts[j]:starts[j+1]] for j in order]),oldvar[-1]]
  shape=np.concatenate([(wavelet_matrix(len(p))@((p-centre)/scale)).ravel() for p in current]);h=np.asarray(data['h'][index],float).copy();h[2:-1]=shape;hs.append(h);vars.append(var);reference=current;reference_relative=[(p.mean(0)-centre)/scale for p in current]
 return np.asarray(hs),np.asarray(vars),allocation,roles,parents

def prepare(case,rows,data,n,hidden=None,global_gauge=False):
 ids=np.array(case['indices'],int);time=(np.array([rows[i]['timestamp_us'] for i in ids])-case['window_end_us']+1000000)/1e6
 keep=np.ones(len(ids),bool) if hidden is None else ~np.asarray(hidden,bool)
 retained_before_filter=int(keep.sum());keep&=np.array([rows[i][f'fit{n}_valid'] for i in ids],bool)
 ids=ids[keep];t=time[keep]
 if len(t)<8:raise ValueError('too_few_retained_observations')
 if t[-1]-t[0]<.6:raise ValueError('insufficient_retained_span')
 if max(np.diff(np.r_[0,t,1]))>.35:raise ValueError('retained_gap')
 if global_gauge:
  if len({rows[i][f'epoch{n}'] for i in ids})!=1:raise ValueError('global_correspondence_epoch_change')
  first=rows[ids[0]];allocation=[16] if n==16 else first['allocation'];roles=['outer'] if n==16 else first['roles'];parents=[-1] if n==16 else first['parents'];h=np.asarray(data['h'][ids],float);var=np.asarray(data['variance'][ids],float)
 else:h,var,allocation,roles,parents=regauge(rows,data,ids,n)
 return dict(dropped_invalid_observations=retained_before_filter-len(ids),time=t,h=h,variance=var,prior=np.array([rows[i]['prior'] for i in ids]),quality=np.array([rows[i]['quality'] for i in ids]),ids=ids,allocation=allocation,roles=roles,parents=parents)

def holdout(case,rows,protocol):
 t=(np.array([rows[i]['timestamp_us'] for i in case['indices']])-case['window_end_us']+1000000)/1e6
 centre=[.25,.5,.75][int(rank(case['entity_track_id']+str(case['window_end_us']))[:8],16)%3]
 if protocol=='single':
  hidden=np.zeros(len(t),bool)
  if len(t):hidden[np.argmin(abs(t-centre))]=True
 elif protocol=='gap100':hidden=abs(t-centre)<=.05
 else:hidden=t>=.9
 return t,hidden

def target_h(rows,data,ids,n,prepared):
 """Evaluation targets align to an observed reference; never modify inputs."""
 result=[];compatible=[];last=prepared['h'][-1];allocation=prepared['allocation'];roles=prepared['roles'];parents=prepared['parents'];offsets=np.cumsum([2]+[2*m for m in allocation]);refs=[wavelet_matrix(m).T@last[offsets[j]:offsets[j+1]].reshape(m,2) for j,m in enumerate(allocation)]
 for index in ids:
  r=rows[index];h=np.asarray(data['h'][index],float).copy();aa=[16] if n==16 else r['allocation'];rr=['outer'] if n==16 else r['roles'];pp=[-1] if n==16 else r['parents'];choices=[];pieces=[];at=0
  for m in aa:pieces.append((np.asarray(data['controls'][index][at:at+m],float)-np.asarray(r['centre_native']))/r['scale_native']);at+=m
  if r[f'fit{n}_valid'] and len(aa)==len(allocation):
   for order in itertools.permutations(range(len(aa))):
    remap={j:i for i,j in enumerate(order)}
    if any(aa[j]!=allocation[i] or rr[j]!=roles[i] or remap.get(pp[j],-1)!=parents[i] for i,j in enumerate(order)):continue
    cost=sum(np.linalg.norm(pieces[j].mean(0)-refs[i].mean(0)) for i,j in enumerate(order));choices.append((cost,order))
  compatible.append(bool(choices))
  if choices:
   choices.sort();order=choices[0][1];aligned=[phase(pieces[j],refs[i]) for i,j in enumerate(order)];h[2:-1]=np.concatenate([(wavelet_matrix(m)@p).ravel() for m,p in zip(allocation,aligned)])
  result.append(h)
 return np.asarray(result),np.asarray(compatible)

def case_error(v,n,case,rows,data,options,scale,detail=None):
 result=[]
 for protocol in ['single','gap100','endpoint100']:
  times,hidden=holdout(case,rows,protocol)
  if not hidden.any():continue
  try:p=prepare(case,rows,data,n,hidden)
  except ValueError:continue
  f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,scale);target_ids=np.array(case['indices'])[hidden];targets,compatible=target_h(rows,data,target_ids,n,p);pred=basis(2*times[hidden]-1)@f['coefficients']
  if detail and detail['count']:
   m=fit_detail(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,scale,count=detail['count'],strength=detail['strength']);pred=detail_evaluate(m,times[hidden])
  error=(pred-targets)/scale;position=float(np.mean(error[:,:2]**2));shape=float(np.mean(error[compatible,2:-1]**2)) if compatible.any() else None;size=float(np.mean(error[:,-1]**2));objective=.35*position+.15*size+(.5*shape if shape is not None else .5)
  variance=np.maximum(np.einsum('ni,dij,nj->nd',basis(2*times[hidden]-1),f['covariance'],basis(2*times[hidden]-1)),1e-15);result.append(dict(video_id=v,model=n,protocol=protocol,objective=objective,position_mse=position,shape_mse=shape,scale_mse=size,compatible_targets=int(compatible.sum()),targets=len(targets),error=(pred-targets),variance=variance,compatible=compatible))
 return result

def scales():
 result={}
 for n in [16,32]:
  values=[]
  for v in CFG['train']:
   rows,d=load(v,n);ids=[r['row_index'] for r in rows if r[f'fit{n}_valid']];a=np.asarray(d['h'][ids],float);values.append(a)
  x=np.concatenate(values);centre=np.median(x,axis=0);result[str(n)]=np.maximum(1.4826*np.median(abs(x-centre),axis=0),.01).tolist()
 return result

def search_worker(job):
 v,n,options,scale,detail=job;rows,data=load(v,n);all_cases=proposals(v,rows);eligible=[c for c in all_cases if timing_eligible(c,rows)];cases=sorted(eligible,key=lambda c:rank(c['entity_track_id']+str(c['window_end_us'])))[:CFG['max_search_cases_per_video']];records=[];ratios=[]
 for case in cases:
  for r in case_error(v,n,case,rows,data,options,np.array(scale),detail):
   e=r.pop('error');var=r.pop('variance');compatible=r.pop('compatible');ratio=e**2/var
   if not compatible.all():ratio[~compatible,2:-1]=np.nan
   ratios.append((r['protocol'],ratio));records.append(r)
 return dict(video_id=v,model=n,options=options,detail=detail,proposed_cases=len(all_cases),timing_eligible_cases=len(eligible),sampled_cases=len(cases),records=records,ratios=ratios)

def summarize(jobs):
 vals=[]
 for job in jobs:
  if job['records']:vals.append(float(np.mean([r['objective'] for r in job['records']])))
 return float(np.mean(vals)) if vals else float('inf')

def search(monitor):
 fs=scales();atomic_json(folder('calibration')/'scales.json',fs);trial_options=[dict(regularization=r,loss=loss,variance_power=.25,provenance_power=.5,mode_penalty=[0,0,1,9],floor=.01,variance_clip=100,irls=3,delta=3.) for r in [.003,.01]+CFG['regularization_trials']+[10.,30.,100.] for loss in CFG['loss_trials']];selected={};ledger=[]
 for n in [16,32]:
  trials=[]
  for options in trial_options:
   jobs=pool(search_worker,[(v,n,options,fs[str(n)],None) for v in CFG['train']],monitor,workers=5);score=summarize(jobs);trials.append((score,options,jobs));ledger.append(dict(model=n,options=options,train_objective=score,by_video=[dict(video_id=j['video_id'],proposed_cases=j['proposed_cases'],timing_eligible_cases=j['timing_eligible_cases'],sampled_cases=j['sampled_cases'],supported_holdout_probes=len(j['records'])) for j in jobs]));LOG.info('Train search M%d %s score %.6f',n,options,score)
  score,options,jobs=min(trials,key=lambda x:x[0]);options=dict(options)
  def component_score(values,key,protocol=None):
   means=[]
   for job in values:
    values=[r[key] for r in job['records'] if r[key] is not None and (protocol is None or r['protocol']==protocol)]
    if values:means.append(float(np.mean(values)))
   return float(np.mean(means)) if means else float('inf')
  families={}
  for family,key in [('position','position_mse'),('shape','shape_mse'),('scale','scale_mse')]:
   best=min(trials,key=lambda trial:component_score(trial[2],key));families[family]={k:best[1][k] for k in ['regularization','loss']}
  options['families']=families
  jobs=pool(search_worker,[(v,n,options,fs[str(n)],None) for v in CFG['train']],monitor,workers=5);score=summarize(jobs);position_score=component_score(jobs,'position_mse');endpoint_score=component_score(jobs,'position_mse','endpoint100');detail_trials=[(position_score,dict(count=0,strength=0.),score)]
  LOG.info('Selected families M%d %s combined=%.6f position=%.6f',n,families,score,position_score)
  for count in [2,3]:
   for strength in [.01,.1]:
    detail=dict(count=count,strength=strength);values=pool(search_worker,[(v,n,options,fs[str(n)],detail) for v in CFG['train']],monitor,workers=5);detail_score=component_score(values,'position_mse');endpoint_detail=component_score(values,'position_mse','endpoint100');combined=summarize(values)
    if endpoint_detail<=endpoint_score*1.02:detail_trials.append((detail_score,detail,combined))
    LOG.info('Train detail M%d %s position=%.6f endpoint=%.6f combined=%.6f',n,detail,detail_score,endpoint_detail,combined)
  bestposition,detail,bestscore=min(detail_trials,key=lambda x:x[0])
  if bestposition>position_score*.998:detail=dict(count=0,strength=0.)
  qs=[]
  for protocol in ['all','endpoint100']:
   a=[r for j in jobs for p,r in j['ratios'] if protocol=='all' or p==protocol]
   ratio=np.concatenate(a) if a else np.ones((1,2*n+3));qs.append(np.maximum(np.nanquantile(ratio,.95,axis=0)/1.9599639845**2,1e-4))
  multiplier=np.nan_to_num(np.maximum(*qs),nan=1.,posinf=1e6);array(folder('calibration')/f'm{n}.multipliers.npy',multiplier)
  selected[str(n)]=dict(options=options,detail=detail,feature_scale=fs[str(n)],train_objective=score,detail_train_objective=bestscore,position_train_objective=position_score,detail_position_train_objective=bestposition,detail_selection='position-only improvement >=0.2%; endpoint MSE <=2% worse; endpoint contribution remains exactly zero',multipliers=multiplier.tolist(),covariance_scope='approximate held-out source representation, not independent human/motion uncertainty')
 atomic_json(REPORT/'search.json',dict(trials=ledger,selected=selected,selection_split='five training participants only',validation_used_for_selection=False,maximum_cases_per_video=CFG['max_search_cases_per_video'],code=code()));atomic_json(folder('calibration')/'model.json',selected)

def build_worker(job):
 v,n=job[:2];settings=job[2] if len(job)>2 else None;root=Path(settings['root']) if settings else OUT;reports=Path(settings['reports']) if settings else REPORT;rows,data=load(v,n,root/'spatial');episodes=pq.read_table(settings['episodes']).to_pylist() if settings else None;episodes=[r for r in episodes if r['video_id']==v] if episodes is not None else None;cases=proposals(v,rows,episodes);selected=json.loads((root/'calibration/model.json').read_text())[str(n)];scale=np.array(selected['feature_scale']);options=selected['options'];d=2*n+3;out=root/'functional';out.mkdir(exist_ok=True);meta=[];values=collections.defaultdict(list);appearance=np.load(root/'appearance'/f'{v}.projected.npy',mmap_mode='r');appmeta=pq.read_table(root/'appearance'/f'{v}.parquet',columns=['mask_uid','row_index']).to_pylist();app_lookup={r['mask_uid']:r['row_index'] for r in appmeta};started=time.monotonic()
 for case in cases:
  valid=False;reason=None;p=None;detail=selected['detail'];c=np.zeros((4,d));cov=np.zeros((d,4,4));gc=c.copy();xi=np.zeros((4,2*d-2));xidot=xi.copy();res=np.zeros((3,2));transform=np.zeros((7,3));app=np.zeros((4,2,128));kp=np.zeros(2);kv=np.zeros(2);kc=np.zeros((4,4));ks=False;certificate=None;alpha=None;source_complete=False;geometry_valid=False;neff=0.;latest=None;orth=None;fit_rmse=None;fit_r2=None
  raw_ids=np.asarray(case['indices'],int);raw_time=(np.array([rows[i]['timestamp_us'] for i in raw_ids])-case['window_end_us']+1000000)/1e6
  kin=estimate(raw_time,np.asarray(data['h'][raw_ids,:2],float),np.asarray(data['variance'][raw_ids,:2],float),1.);ks=kin.support_valid
  if ks:kp=kin.position;kv=kin.velocity;kc=kin.covariance
  kin_latest=int(case['window_end_us']-1000000+round(kin.latest_input_time*1e6)) if kin.latest_input_time is not None else None
  try:
   p=prepare(case,rows,data,n,global_gauge=True);f=fit(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,scale);c=f['coefficients'];cov=f['covariance']*np.array(selected['multipliers'])[:,None,None];neff=f['neff']
   if neff<6 or f['condition']>=1e10:raise ValueError('ill_conditioned_or_low_effective_support')
   valid=True;latest=int(max(rows[i]['timestamp_us'] for i in p['ids']));errors=p['h']-basis(2*p['time']-1)@c;fit_rmse=float(np.sqrt(np.mean((errors/scale)**2)));ss=((p['h']-p['h'].mean(0))**2).sum(0);active=ss>1e-8*np.maximum((p['h']**2).sum(0),1e-12);fit_r2=float(np.median(1-(errors[:,active]**2).sum(0)/ss[active])) if active.any() else None;nodes=collocation(c,1.);xi=nodes['xi'];xidot=nodes['xi_dot'];source_complete=all((rows[i]['source_rings']==1 if n==16 else rows[i]['source_complete']) for i in p['ids'])
   if detail['count']:
    dm=fit_detail(p['time'],p['h'],p['variance'],p['prior'],p['quality'],options,scale,count=detail['count'],strength=detail['strength']);res[:detail['count']]=dm['detail'];transform[:,:detail['count']]=dm['transform'];orth=dm['orthogonality_error']
   app_ids=[app_lookup[rows[i]['mask_uid']] for i in p['ids']];y=np.asarray(appearance[app_ids],float).reshape(len(app_ids),256);phi=basis(2*p['time']-1);w=p['prior']**.5*p['quality'];w*=len(w)/w.sum();app=np.linalg.solve(phi.T@(w[:,None]*phi)+np.diag([1e-8,1e-8,.01,.09]),phi.T@(w[:,None]*y)).reshape(4,2,128)
   # Certified decoder gets serialized coefficients. A bounded shape rollback is
   # a separate geometry view; it never changes authoritative local kinematics.
   gc=c.copy()
   for amount in [1.,.75,.25,0.]:
    trial=c.copy();trial[:,2:-1]*=amount;trial[0,2:-1]+=(1-amount)*p['h'][-1,2:-1];trial=trial.astype(np.float32).astype(float);cert=certify(trial,p['allocation'],p['roles'],p['parents'],window_seconds=1.,max_depth=8,max_nodes=255)
    if amount==1. or cert['certified']:gc=trial;certificate=cert;alpha=amount
    if cert['certified']:geometry_valid=True;break
  except (ValueError,np.linalg.LinAlgError) as e:reason=str(e);valid=False
  for name,a in dict(coefficients=c,coefficient_cov=cov,geometry_coefficients=gc,xi=xi,xi_dot=xidot,detail=res,detail_transform=transform,appearance_coefficients=app,kinematic_position=kp,kinematic_velocity=kv,kinematic_cov=kc).items():values[name].append(a)
  allocation=p['allocation'] if p else ([16] if n==16 else []);roles=p['roles'] if p else [];parents=p['parents'] if p else []
  meta.append(dict(video_id=v,split=split(v),model=n,row_index=len(meta),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],window_start_us=case['window_end_us']-1000000,window_seconds=1.,hop_seconds=.25,gauge_scope='causal provisional-track gauge; holdouts re-encode retained inputs',state_channels=d,physical_channels=2*d-2,valid=bool(valid),failure_reason=reason,source_fit_nrmse=fit_rmse,source_median_nonconstant_r2=fit_r2,retained_mask_uids=[rows[i]['mask_uid'] for i in p['ids']] if p is not None else [],coordinate_mode='intrinsic_normalized_2D',geometry_schema='canonical_geometry2',source_input_observations=len(case['indices']),retained_input_observations=len(p['ids']) if p else 0,dropped_spatial_invalid_observations=p['dropped_invalid_observations'] if p else 0,effective_samples=float(neff),latest_input_us=latest,allocation=allocation,roles=roles,parents=parents,source_complete=bool(source_complete),certified_geometry_valid=bool(valid and geometry_valid),geometry_training_eligible=bool(valid and geometry_valid and source_complete),certificate_json=json.dumps(certificate,sort_keys=True),geometry_guard_alpha=alpha,geometry_covariance_available=False,kinematics_valid=bool(ks),kinematics_latest_input_us=kin_latest,kinematics_observation_count=kin.observation_count,kinematics_failure_reason=kin.failure_reason,kinematics_scope='recent source centroids; independent of functional and spline-fit support',kinematic_covariance_calibrated=False,identity_verified=False,online_observation_allowed=False,detail_count=detail['count'] if valid else 0,detail_fit_orthogonality_error=orth,appearance_valid=bool(valid),covariance_calibration_scope='raw functional source-representation approximation only',detail_covariance_calibrated=False,episode_id=case['episode_id'],epoch=rows[p['ids'][0]][f'epoch{n}'] if p is not None else None))
  if len(meta)%64==0:guard()
 for name,arrays in values.items():array(out/f'{v}.m{n}.{name}.npy',arrays)
 save(out/f'{v}.m{n}.parquet',meta);summary=dict(video_id=v,model=n,attempted=len(meta),valid=sum(r['valid'] for r in meta),certified_geometry=sum(r['certified_geometry_valid'] for r in meta),complete_certified_geometry=sum(r['geometry_training_eligible'] for r in meta),kinematic_supported=sum(r['kinematics_valid'] for r in meta),seconds=time.monotonic()-started);atomic_json(reports/f'functional.{v}.m{n}.json',summary);return summary

def build(monitor):
 results=pool(build_worker,[(v,n) for v in VIDEOS for n in [16,32]],monitor,workers=7);atomic_json(REPORT/'functional.json',dict(videos=results,online_inputs=False,source='dataset_v2_native_partition',kinematics_independent_of_functional_derivative=True,code=code()))
