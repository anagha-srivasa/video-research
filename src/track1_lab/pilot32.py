"""Paired end-to-end 32-control pilot; preserves source IDs and never loads exposed test videos."""
from collections import defaultdict
import json,time
import numpy as np
import pyarrow.parquet as pq
import duckdb
from pipeline.common import ROOT,atomic_json
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.numerics import wavelet_matrix,basis
from track1.spatial import resample,dominant_ring,wavelet_state
from track1.camera import undistort
from refinement.kernels import periodic_design
from refinement.protocol import rank
from refinement.data import temporal_cases,holdout,spatial_cases
from .architecture import general_fit,curve_coordinates,polynomial_fit
from .geometry_contract import dominant_ring
from .common import config,path,allowed,split,report,pool,LOG,identity
from .working import directory
from .variants import collocation,validity


def dest():
 p=path('derived')/'pilot32';p.mkdir(exist_ok=True);return p


def selected_cases(v):return sorted(temporal_cases(allowed([v])),key=lambda c:rank('pilot32'+c['entity_track_id']+str(c['window_end_us'])))[:48]


def spatial_video(v):
 allowed([v]);started=time.monotonic();old=load_shard(directory('spatial')/f'{v}.pt');allrows=pq.read_table(directory('spatial')/f'{v}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r for r in allrows};cases=selected_cases(v);wanted={lookup[(c['entity_track_id'],int(c['window_end_us']-1000000+round(t*1e6)))]['mask_uid'] for c in cases for t in c['time']};rows=[r for r in allrows if r['mask_uid'] in wanted]
 with duckdb.connect() as db:
  db.execute('SET threads=1');geometry={uid:(rings,aw,ah) for uid,rings,aw,ah in db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE mask_uid IN (SELECT unnest(?))',[str(ROOT/f'canonical/v1/masks.{v}.parquet'),list(wanted)]).fetchall()}
 camera=json.loads((directory('spatial')/f'{v}.camera.json').read_text());k=np.asarray(camera['intrinsic_native']);dist=np.asarray(camera['distortion']);w=wavelet_matrix(32);phi=periodic_design(np.arange(64)/64,m=32);meanphi=phi.mean(0);arrays=defaultdict(list);meta=[]
 for number,r in enumerate(rows):
  if number%256==0:memory_guard(config())
  i=r['row_index'];rings,aw,ah=geometry[r['mask_uid']];ring,components,holes=dominant_ring(dict(contours_json=rings),np.diag([r['width']/aw,r['height']/ah,1.]));y=np.roll(resample(ring,128),-2*r['seam_shift'],axis=0);f=general_fit(y,m=32);p=f['controls'];cov=f['covariance'];centre=old['centre_native'][i].astype(float);scale=float(old['scale_native'][i]);z,zcov=wavelet_state(p,cov,centre,scale);h=np.r_[undistort(centre[None],k,dist)[0],z.reshape(-1),np.log(scale/np.sqrt(k[0,0]*k[1,1]))];cv=float(meanphi@cov@meanphi);curve=phi@p;gx=(np.roll(curve[:,1],-1)-np.roll(curve[:,1],1))/2;gy=(np.roll(curve[:,0],1)-np.roll(curve[:,0],-1))/2;ax=phi.T@gx;ay=phi.T@gy;lv=float((ax@cov@ax+ay@cov@ay)/(4*max((np.pi*scale**2)**2,1)));sv=np.repeat(np.diag(zcov),2)+np.repeat((w@np.ones(32))**2*cv/scale**2,2)+z.reshape(-1)**2*lv;var=np.r_[cv/k[0,0]**2,cv/k[1,1]**2,sv,lv]
  for key,a in dict(controls=p,covariance=cov,h=h,variance=var,boundary=y).items():arrays[key].append(np.asarray(a,np.float32))
  meta.append(dict(video_id=v,row_index=len(meta),original_row_index=i,mask_uid=r['mask_uid'],entity_track_id=r['entity_track_id'],timestamp_us=r['timestamp_us'],source_kind=r['source_kind'],source_spatial_valid=r['spatial_valid'],spatial_valid=f['valid'],component_count=components,hole_count=holes,width=r['width'],height=r['height']))
 save_shard(dest()/f'{v}.pt',{k:np.stack(a) for k,a in arrays.items()});save_index(dest()/f'{v}.parquet',meta);return dict(video_id=v,observations=len(meta),valid=sum(r['spatial_valid'] for r in meta),seconds=time.monotonic()-started)


def global_calibration(monitor):
 cfg=config();ratios=[]
 for v in cfg['train']:
  for case in [c for c in spatial_cases([v]) if c['source_kind']=='manual_sparse'][:32]:
   y=case['y'];weights=np.ones(128);weights[::4]=0;f=general_fit(y,m=32,sample_weights=weights)
   if not f['valid']:continue
   target=y[::8];_,u=curve_coordinates(target,f['controls'],32);phi=periodic_design(u,m=32);dp=periodic_design(u,m=32,derivative=1)@f['controls'];normal=np.c_[dp[:,1],-dp[:,0]];normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);e=np.sum((target-phi@f['controls'])*normal,axis=1);var=np.maximum(np.einsum('ni,ij,nj->n',phi,f['covariance'],phi),1e-12);ratios.extend((e*e/var).tolist())
  monitor.check()
 kappa=float(np.quantile(ratios,.95)/1.9599639845**2);scales={}
 for m in [16,32]:
  vals=[]
  for v in cfg['train']:
   p=dest() if m==32 else directory('spatial');s=load_shard(p/f'{v}.pt')
   if m==16:
    rows=pq.read_table(dest()/f'{v}.parquet',columns=['original_row_index']).column(0).to_numpy();vals.append(s['h'][rows].astype(float))
   else:vals.append(s['h'].astype(float))
  x=np.concatenate(vals);centre=np.median(x,axis=0);scales[str(m)]=np.maximum(np.median(abs(x-centre),axis=0)*1.4826,.01).tolist()
 model=dict(kappa32=kappa,kappa16=json.loads((directory('calibration')/'model.json').read_text())['kappa_cov'],feature_scales=scales,training_videos=cfg['train'],spatial_calibration_points=len(ratios),state='pilot_not_finalized',**identity());atomic_json(dest()/'model.json',model);return model


def evaluate_video(v):
 import cv2
 from pipeline import Dataset
 from track1.evaluation import geometry_metrics
 from .motion import project
 cv2.setNumThreads(1);model=json.loads((dest()/'model.json').read_text());s32=load_shard(dest()/f'{v}.pt');r32=pq.read_table(dest()/f'{v}.parquet').to_pylist();lookup={(r['entity_track_id'],r['timestamp_us']):r for r in r32};s16=load_shard(directory('spatial')/f'{v}.pt');camera=json.loads((directory('spatial')/f'{v}.camera.json').read_text());k=np.asarray(camera['intrinsic_native']);records=[];models=[(16,1.,3,1/3),(16,.5,3,3.),(32,1.,3,1/3),(32,.5,3,3.),(32,1.,5,1/3)];examples=[];saved=defaultdict(list)
 with Dataset() as ds:
  for case_number,c in enumerate(selected_cases(v)):
   t=c['time'];rows=[lookup[(c['entity_track_id'],int(c['window_end_us']-1000000+round(tt*1e6)))] for tt in t];ids32=np.array([r['row_index'] for r in rows]);ids16=np.array([r['original_row_index'] for r in rows])
   for protocol in ['single','gap100','endpoint100']:
    hidden=holdout(t,protocol,.875)
    if not hidden.any():continue
    shared=(~hidden)&(t>=.5)
    if shared.sum()<8:continue
    target_index=np.flatnonzero(hidden)[-1];target_meta=rows[target_index];target=ds.get_mask(target_meta['mask_uid']);height,width=target.shape
    for m,window,degree,strength in models:
     state=s16 if m==16 else s32;ids=ids16 if m==16 else ids32;keep=(~hidden)&(t>=1-window);h=state['h'][ids].astype(float);scale=np.asarray(model['feature_scales'][str(m)]);result=polynomial_fit(t[keep],h[keep],state['variance'][ids[keep]].astype(float)*model['kappa'+str(m)],c['prior'][keep],c['quality'][keep],window,degree,strength,scale);ph=basis(2*(t[hidden]-(1-window))/window-1,degree=degree);pred=ph@result['coefficients'];error=pred-h[hidden];hp=pred[-1];centre=project(hp[:2],camera);apparent=np.sqrt(k[0,0]*k[1,1])*np.exp(np.clip(hp[-1],-15,2));controls=wavelet_matrix(m).T@hp[2:-1].reshape(m,2)*apparent+centre;curve=periodic_design(np.arange(512)/512,m=m)@controls;mask=np.zeros_like(target);cv2.fillPoly(mask,[np.rint(curve*[width/target_meta['width'],height/target_meta['height']]).astype(np.int32)],1);gm=geometry_metrics(target,mask);groups={'position':slice(0,2),'coarse_shape':slice(2,2+m),'fine_shape':slice(2+m,2+2*m),'scale':slice(2+2*m,3+2*m)}
     for group,sl in groups.items():records.append(dict(video_id=v,split=split(v),entity_track_id=c['entity_track_id'],window_end_us=c['window_end_us'],mask_uid=target_meta['mask_uid'],target_source_kind=target_meta['source_kind'],target_m32_spatial_valid=bool(target_meta['spatial_valid']),input_m32_valid_fraction=float(np.mean([rr['spatial_valid'] for rr in np.array(rows,dtype=object)[keep]])),scale_clipped=bool(hp[-1]<-15 or hp[-1]>2),controls=m,window_seconds=window,degree=degree,protocol=protocol,group=group,normalised_mse=float(np.mean((error[:,sl]/scale[sl])**2)),state_dimension=2*m+3,physical_dimension=4*m+4,coefficient_count=(degree+1)*(2*m+3),input_observations=int(keep.sum()),**gm))
     if m==32 and protocol=='endpoint100':
      name=f'T{window:g}_D{degree}';saved[name+'_coefficients'].append(result['coefficients'].astype(np.float32));nodes=collocation(result['coefficients'],window);saved[name+'_xi'].append(nodes['xi'].astype(np.float32))
   if case_number%16==0:memory_guard(config())
 save_index(path('reports')/f'pilot32.{v}.parquet',records);save_shard(dest()/f'functional.{v}.pt',{k:np.stack(a) for k,a in saved.items()});return dict(video_id=v,case_rows=len(records))


def run(monitor):
 cfg=config();front=pool(spatial_video,cfg['train']+cfg['validation'],monitor,workers=7);global_calibration(monitor);pool(evaluate_video,cfg['train']+cfg['validation'],monitor,workers=7);records=[]
 for v in cfg['train']+cfg['validation']:records+=pq.read_table(path('reports')/f'pilot32.{v}.parquet').to_pylist()
 save_index(path('reports')/'pilot32.parquet',records);grouped=defaultdict(list)
 for r in records:grouped[(r['split'],r['controls'],r['window_seconds'],r['degree'],r['protocol'],r['group'],r['video_id'])].append(r)
 summary={}
 for m,window,degree in [(16,1.,3),(16,.5,3),(32,1.,3),(32,.5,3),(32,1.,5)]:
  name=f'M{m}_T{window:g}_D{degree}';summary[name]={}
  for protocol in ['single','gap100','endpoint100']:
   summary[name][protocol]={}
   for group in ['position','coarse_shape','fine_shape','scale']:
    byvideo={}
    for v in cfg['validation']:
     rs=grouped[('validation',m,window,degree,protocol,group,v)]
     if rs:byvideo[v]=dict(n=len(rs),normalised_rmse=float(np.sqrt(np.mean([r['normalised_mse'] for r in rs]))),iou=float(np.mean([r['iou'] for r in rs])),boundary_fscore_2px=float(np.mean([r['boundary_fscore_2px'] for r in rs])))
    if byvideo:summary[name][protocol][group]={k:float(np.mean([r[k] for r in byvideo.values()])) for k in ['normalised_rmse','iou','boundary_fscore_2px']}
 report('pilot32.json',dict(spatial=front,validation_metrics=summary,windows_per_video=48,limits=['Paired deterministic source windows; same hidden physical observations for all models.','Per-feature scales learned on identical training-pilot observations, separately for each representation. Normalised feature errors across different channel counts are not identical physical quantities.','Mask metrics are true full source-mask reconstructions at the last hidden observation, stratifiable by source provenance. They are not RGB segmentation predictions.','Single-ring 32-control temporal encoding only; the strong multi-ring geometric result still needs persistent loop-slot correspondence before temporal integration.','Polynomial coefficients and physical collocation tensors are saved for 32-control models; this is a bounded pilot, not a full-release store or scientific acceptance.'],state='pilot_not_finalized',**identity()))
