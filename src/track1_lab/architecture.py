"""Bounded architecture ablations, isolated from fixed-budget working builds."""
from collections import defaultdict
import json,time
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,LinearRing
from shapely.ops import unary_union
from shapely import make_valid
from pipeline.common import ROOT
from track1.artifacts import save_index,load_shard,save_shard
from track1.spatial import resample
from track1.numerics import weighted_solve,huber_weights,basis
from refinement.kernels import periodic_design
from refinement.data import spatial_cases
from refinement.protocol import rank
from .spatial import closest_polyline
from .common import config,path,allowed,split,pool,report,LOG,identity
from .geometry_contract import shape as contract_shape,select as select_geometry,render as render_geometry


def curve_coordinates(query,p,m,samples=512):
 a=periodic_design(np.arange(samples)/samples,m=m)@p;ab=np.roll(a,-1,axis=0)-a;delta=query[:,None]-a;u=np.clip(np.sum(delta*ab,axis=2)/np.maximum(np.sum(ab*ab,axis=1),1e-20),0,1);projected=a+u[:,:,None]*ab;d=np.sum((query[:,None]-projected)**2,axis=2);indices=d.argmin(1);return np.sqrt(d[np.arange(len(query)),indices]),(indices+u[np.arange(len(query)),indices])/samples


def general_fit(y,m=16,reverse=1.,steps=12,sample_weights=None):
 y=np.asarray(y,float);weights=np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights,float);u=np.arange(len(y))/len(y);observed=y[weights>0];d2=np.roll(np.eye(m),1,axis=0)-2*np.eye(m)+np.roll(np.eye(m),-1,axis=0);p=None;best=None;bestscore=np.inf;grid=periodic_design(np.arange(256)/256,m=m)
 # Ordered initial reparameterization, followed by symmetric contour projection.
 for iteration in range(7):
  phi=periodic_design(u,m=m);w=weights.copy()
  for _ in range(3):
   gram=phi.T@(w[:,None]*phi);prior=.0001*np.trace(gram)/np.trace(d2.T@d2)*(d2.T@d2)+1e-6*np.eye(m);p,cov,condition,solver=weighted_solve(phi,y,w,prior);residual=np.linalg.norm(phi@p-y,axis=1);sigma=max(float(np.median(residual[weights>0])/.67449),1.);w=weights*huber_weights(residual/(sigma+1.),1.5)
  if not LinearRing(grid@p).is_simple:break
  projected,distance=closest_polyline(observed,grid@p);score=float(np.mean(distance**2))
  if score>bestscore:break
  sigma2=max(float(np.sum(w*np.sum((phi@p-y)**2,axis=1))/(2*max(w.sum()-m,1))),1.);best=dict(controls=p.copy(),covariance=cov*sigma2,valid=True,condition=condition);bestscore=score;delta=y-phi@p;tangent=periodic_design(u,m=m,derivative=1)@p;step=np.sum(delta*tangent,axis=1)/np.maximum(np.sum(tangent*tangent,axis=1),1e-12);step=np.clip(step,-1/64,1/64)*.5;step[weights==0]=0;step[0]=0;uu=np.clip(u+step,0,1-1e-6)
  for j in range(1,len(uu)):uu[j]=max(uu[j],uu[j-1]+.01/len(uu))
  if uu[-1]>=1:uu*=((1-.01/len(uu))/uu[-1])
  uu[0]=0;u=uu
 if best is None:return dict(controls=p,covariance=cov,valid=False,condition=condition)
 p=best['controls'];anchor=p.copy();reverse_design=periodic_design(np.arange(128)/128,m=m)
 def score(pp):
  _,forward=closest_polyline(observed,grid@pp);projection,back=closest_polyline(reverse_design@pp,observed);return float(np.mean(forward**2)+reverse*np.mean(back**2)),projection
 bestscore,projection=score(p)
 for _ in range(steps):
  _,candidate_u=curve_coordinates(y,p,m);phi=periodic_design(candidate_u,m=m);residual=np.linalg.norm(phi@p-y,axis=1);w=weights*huber_weights(residual/max(float(np.median(residual[weights>0])/.67449),1.),2.);design=np.vstack([phi,reverse_design,np.eye(m)]);targets=np.vstack([y,projection,anchor]);ww=np.r_[w,np.full(128,reverse*w.sum()/128),np.full(m,.0001*w.sum()/m)];gram=design.T@(ww[:,None]*design);prior=.00001*np.trace(gram)/np.trace(d2.T@d2)*(d2.T@d2)+1e-6*np.eye(m);pp,cov,condition,solver=weighted_solve(design,targets,ww,prior);newscore,newprojection=score(pp)
  if newscore>bestscore or not LinearRing(grid@pp).is_simple:break
  r=targets-design@pp;sigma2=max(float(np.sum(ww*np.sum(r*r,axis=1))/(2*max(ww.sum()-m,1))),1.);best=dict(controls=pp.copy(),covariance=cov*sigma2,valid=True,condition=condition);bestscore=newscore;projection=newprojection
  if np.max(abs(pp-p))<1e-4:break
  p=pp
 return best


def source_geometry(row):
 return contract_shape(row['contours_json'])


def spatial_video(v):
 import cv2
 from pipeline import Dataset
 from track1.evaluation import geometry_metrics
 cv2.setNumThreads(1);allowed([v]);source={r['mask_uid']:r for r in pq.read_table(ROOT/f'canonical/v1/masks.{v}.parquet').to_pylist()};old=load_shard(ROOT/f'derived/track1_refined_v2/spatial/{v}.pt');meta={r['mask_uid']:r for r in pq.read_table(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').to_pylist()};manual=[c for c in spatial_cases([v]) if c['source_kind']=='manual_sparse'];manual=sorted(manual,key=lambda c:rank('architecture'+c['mask_uid']))[:64];records=[];previews=[]
 with Dataset() as ds:
  for case in manual:
   uid=case['mask_uid'];r=meta[uid];sr=source[uid];target=ds.get_mask(uid);height,width=target.shape;truth,rings=source_geometry(sr);external=[rr for rr in rings if rr['role']=='outer' and len(rr['points'])>=3];main=max(external,key=lambda rr:abs(rr['signed_area']));y=resample(main['points'],128);scale=max(float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1)))),1.);bbox=np.array(truth.bounds);tol=max(1.,.01*np.hypot(bbox[2]-bbox[0],bbox[3]-bbox[1]));bs=target.astype(np.uint8)-cv2.erode(target.astype(np.uint8),np.ones((3,3),np.uint8));dst=cv2.distanceTransform(1-bs,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
   for mode in ['v2_16','symmetric_16','symmetric_24','symmetric_32','multi_ring_32']:
    started=time.perf_counter();fitted_rings=[];controls_count=0;valid=True
    if mode=='v2_16':
     curve=periodic_design(np.arange(512)/512)@old['controls'][r['row_index']]*[width/r['width'],height/r['height']];fitted_rings=[('outer',curve)];controls_count=16;valid=r['spatial_valid']
    else:
     if mode=='multi_ring_32':
      chosen=select_geometry(rings,3);allocation=[32] if len(chosen)==1 else [24,8] if len(chosen)==2 else [16,8,8]
     else:chosen=[main];allocation=[int(mode.rsplit('_',1)[1])]
     for ring,m in zip(chosen,allocation):
      yy=resample(ring['points'],128);result=general_fit(yy,m=m);curve=periodic_design(np.arange(512)/512,m=m)@result['controls'];fitted_rings.append((ring['role'],curve));controls_count+=m;valid&=result['valid']
    if mode=='v2_16':
     mask=np.zeros_like(target,dtype=np.uint8);cv2.fillPoly(mask,[np.rint(fitted_rings[0][1]).astype(np.int32)],1)
    else:
     model_rings=[dict(rr,points=curve.tolist()) for rr,(_,curve) in zip(chosen,fitted_rings)]
     if len(model_rings)==1:model_rings[0].update(parent_ring=-1,depth=0)
     mask=render_geometry(model_rings,target.shape)
    gm=geometry_metrics(target,mask);bf=mask-cv2.erode(mask,np.ones((3,3),np.uint8));df=cv2.distanceTransform(1-bf,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);precision=float(np.mean(dst[bf>0]<=tol)) if bf.any() else 0.;recall=float(np.mean(df[bs>0]<=tol)) if bs.any() else 0.;records.append(dict(video_id=v,split=split(v),mask_uid=uid,mode=mode,entity_name=r['entity_name'],multi_component=r['component_count']>1,holes=r['hole_count'],control_points=controls_count,loop_count=len(fitted_rings),represented_rings=len(fitted_rings),source_rings=len(rings),fit_valid=bool(valid),fscore_scale=2*precision*recall/max(precision+recall,1e-20),fit_seconds=time.perf_counter()-started,**gm))
    if len(previews)<20:
     image=np.zeros((height,width,3),np.uint8);image[target>0]=(90,90,90)
     for role,c in fitted_rings:cv2.polylines(image,[np.rint(c).astype(np.int32)],True,(50,255,230) if role=='outer' else (255,180,60),2)
     from PIL import Image
     im=Image.fromarray(image);im.thumbnail((480,270));previews.append((uid,mode,im.copy(),gm['iou']))
  from PIL import Image,ImageDraw
  sheet=Image.new('RGB',(2400,1200),(20,20,20));draw=ImageDraw.Draw(sheet)
  for i,(uid,mode,im,iou) in enumerate(previews):sheet.paste(im,(i%5*480,i//5*300));draw.text((i%5*480+6,i//5*300+275),f'{mode} IoU={iou:.3f}',fill='white')
  sheet.save(path('reports')/f'architecture_spatial.{v}.jpg',quality=92)
 save_index(path('reports')/f'architecture_spatial.{v}.parquet',records);return dict(video_id=v,masks=len(manual))


def spatial(monitor):
 cfg=config();pool(spatial_video,cfg['train']+cfg['validation'],monitor,workers=2);records=[]
 for v in cfg['train']+cfg['validation']:records+=pq.read_table(path('reports')/f'architecture_spatial.{v}.parquet').to_pylist()
 save_index(path('reports')/'architecture_spatial.parquet',records);summary={}
 for part in ['train','validation']:
  summary[part]={}
  for mode in sorted(set(r['mode'] for r in records)):
   vs={}
   for v in cfg[part]:
    rs=[r for r in records if r['video_id']==v and r['mode']==mode];vs[v]={k:float(np.mean([r[k] for r in rs])) for k in ['iou','boundary_fscore_2px','fscore_scale','symmetric_boundary_distance_px','fit_seconds','fit_valid']}
   summary[part][mode]=dict(**{k:float(np.mean([r[k] for r in vs.values()])) for k in next(iter(vs.values()))},by_video=vs)
 report('architecture_spatial.json',dict(metrics=summary,controls=[16,24,32],compound_budget=32,limits=['Source-contour reconstruction, not unseen segmentation. Same manual masks and source sampling for each variant.','32-control compound model allocates 32, 24+8, or 16+8+8 across the dominant ring and at most two additional outer/hole rings. Omitted rings remain counted.','Multi-ring temporal slot correspondence/presence is not yet implemented; this is a geometric distortion/cost ablation.','Validation participants have already been used for iterative research; no new external benchmark or final selection.'],state='architecture_ablation_not_finalized',**identity()))


def polynomial_fit(t,h,variance,prior,quality,window=1.,degree=3,strength=1/3,scale=None):
 from math import comb
 z=2*(t-(1-window))/window-1;phi=basis(z,degree=degree);q=degree+1;n,d=h.shape;dt=np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]]) if n>1 else np.ones(1);effective=np.maximum(variance,1e-12)+.01*scale[None]**2;middle=np.median(effective,axis=0);effective=np.clip(effective,middle/100,middle*100);w=dt[:,None]*prior[:,None]**.5*quality[:,None]/effective**.25;w*=n/np.maximum(w.sum(0),1e-20);penalty=np.array([0 if j<2 else comb(j,2)**2 for j in range(q)]);gram=np.einsum('ni,nd,nj->dij',phi,w,phi)+np.diag(penalty*strength+1e-8)[None];rhs=np.einsum('ni,nd,nd->di',phi,w,h);chol=np.linalg.cholesky(gram);c=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,rhs[...,None]))[...,0].T;cov=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,np.broadcast_to(np.eye(q),(d,q,q))));residual=h-phi@c;sigma=np.maximum(np.sum(w*residual**2,axis=0)/max(n-q,1),1e-10);cov*=sigma[:,None,None];return dict(coefficients=c,covariance=cov,condition=float(np.linalg.cond(gram).max()),neff=float(np.min(w.sum(0)**2/np.maximum(np.sum(w*w,axis=0),1e-30))),window=window,degree=degree)


def temporal(monitor,videos=None,worker_mode=False):
 from refinement.data import temporal_cases,holdout
 from .temporal import GROUPS
 cfg=config();cal=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text());scale=np.asarray(cal['feature_scale']);trials=[(1.,3,1/3),(.5,3,.03),(.5,3,1/3),(.5,3,3.),(.25,3,.03),(.25,3,1/3),(1.,5,.03),(1.,5,1/3),(1.,5,3.),(.5,5,.03),(.5,5,1/3)];records=[];yield_records=[]
 for video in ([] if videos==[] else (videos or cfg['train']+cfg['validation'])):
  for case in temporal_cases(allowed([video])):
   t=case['time'];h=case['h']
   for window,degree,strength in trials:
    for protocol in ['single','gap100','endpoint100']:
     hidden=holdout(t,protocol,.875);keep=(~hidden)&(t>=1-window);eligible=hidden.any() and keep.sum()>=max(8,degree+2);yield_records.append(dict(video_id=video,split=split(video),window=window,degree=degree,regularization=strength,protocol=protocol,eligible=bool(eligible)))
     if not eligible:continue
     outputs=[]
     for method,w,d,reg,kp in [('paired_v2',1.,3,1/3,~hidden),('ablation',window,degree,strength,keep)]:
      fit=polynomial_fit(t[kp],h[kp],case['variance'][kp]*cal['kappa_cov'],case['prior'][kp],case['quality'][kp],w,d,reg,scale);pred=basis(2*(t[hidden]-(1-w))/w-1,degree=d)@fit['coefficients'];error=pred-h[hidden]
      for group,sl in GROUPS.items():records.append(dict(video_id=video,split=split(video),entity_track_id=case['entity_track_id'],window_end_us=case['window_end_us'],window_seconds=window,degree=degree,regularization=strength,method=method,protocol=protocol,group=group,normalised_mse=float(np.mean((error[:,sl]/scale[sl])**2)),condition=fit['condition'],input_observations=int(kp.sum()),coefficients_per_channel=d+1,collocation_nodes_required=d+1))
  monitor.check();LOG.info('Temporal architecture ablation %s',video)
 if videos==[]:
  for vv in cfg['train']+cfg['validation']:
   records+=pq.read_table(path('reports')/f'architecture_temporal.{vv}.parquet').to_pylist();yield_records+=pq.read_table(path('reports')/f'architecture_temporal_yield.{vv}.parquet').to_pylist()
 if worker_mode:
  video=videos[0];save_index(path('reports')/f'architecture_temporal.{video}.parquet',records);save_index(path('reports')/f'architecture_temporal_yield.{video}.parquet',yield_records);return dict(video_id=video,rows=len(records))
 save_index(path('reports')/'architecture_temporal.parquet',records);save_index(path('reports')/'architecture_temporal_yield.parquet',yield_records);summary={};grouped=defaultdict(list)
 for r in records:grouped[(r['video_id'],r['window_seconds'],r['degree'],r['regularization'],r['protocol'],r['method'],r['group'])].append(r)
 for window,degree,strength in trials:
  name=f'T{window:g}_D{degree}_R{strength:g}';summary[name]={}
  for protocol in ['single','gap100','endpoint100']:
   summary[name][protocol]={}
   for method in ['paired_v2','ablation']:
    summary[name][protocol][method]={}
    for group in GROUPS:
     byvideo={}
     for v in cfg['validation']:
      rs=grouped[(v,window,degree,strength,protocol,method,group)]
      if rs:byvideo[v]=dict(n=len(rs),normalised_rmse=float(np.sqrt(np.mean([r['normalised_mse'] for r in rs]))))
     if byvideo:summary[name][protocol][method][group]=dict(normalised_rmse=float(np.mean([r['normalised_rmse'] for r in byvideo.values()])),by_video=byvideo)
 report('architecture_temporal.json',dict(validation_metrics=summary,grids=[dict(window_seconds=w,degree=d,regularization=r,coefficient_dimension=[d+1,35],minimum_gauss_legendre_nodes=d+1,physical_derivative_scale=2/w) for w,d,r in trials],protocol='Same physical hidden targets within each comparison: single/gap centered at 0.875 of the original second; final 100ms endpoint. Each ablation and paired v2 receive identical target rows. Shorter windows discard older observations; unsupported cases are counted separately.',limits=['A shorter current state retains less history; fixed 0.25s output hop does not increase the state rate.','Degree five changes the coefficient and quadrature interface; six GL nodes integrate squared degree-five polynomials exactly.','Validation results are exploratory and not an independent final selection. Source annotations include interpolated states.'],state='architecture_ablation_not_finalized',**identity()))


def run(monitor):
 spatial(monitor);temporal_parallel(monitor)


def temporal_worker(video):
 class LocalMonitor:
  def check(self):
   from track1.artifacts import memory_guard
   memory_guard(config())
 return temporal(LocalMonitor(),videos=[video],worker_mode=True)

def temporal_parallel(monitor):
 cfg=config();pool(temporal_worker,cfg['train']+cfg['validation'],monitor,workers=7);temporal(monitor,videos=[])
