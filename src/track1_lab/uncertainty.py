"""Training-only local representation bands; no independent physical-geometry claim."""
from collections import defaultdict
import json
import numpy as np
from scipy.optimize import minimize
from pipeline.common import ROOT
from track1.artifacts import save_index,save_shard
from refinement.data import spatial_cases
from refinement.kernels import nearest_curve,periodic_design,spatial_errors
from refinement.protocol import rank
from refinement.diagnostics import reliability
from .common import config,path,allowed,report,pool,identity,LOG
from .spatial import fit


def variance_features(points,result,query,bandwidth=.06):
 p=result['controls'];_,u=nearest_curve(points,p,512);phi=periodic_design(u);tangent=periodic_design(u,derivative=1)@p;n=np.stack([tangent[:,1],-tangent[:,0]],axis=1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12);residual=np.sum((points-phi@p)*n,axis=1);_,q=nearest_curve(query,p,512);distance=np.minimum(abs(q[:,None]-u[None]),1-abs(q[:,None]-u[None]));w=np.exp(-.5*(distance/bandwidth)**2);local=(w@residual**2)/np.maximum(w.sum(1),1e-20);d=periodic_design(q);base=np.maximum(np.einsum('ni,ij,nj->n',d,result['covariance'],d),1e-12)
 return np.c_[base,local,np.ones(len(q))]


def collect(v):
 selected=json.loads((path('reports')/'spatial_candidate.json').read_text())['options'];records=[];features=[];errors=[]
 for c in spatial_cases(allowed([v])):
  if c['source_kind']!='manual_sparse':continue
  y=c['y'];mask=np.ones(128);mask[::4]=0;result=fit(y,selected,mask,c['geometry_weight']*c['provenance_weight'])
  if not result['valid']:continue
  target=y[::8];error,_,_=spatial_errors(target,result);x=variance_features(y[mask>0],result,target)
  for j,(e,z) in enumerate(zip(error,x)):records.append(dict(video_id=v,split=c['split'],mask_uid=c['mask_uid'],point_index=j,normal_error=float(e),base_variance=float(z[0]),local_residual_mse=float(z[1]),constant=1.))
 save_index(path('reports')/f'uncertainty_training.{v}.parquet',records);return dict(video_id=v,n=len(records))


def run(monitor):
 import pyarrow.parquet as pq
 cfg=config();pool(collect,cfg['train']+cfg['validation'],monitor,workers=7);rs=[]
 for v in cfg['train']+cfg['validation']:rs+=pq.read_table(path('reports')/f'uncertainty_training.{v}.parquet').to_pylist()
 x=np.array([[r['base_variance'],r['local_residual_mse'],1.] for r in rs]);e=np.array([r['normal_error'] for r in rs]);train=np.array([r['split']=='train' for r in rs]);normalizer=np.maximum(np.median(x[train],axis=0),1e-4);xx=x/normalizer
 def objective(beta):
  var=np.maximum(xx[train]@beta,1e-8);ee=e[train]**2;value=.5*np.mean(np.log(var)+ee/var);gradient=.5*np.mean(xx[train]*(1/var-ee/var**2)[:,None],axis=0);return value,gradient
 optimized=minimize(objective,np.array([1.,1.,.1]),method='L-BFGS-B',jac=True,bounds=[(1e-6,None)]*3);assert optimized.success,optimized.message;coeff=optimized.x/normalizer;v=x@coeff;factor=float(np.quantile(e[train]**2/np.maximum(v[train],1e-12),.95)/1.9599639845**2);coeff*=factor
 global_k=float(np.quantile(e[train]**2/np.maximum(x[train,0],1e-12),.95)/1.9599639845**2);evaluations={}
 for split in ['train','validation']:
  keep=np.array([r['split']==split for r in rs]);evaluations[split]=dict(global_only=reliability(e[keep],x[keep,0]*global_k),local_residual=reliability(e[keep],x[keep]@coeff))
 model=dict(global_kappa=global_k,local_coefficients=coeff.tolist(),feature_names=['conditional_coefficient_variance','observed_neighbour_normal_residual_mse','native_pixel_variance_floor'],bandwidth_periodic_u=.06,training_videos=cfg['train'],evaluations=evaluations,interpretation='Empirical contour-representation bands. Local residuals are computed only from retained source vertices; hidden calibration vertices never enter the feature estimator. Not label/physical-geometry uncertainty.',state='experimental_not_finalized',**identity());report('uncertainty_candidate.json',model);return model
