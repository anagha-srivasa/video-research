"""Resumable working spatial store. No release/finalization path is provided."""
from collections import defaultdict
import json,time,logging
import numpy as np
import pyarrow.parquet as pq
import duckdb
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_shard,save_index,memory_guard
from track1.spatial import dominant_ring,resample,wavelet_state
from track1.camera import undistort
from track1.numerics import wavelet_matrix
from refinement.kernels import periodic_design,nearest_curve
from .geometry_contract import dominant_ring
from .common import config,path,allowed,Monitor,pool,LOG,report
from .spatial import fit,closest_polyline
from .uncertainty import variance_features
_CONTEXT={}


def directory():
 p=path('derived')/'spatial';p.mkdir(exist_ok=True);return p

def chunk(job):
 v,begin,end=job;allowed([v]);cfg=config();memory_guard(cfg);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True);spec=json.loads((path('work')/'spatial_build_spec.json').read_text());base=ROOT/'derived/track1_refined_v2/spatial'
 if _CONTEXT.get('video')!=v:
  _CONTEXT.clear();old=load_shard(base/f'{v}.pt');rows=pq.read_table(base/f'{v}.parquet').to_pylist();camera=json.loads((base/f'{v}.camera.json').read_text())
  with duckdb.connect() as db:
   db.execute('SET threads=1');geometry={uid:(rings,aw,ah) for uid,rings,aw,ah in db.execute('SELECT mask_uid,contours_json,annotation_width,annotation_height FROM read_parquet(?) WHERE timestamp_us IS NOT NULL AND geometry_status!=?',[str(ROOT/f'canonical/v1/masks.{v}.parquet'),'quarantined']).fetchall()}
  _CONTEXT.update(video=v,old=old,rows=rows,camera=camera,geometry=geometry)
 old=_CONTEXT['old'];rows=[dict(r) for r in _CONTEXT['rows'][begin:end]];camera=_CONTEXT['camera'];geometry=_CONTEXT['geometry'];arrays={k:np.empty((len(rows),)+a.shape[1:],a.dtype) for k,a in old.items()};arrays['normal_variance_representation']=np.empty((len(rows),64),np.float32);phi=periodic_design(np.arange(64)/64);dp=periodic_design(np.arange(64)/64,derivative=1);meanphi=phi.mean(0);k=np.asarray(camera['intrinsic_native']);dist=np.asarray(camera['distortion']);w=wavelet_matrix();fallback=0;rescued=0;shifts=0;started=time.monotonic()
 for j,r in enumerate(rows):
  i=r['row_index']
  if j%256==0:memory_guard(cfg);LOG.info('%s spatial %d/%d',v,i,end)
  rings,aw,ah=geometry[r['mask_uid']];ring,components,holes=dominant_ring(dict(contours_json=rings),np.diag([r['width']/aw,r['height']/ah,1.]));y=np.roll(resample(ring,128),-2*r['seam_shift'],axis=0);result=fit(y,spec['spatial_options'],reliability=r['geometry_weight']*r['provenance_weight']);p=result['controls'];cov=result['covariance'];a,_=nearest_curve(y,old['controls'][i]);b,_=nearest_curve(y,p);use_old=bool(r['spatial_valid'] and (not result['valid'] or np.mean(b*b)>np.mean(a*a)))
  if use_old:p=old['controls'][i].astype(float);cov=old['covariance'][i].astype(float);fallback+=1
  valid=bool(r['spatial_valid'] or result['valid']);rescued+=int(valid and not r['spatial_valid'])
  # Exact cyclic control permutation preserves geometry and the old temporal seam gauge.
  shift=int(np.argmin([np.sum((np.roll(p,-q,axis=0)-old['controls'][i])**2) for q in range(16)]));order=np.roll(np.arange(16),-shift);p=p[order];cov=cov[np.ix_(order,order)];shifts+=int(shift!=0)
  centre=old['centre_native'][i].astype(float);scale=float(old['scale_native'][i]);area=np.pi*scale**2;z,zcov=wavelet_state(p,cov,centre,scale);cn=undistort(centre[None],k,dist)[0];h=np.r_[cn,z.reshape(-1),np.log(scale/np.sqrt(k[0,0]*k[1,1]))];centrevar=float(meanphi@cov@meanphi);curve=phi@p;gx=(np.roll(curve[:,1],-1)-np.roll(curve[:,1],1))/2;gy=(np.roll(curve[:,0],1)-np.roll(curve[:,0],-1))/2;ax=phi.T@gx;ay=phi.T@gy;logvar=float((ax@cov@ax+ay@cov@ay)/(4*max(area*area,1)));shapevar=np.repeat(np.diag(zcov),2)+np.repeat((w@np.ones(16))**2*centrevar/scale**2,2)+z.reshape(-1)**2*logvar;variance=np.r_[centrevar/k[0,0]**2,centrevar/k[1,1]**2,shapevar,logvar];tangent=dp@p;normal=np.c_[tangent[:,1],-tangent[:,0]];normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12);near,_=closest_polyline(curve,y);representation=variance_features(y,dict(controls=p,covariance=cov),curve)@np.asarray(spec['uncertainty']['local_coefficients'])
  values=dict(controls=p,covariance=cov,h=h,variance=variance,boundary=old['boundary'][i],normals=normal,normal_variance=np.maximum(np.einsum('ni,ij,nj->n',phi,cov,phi),0),normal_variance_representation=representation,residual_normal=np.sum((near-curve)*normal,axis=1),centre_native=centre,scale_native=np.asarray(scale),world_bearing=old['world_bearing'][i])
  for key,value in values.items():arrays[key][j]=value
  r.update(hole_count=holes,component_count=components)
  r.update(v2_spatial_valid=r['spatial_valid'],spatial_valid=valid,lab_fallback=use_old,lab_phase_control_shift=shift,lab_fit_rmse_px=float(np.sqrt(np.mean((a if use_old else b)**2))),residual_normal_metric='nearest_input_polyline_at_curve_parameters',representation_covariance_kind='train_only_local_normal_representation_band_separate_from_coefficient_covariance')
 for key in ['controls','covariance','h','variance','normal_variance_representation']:assert np.isfinite(arrays[key]).all()
 out=path('work')/'spatial_chunks';out.mkdir(exist_ok=True);stem=f'{v}.{begin:08d}';save_shard(out/f'{stem}.pt',arrays);save_index(out/f'{stem}.parquet',rows);metrics=dict(video_id=v,begin=begin,end=end,n=len(rows),valid=sum(r['spatial_valid'] for r in rows),fallback=fallback,rescued=rescued,phase_permutations=shifts,seconds=time.monotonic()-started,spec_sha256=digest(path('work')/'spatial_build_spec.json'),tensor_sha256=digest(out/f'{stem}.pt'),index_sha256=digest(out/f'{stem}.parquet'));atomic_json(out/f'{stem}.json',metrics);return metrics


def assemble(v):
 out=path('work')/'spatial_chunks';markers=sorted(out.glob(v+'.*.json'));records=[json.loads(p.read_text()) for p in markers];assert records and records[0]['begin']==0
 for a,b in zip(records,records[1:]):assert a['end']==b['begin']
 n=pq.read_metadata(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').num_rows;assert records[-1]['end']==n;first=load_shard(markers[0].with_suffix('.pt'));arrays={k:np.empty((n,)+a.shape[1:],a.dtype) for k,a in first.items()};rows=[]
 for marker,r in zip(markers,records):
  assert r['spec_sha256']==digest(path('work')/'spatial_build_spec.json') and r['tensor_sha256']==digest(marker.with_suffix('.pt')) and r['index_sha256']==digest(marker.with_suffix('.parquet'));a=load_shard(marker.with_suffix('.pt'))
  for key,value in a.items():arrays[key][r['begin']:r['end']]=value
  rows+=pq.read_table(marker.with_suffix('.parquet')).to_pylist()
 dest=directory();save_shard(dest/f'{v}.pt',arrays);save_index(dest/f'{v}.parquet',rows);atomic_json(dest/f'{v}.camera.json',json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text()));summary={key:sum(r[key] for r in records) for key in ['n','valid','fallback','rescued','phase_permutations','seconds']};atomic_json(dest/f'{v}.manifest.json',dict(state='working_not_finalized',spec_sha256=digest(path('work')/'spatial_build_spec.json'),tensor_sha256=digest(dest/f'{v}.pt'),index_sha256=digest(dest/f'{v}.parquet'),metrics=summary));return summary


def run(monitor):
 cfg=config();specpath=path('work')/'spatial_build_spec.json';spec=dict(spatial_options=json.loads((path('reports')/'spatial_candidate.json').read_text())['options'],uncertainty=json.loads((path('reports')/'uncertainty_candidate.json').read_text()),spatial_kernel_sha256=digest(ROOT/'src/track1_lab/spatial.py'),uncertainty_kernel_sha256=digest(ROOT/'src/track1_lab/uncertainty.py'),builder_sha256=digest(ROOT/'src/track1_lab/build.py'),v2_freeze_sha256=digest(ROOT/'reports/track1_refinement/freeze_manifest.json'))
 if specpath.exists():assert json.loads(specpath.read_text())==spec,'Build inputs changed; keep the existing working build separate'
 else:atomic_json(specpath,spec)
 videos=sorted(cfg['train']+cfg['validation'],key=lambda v:-pq.read_metadata(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').num_rows)
 for v in videos:
  n=pq.read_metadata(ROOT/f'derived/track1_refined_v2/spatial/{v}.parquet').num_rows;jobs=[]
  for begin in range(0,n,1024):
   end=min(begin+1024,n);marker=path('work')/'spatial_chunks'/f'{v}.{begin:08d}.json';reuse=False
   if marker.exists():
    r=json.loads(marker.read_text());reuse=r['end']==end and r['spec_sha256']==digest(specpath) and r['tensor_sha256']==digest(marker.with_suffix('.pt')) and r['index_sha256']==digest(marker.with_suffix('.parquet'))
   if not reuse:jobs.append((v,begin,end))
  if jobs:pool(chunk,jobs,monitor,workers=12)
  LOG.info('WORKING SPATIAL %s %s',v,assemble(v));monitor.check()
 report('working_store.json',dict(state='open',finalized=False,ingestion_ready=False,videos=videos,note='Train/validation spatial candidate only; remaining appearance/temporal candidate selection is open. Frozen releases untouched.'))
