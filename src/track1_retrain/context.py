"""Targeted uncompressed six-second continuity supplement; locked Track 1 fits."""
import collections,json,time,shutil,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import atomic_json,digest
from .common import *
from .readiness import R,W
CTX=ROOT/'dataset/context_v3';DEST=ROOT/'derived/track1_context_v3'
for p in [DEST]:p.mkdir(exist_ok=True)

def stage_folder(name):
 p=DEST/name;p.mkdir(exist_ok=True);return p

def extract(v):
 import subprocess
 subprocess.run([sys.executable,'-m','track1_retrain.native_context',v],check=True)
 return json.loads((CTX/f'{v}.json').read_text())

class Frames:
 def __init__(self):
  from data_v2.store import Dataset
  self.base=Dataset();rows=pq.read_table(CTX/'extra_frames.parquet').to_pylist();self.lookup={(r['video_id'],r['timestamp_us']):r for r in rows};self.cache=collections.OrderedDict()
 def close(self):
  self.base.close()
  for a in self.cache.values():a._mmap.close()
  self.cache.clear()
 def get_frame(self,v,t):
  if (v,t) not in self.lookup:return self.base.get_frame(v,t)
  if v not in self.cache:
   if len(self.cache)>=2:self.cache.popitem(last=False)[1]._mmap.close()
   self.cache[v]=np.load(CTX/f'{v}.rgb.npy',mmap_mode='r')
  self.cache.move_to_end(v);return np.array(self.cache[v][self.lookup[(v,t)]['storage_row']],copy=True)

def materialize(monitor):
 CTX.mkdir(exist_ok=True);extra=pq.read_table(W/'context_frames.parquet').to_pylist();episodes=pq.read_table(W/'context_episodes.parquet').to_pylist();save(CTX/'extra_frames.parquet',extra)
 for e in episodes:e['episode_id']=e['context_id']
 save(CTX/'episodes.parquet',episodes)
 needed=sum(r['new_rgb_bytes'] for r in episodes if not ((CTX/f"{r['video_id']}.json").exists() and (CTX/f"{r['video_id']}.rgb.npy").exists()))
 if shutil.disk_usage(ROOT).free<needed+CFG['disk_reserve_bytes']:raise OSError('Insufficient space for context supplement and reserve')
 results=pool(extract,VIDEOS,monitor,workers=2)
 # View index references existing v2 payloads rather than duplicating central frames.
 base=pq.read_table(ROOT/'dataset/partition_v2/frames.parquet').to_pylist();allframes={}
 for origin,rows in [('base',base),('context',extra)]:
  for r in rows:allframes[(r['video_id'],r['native_frame_uid'])]=dict(r,storage_origin=origin)
 view=[]
 for r in allframes.values():
  if any(e['video_id']==r['video_id'] and e['start_us']<=r['timestamp_us']<e['end_us'] for e in episodes):view.append(r)
 view.sort(key=lambda r:(r['video_id'],r['timestamp_us']));save(CTX/'frames.parquet',view);save(R/'context_views.parquet',episodes)
 plan=json.loads((R/'context_plan.json').read_text());plan.update(materialized=True,shards=results,context_view_frames=len(view),extra_frames_sha256=digest(CTX/'extra_frames.parquet'),frames_sha256=digest(CTX/'frames.parquet'));atomic_json(R/'context_plan.json',plan)


def input_rows(v):
 import duckdb
 with duckdb.connect() as db:
  db.execute('SET threads=1');db.execute("SET memory_limit='1GB'")
  cur=db.execute('''SELECT s.mask_uid,s.entity_track_id,s.timestamp_us,s.identity_evidence,s.row_index AS baseline_row_index,
      m.source_kind,m.geometry_status,m.area_px,m.centroid_x,m.centroid_y,m.annotation_width,m.annotation_height,
      m.contours_json,m.segments_raw,m.component_count,m.hole_count,s.width,s.height,
      m.dependency_group,m.available_at_us,m.online_observation_allowed
      FROM read_parquet(?) s JOIN read_parquet(?) m USING(mask_uid)
      JOIN read_parquet(?) f ON f.video_id=s.video_id AND f.timestamp_us=s.timestamp_us
      ORDER BY s.entity_track_id,s.timestamp_us,s.mask_uid''',[str(ROOT/f'canonical/v2/spatial_metadata.{v}.parquet'),str(ROOT/f'canonical/v2/masks.{v}.parquet'),str(CTX/'frames.parquet')]);names=[d[0] for d in cur.description];return [dict(zip(names,r),video_id=v,split=split(v)) for r in cur.fetchall()]

def spatial(monitor):
 from .spatial import worker,align
 jobs=[];reused={};inputs={}
 for v in VIDEOS:
  new=input_rows(v);inputs[v]=new;save(W/f'inputs.{v}.parquet',new);old=pq.read_table(folder('spatial')/f'{v}.parquet').to_pylist();wanted={r['mask_uid'] for r in new};arrays={k:np.load(folder('spatial')/f'{v}.{k}.npy',mmap_mode='r') for k in ['h16','variance16','controls16','h32','variance32','controls32']};reused[v]={}
  for r in old:
   if r['mask_uid'] in wanted:reused[v][r['mask_uid']]=dict(r,**{k:np.array(a[r['row_index']]).tolist() for k,a in arrays.items()})
  for a in arrays.values():a._mmap.close()
  fresh=[r for r in new if r['mask_uid'] not in reused[v]]
  for lo in range(0,len(fresh),CFG['chunk_size']):
   chunk=fresh[lo:lo+CFG['chunk_size']];jobs.append((v,900000+lo//CFG['chunk_size'],chunk,dict(scope='context_v3',source_sha256=digest(CTX/'frames.parquet'),uids=hashlib.sha256(''.join(r['mask_uid'] for r in chunk).encode()).hexdigest(),code=spatial_identity())))
  LOG.info('Context %s: %d observations; reuse=%d refit=%d',v,len(new),len(reused[v]),len(fresh))
 results=pool(worker,jobs,monitor);dest=stage_folder('spatial');summary=[]
 for v in VIDEOS:
  lookup=dict(reused[v])
  for result in [r for r in results if r['video_id']==v]:
   for r in pq.read_table(WORK/f'spatial_chunks/{v}.{result["chunk"]:05d}.parquet').to_pylist():lookup[r['mask_uid']]=r
  rows=align([lookup[r['mask_uid']] for r in inputs[v]])
  for name in ['h16','variance16','controls16','h32','variance32','controls32']:array(dest/f'{v}.{name}.npy',[r.pop(name) for r in rows])
  for i,r in enumerate(rows):r['row_index']=i
  save(dest/f'{v}.parquet',rows);summary.append(dict(video_id=v,observations=len(rows),reused=len(reused[v]),refitted=len(rows)-len(reused[v])))
 atomic_json(R/'context_spatial.json',dict(videos=summary,pointwise_fit_method_unchanged=True,causal_gauge_recomputed_on_context=True))


def appearance(monitor):
 import torch,cv2
 from track1.appearance import backbone
 from pipeline.geometry import rasterize
 torch.set_num_threads(2);cv2.setNumThreads(1);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=backbone().to(device);mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];dest=stage_folder('appearance');reader=Frames();summary=[]
 means=np.load(folder('appearance')/'mean.npy');components=np.load(folder('appearance')/'components.npy')
 def resize(a):
  factor=min(224/a.shape[1],224/a.shape[0]);w=max(1,round(a.shape[1]*factor));h=max(1,round(a.shape[0]*factor));canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[(224-h)//2:(224-h)//2+h,(224-w)//2:(224-w)//2+w]=cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA);return canvas
 try:
  for v in VIDEOS:
   rows=pq.read_table(W/f'inputs.{v}.parquet').to_pylist();oldmeta=pq.read_table(folder('appearance')/f'{v}.parquet').to_pylist();oldidx={r['mask_uid']:r['row_index'] for r in oldmeta};old=np.load(folder('appearance')/f'{v}.features.npy',mmap_mode='r');a=np.empty((len(rows),2,576),np.float32);todo=collections.defaultdict(list);reused=0
   for i,r in enumerate(rows):
    if r['mask_uid'] in oldidx:a[i]=old[oldidx[r['mask_uid']]];reused+=1
    else:todo[r['timestamp_us']].append((i,r))
   old._mmap.close();batch=[];indices=[]
   def flush():
    if not batch:return
    tensor=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).to(device,dtype=torch.float32)/255
    with torch.inference_mode():z=model((tensor-mean)/std).mean((-1,-2)).cpu().numpy().reshape(-1,2,576)
    a[indices]=z;batch.clear();indices.clear();monitor.check()
   for count,(timestamp,items) in enumerate(sorted(todo.items())):
    rgb=reader.get_frame(v,timestamp)
    for i,r in items:
     mask=rasterize(json.loads(r['segments_raw']),r['annotation_width'],r['annotation_height']);yy,xx=np.nonzero(mask);sx=r['width']/r['annotation_width'];sy=r['height']/r['annotation_height'];x0=max(0,int(np.floor(xx.min()*sx))-8);x1=min(r['width'],int(np.ceil((xx.max()+1)*sx))+8);y0=max(0,int(np.floor(yy.min()*sy))-8);y1=min(r['height'],int(np.ceil((yy.max()+1)*sy))+8);xs=np.clip(np.floor(np.arange(x0,x1)/sx).astype(int),0,mask.shape[1]-1);ys=np.clip(np.floor(np.arange(y0,y1)/sy).astype(int),0,mask.shape[0]-1);foreground=mask[np.ix_(ys,xs)].astype(bool);crop=rgb[y0:y1,x0:x1];obj=crop.copy();obj[~foreground]=[124,116,104];batch.extend([resize(obj),resize(crop)]);indices.append(i)
     if len(batch)>=16:flush()
    if count%64==0:reader.close()
    if count%256==0:LOG.info('Context features %s frames=%d/%d',v,count,len(todo))
   flush();assert np.isfinite(a).all();array(dest/f'{v}.features.npy',a);projection=np.einsum('njc,jcd->njd',a-means,components);array(dest/f'{v}.projected.npy',projection);save(dest/f'{v}.parquet',[{k:r[k] for k in ['video_id','split','mask_uid','entity_track_id','timestamp_us','source_kind','dependency_group']}|dict(row_index=i) for i,r in enumerate(rows)]);summary.append(dict(video_id=v,observations=len(rows),reused_features=reused,new_features=len(rows)-reused))
 finally:reader.close()
 atomic_json(R/'context_appearance.json',dict(videos=summary,backbone_frozen=True,pca_frozen_from_training_only_v2=True,device=str(device)))


def build(monitor):
 from .temporal import build_worker
 cal=stage_folder('calibration');source=folder('calibration')/'model.json';(cal/'model.json').write_text(source.read_text());episodes=str(CTX/'episodes.parquet');settings=dict(root=str(DEST),episodes=episodes,reports=str(R),scope='context_v3_with_locked_v2_parameters')
 jobs=pool(build_worker,[(v,n,settings) for v in VIDEOS for n in [16,32]],monitor,workers=7);atomic_json(R/'context_functional.json',dict(videos=jobs,parameters_locked=True,parameters_sha256=digest(source),covariance_calibration_not_revalidated=True))

def verify(monitor):
 from .store import sequences,Dataset
 from track1_lab.variants import collocation
 from track1_lab.topology_certificate import input_hashes
 import pyarrow as pa
 allrows=[];summary=[]
 with Dataset(root=DEST) as ds:
  for v in VIDEOS:
   for n in [16,32]:
    rows=pq.read_table(stage_folder('functional')/f'{v}.m{n}.parquet').to_pylist();allrows+=rows;checked=0
    arrays={name:np.load(stage_folder('functional')/f'{v}.m{n}.{name}.npy',mmap_mode='r') for name in ['coefficients','coefficient_cov','geometry_coefficients','xi','xi_dot','detail','detail_transform','appearance_coefficients','kinematic_position','kinematic_velocity','kinematic_cov']}
    for a in arrays.values():assert np.isfinite(a).all()
    eigen=np.linalg.eigvalsh(arrays['coefficient_cov'].astype(float));assert np.all(eigen[...,0]>=-1e-5*np.maximum(abs(eigen).max(-1),1e-20))
    for r in rows:
     if not r['valid']:continue
     i=r['row_index'];nodes=collocation(arrays['coefficients'][i].astype(float),1.);np.testing.assert_allclose(nodes['xi'],arrays['xi'][i],atol=2e-5,rtol=1e-4);np.testing.assert_allclose(nodes['xi_dot'],arrays['xi_dot'][i],atol=2e-4,rtol=1e-4);assert r['latest_input_us']<=r['window_end_us']
     if r['certified_geometry_valid']:
      certificate=json.loads(r['certificate_json']);expected=input_hashes(arrays['geometry_coefficients'][i],r['allocation'],r['roles'],r['parents'],1.);assert all(certificate[k]==value for k,value in expected.items())
     from track1_lab.history_detail import detail_basis
     assert np.max(abs(detail_basis(np.array([1.]))@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     assert np.max(abs(detail_basis(np.array([1.]),1)@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     if r['kinematics_valid']:
      assert r['kinematics_latest_input_us']<=r['window_end_us']
      eig=np.linalg.eigvalsh(arrays['kinematic_cov'][i].astype(float));assert eig[0]>=-1e-6*max(abs(eig).max(),1e-20)
     checked+=1
    if any(r['valid'] for r in rows):
     r=next(r for r in rows if r['valid']);ds.get_state(v,r['entity_track_id'],r['window_end_us'],model=n)
    summary.append(dict(video_id=v,split=split(v),model=n,attempts=len(rows),valid=checked,complete_certified=sum(r['geometry_training_eligible'] for r in rows)))
    for a in arrays.values():a._mmap.close()
    monitor.check()
 schema=pa.schema([('video_id',pa.string()),('split',pa.string()),('entity_track_id',pa.string()),('model',pa.int64()),('sequence_length',pa.int64()),('row_indices',pa.list_(pa.int64())),('end_times_us',pa.list_(pa.int64())),('all_geometry_valid',pa.bool_()),('identity_verified',pa.bool_())]);counts=[];indices=stage_folder('indices')
 for length in [4,8,16]:
  seq=sequences(allrows,length)
  for v in VIDEOS:
   for n in [16,32]:
    local=[r for r in seq if r['video_id']==v and r['model']==n];counts.append(dict(video_id=v,split=split(v),model=n,length=length,sequences=len(local),complete_certified_sequences=sum(r['all_geometry_valid'] for r in local),tracks=len({r['entity_track_id'] for r in local})))
  for part in ['train','validation']:pq.write_table(pa.Table.from_pylist([r for r in seq if r['split']==part],schema=schema),indices/f'{part}.length{length}.parquet',compression=None)
 save(R/'context_continuity.parquet',counts);atomic_json(R/'context_results.json',dict(videos=summary,continuity=counts,numerical_checks_passed=True,independent_identity_verified=False,scientific_acceptance=False,track2_started=False))
 atomic_json(DEST/'status.json',dict(ready=True,scope='targeted six-second source contexts; numerical Track 1 state/sequence ingestion',finalized=False,scientific_acceptance=False,track2_started=False,parameters_locked_from_v2=True,covariance_revalidation_pending=True))


def run(monitor):
 for name,fn in [('materialize',materialize),('spatial',spatial),('appearance',appearance),('functional',build),('verify',verify)]:
  LOG.info('BEGIN context %s',name);fn(monitor);LOG.info('COMPLETE context %s',name)
