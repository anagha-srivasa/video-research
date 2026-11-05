"""Matched fixed-16 and compound-32 refits, explicit source attempts and epochs."""
import collections,json,time,itertools
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import digest,atomic_json
from track1.spatial import resample
from track1.camera import undistort
from track1.numerics import wavelet_matrix
from track1_lab.architecture import general_fit
from track1_lab.geometry_contract import normalize,select,topology_signature
from refinement.kernels import periodic_design
from .common import *

def inputs(v):
 import duckdb
 with duckdb.connect() as db:
  db.execute('SET threads=1');db.execute("SET memory_limit='1GB'")
  cur=db.execute('''SELECT s.mask_uid,s.entity_track_id,s.timestamp_us,s.identity_evidence,s.row_index AS baseline_row_index,
      m.source_kind,m.geometry_status,m.area_px,m.centroid_x,m.centroid_y,m.annotation_width,m.annotation_height,
      m.contours_json,m.segments_raw,m.component_count,m.hole_count,s.width,s.height,
      m.dependency_group,m.available_at_us,m.online_observation_allowed
      FROM read_parquet(?) s JOIN read_parquet(?) m USING(mask_uid)
      JOIN read_parquet(?) f ON f.video_id=s.video_id AND f.timestamp_us=s.timestamp_us
      ORDER BY s.entity_track_id,s.timestamp_us,s.mask_uid''',[str(ROOT/f'canonical/v2/spatial_metadata.{v}.parquet'),str(ROOT/f'canonical/v2/masks.{v}.parquet'),str(ROOT/'dataset/partition_v2/frames.parquet')])
  names=[d[0] for d in cur.description];rows=[dict(zip(names,r),video_id=v,split=split(v)) for r in cur.fetchall()]
  counts=db.execute('''SELECT m.source_kind,m.geometry_complete,m.partial_geometry_valid,count(*) n FROM read_parquet(?) m
      JOIN read_parquet(?) f USING(video_id,timestamp_us) GROUP BY ALL''',[str(ROOT/f'canonical/v2/masks.{v}.parquet'),str(ROOT/'dataset/partition_v2/frames.parquet')]).fetchall()
 save(REPORT/f'source_attempts.{v}.parquet',[dict(video_id=v,source_kind=k,complete=bool(c),partial_available=bool(p),observations=n) for k,c,p,n in counts])
 return rows

def feature(controls,cov,centre,scale,position,scale_n,pitch,focal):
 out=[];variance=[]
 for p,c in zip(controls,cov):
  w=wavelet_matrix(len(p));out.extend((w@((p-centre)/scale)).ravel());variance.extend(np.full(2*len(p),max(float(np.linalg.eigvalsh(c).max()),1e-10)/scale**2))
 # Annotation-grid noise floor is a conditional engineering approximation.
 position_var=(pitch/np.asarray(focal))**2
 return np.r_[position,out,np.log(max(scale_n,1e-12))],np.maximum(np.r_[position_var,variance,(pitch/max(scale,1))**2],1e-10)

def worker(job):
 v,chunk,rows,fingerprint=job;dest=WORK/'spatial_chunks';dest.mkdir(exist_ok=True);path=dest/f'{v}.{chunk:05d}.parquet';marker=path.with_suffix('.json')
 if marker.exists() and path.exists():
  m=json.loads(marker.read_text())
  if m['inputs']==fingerprint and m['output_sha256']==digest(path):return m
 camera=json.loads((ROOT/f'derived/track1_refined_v2/spatial/{v}.camera.json').read_text());k=np.asarray(camera['intrinsic_native']);dist=np.asarray(camera['distortion']);results=[];started=time.monotonic()
 for row in rows:
  rings=normalize(row['contours_json']);chosen=select(rings,3);main=max((r for r in rings if r['role']=='outer' and len(r['points'])>=3),key=lambda r:abs(r['signed_area']));native=np.array([row['width']/row['annotation_width'],row['height']/row['annotation_height']]);centre=np.array([row['centroid_x'],row['centroid_y']])*native;scale=np.sqrt(row['area_px']*native.prod()/np.pi);position=undistort(centre[None],k,dist)[0];scale_n=scale/np.sqrt(k[0,0]*k[1,1]);pitch=float(native.max());quality=.85 if row['geometry_status']=='repaired_for_rasterisation' else .95 if row['geometry_status']=='clipped_for_rasterisation' else 1.
  fit16=general_fit(resample(np.asarray(main['points'])*native,CFG['spatial_samples']),m=16)
  p16=fit16['controls'];c16=fit16['covariance']*pitch**2
  h16,var16=feature([p16],[c16],centre,scale,position,scale_n,pitch,[k[0,0],k[1,1]])
  allocation=[32] if len(chosen)==1 else [24,8] if len(chosen)==2 else [16,8,8];ps=[];cs=[];valid=True
  for ring,m in zip(chosen,allocation):
   f=general_fit(resample(np.asarray(ring['points'])*native,CFG['spatial_samples']),m=m);ps.append(f['controls']);cs.append(f['covariance']*pitch**2);valid&=f['valid']
  h32,var32=feature(ps,cs,centre,scale,position,scale_n,pitch,[k[0,0],k[1,1]])
  base={k:row[k] for k in ['video_id','split','mask_uid','entity_track_id','timestamp_us','source_kind','identity_evidence','baseline_row_index','width','height','component_count','hole_count','dependency_group']}
  base.update(h16=h16.tolist(),variance16=var16.tolist(),controls16=p16.tolist(),h32=h32.tolist(),variance32=var32.tolist(),controls32=np.concatenate(ps).tolist(),allocation=allocation,roles=[r['role'] for r in chosen],parents=[r['parent_ring'] for r in chosen],source_signature=json.dumps(topology_signature(rings)),source_rings=len(rings),represented_rings=len(chosen),source_complete=len(chosen)==len(rings),fit16_valid=bool(fit16['valid']),fit32_valid=bool(valid),quality=quality,prior=CFG['source_kind_priors'][row['source_kind']],centre_native=centre.tolist(),scale_native=float(scale),position_semantics='visible_mask_centroid',online_observation_allowed=False,covariance_calibrated=False)
  results.append(base)
 save(path,results);m=dict(video_id=v,chunk=chunk,observations=len(rows),seconds=time.monotonic()-started,inputs=fingerprint,output_sha256=digest(path));atomic_json(marker,m);return m

def align(rows):
 previous={};epochs={16:0,32:0};last_track=None;last_time=None
 for r in rows:
  if r['entity_track_id']!=last_track or last_time is None or r['timestamp_us']-last_time>350000:
   previous={};epochs={n:e+1 for n,e in epochs.items()}
  for n in [16,32]:
   ps=np.asarray(r[f'controls{n}']);allocation=[16] if n==16 else r['allocation'];roles=['outer'] if n==16 else r['roles'];parents=[-1] if n==16 else r['parents'];signature='single_outer' if n==16 else r['source_signature'];current=[];at=0
   for m in allocation:current.append(ps[at:at+m]);at+=m
   prior=previous.get(n);ambiguous=False;signature_changed=prior is not None and prior['signature']!=signature
   if prior is not None and prior['signature']==signature and prior['allocation']==allocation:
    orders=[]
    for order in itertools.permutations(range(len(current))):
     if any(roles[j]!=prior['roles'][i] or len(current[j])!=len(prior['controls'][i]) for i,j in enumerate(order)):continue
     remap={j:i for i,j in enumerate(order)}
     if any(remap.get(parents[j],-1)!=prior['parents'][i] for i,j in enumerate(order)):continue
     cost=sum(np.linalg.norm((current[j].mean(0)-r['centre_native'])/r['scale_native']-prior['relative'][i]) for i,j in enumerate(order));orders.append((cost,order))
    orders.sort();ambiguous=not orders or len(orders)>1 and orders[1][0]-orders[0][0]<.05
    if not ambiguous:
     order=orders[0][1];current=[current[j] for j in order];roles=[roles[j] for j in order];remap={j:i for i,j in enumerate(order)};parents=[remap.get(parents[j],-1) for j in order]
     # Reorder wavelet uncertainty channels with each matched ring.
     starts=np.cumsum([2]+[2*m for m in allocation]);oldvar=np.asarray(r[f'variance{n}']);varparts=[oldvar[starts[j]:starts[j+1]] for j in order];r[f'variance{n}']=np.r_[oldvar[:2],np.concatenate(varparts),oldvar[-1]].tolist()
     for i,p in enumerate(current):
      old=prior['controls'][i];a=(p-p.mean(0))/max(np.linalg.norm(p-p.mean(0)),1e-12);b=(old-old.mean(0))/max(np.linalg.norm(old-old.mean(0)),1e-12);shift=int(np.argmin([np.sum((np.roll(a,-j,axis=0)-b)**2) for j in range(len(p))]));current[i]=np.roll(p,-shift,axis=0)
   if signature_changed or ambiguous:epochs[n]+=1
   shape=np.concatenate([(wavelet_matrix(len(p))@((p-r['centre_native'])/r['scale_native'])).ravel() for p in current]);h=np.asarray(r[f'h{n}']);h[2:-1]=shape;r[f'h{n}']=h.tolist();r[f'controls{n}']=np.concatenate(current).tolist();r[f'epoch{n}']=f'{r["entity_track_id"]}:{epochs[n]}:{signature}';r[f'correspondence_ambiguous{n}']=ambiguous
   previous[n]=dict(signature=signature,allocation=allocation,roles=roles,parents=parents,controls=current,relative=[(p.mean(0)-r['centre_native'])/r['scale_native'] for p in current])
   if n==32:r['roles']=roles;r['parents']=parents
  last_track=r['entity_track_id'];last_time=r['timestamp_us']
 return rows

def run(monitor):
 jobs=[];sources={};fingerprint=spatial_identity()
 for v in VIDEOS:
  rows=inputs(v);sources[v]=dict(source_rows=len(rows),source_sha256=digest(ROOT/f'canonical/v2/masks.{v}.parquet'));LOG.info('Spatial source %s: %d eligible observations',v,len(rows))
  save(WORK/f'inputs.{v}.parquet',rows)
  for i in range(0,len(rows),CFG['chunk_size']):
   chunk=rows[i:i+CFG['chunk_size']];identity=dict(code=fingerprint,source=sources[v]['source_sha256'],uids=hashlib.sha256(''.join(r['mask_uid'] for r in chunk).encode()).hexdigest());jobs.append((v,i//CFG['chunk_size'],chunk,identity))
 results=pool(worker,jobs,monitor);out=folder('spatial');summary=[]
 for v in VIDEOS:
  rows=[]
  for m in sorted([r for r in results if r['video_id']==v],key=lambda r:r['chunk']):rows+=pq.read_table(WORK/f'spatial_chunks/{v}.{m["chunk"]:05d}.parquet').to_pylist()
  rows=align(rows)
  for name in ['h16','variance16','controls16','h32','variance32','controls32']:array(out/f'{v}.{name}.npy',[r.pop(name) for r in rows])
  for i,r in enumerate(rows):r['row_index']=i
  save(out/f'{v}.parquet',rows);summary.append(dict(video_id=v,observations=len(rows),fit16_valid=sum(r['fit16_valid'] for r in rows),fit32_valid=sum(r['fit32_valid'] for r in rows),source_complete=sum(r['source_complete'] for r in rows),hole_masks=sum(r['hole_count']>0 for r in rows)));monitor.check()
 atomic_json(REPORT/'spatial.json',dict(videos=summary,sources=sources,models=[16,32],source_mode='retrospective',code=spatial_identity(),identity_verified=False))
