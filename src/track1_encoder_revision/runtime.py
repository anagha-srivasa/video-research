"""Certificate-bound precompiled geometry; fitting and proof stay offline."""
import collections,json
import numpy as np
import pyarrow.parquet as pq
from track1_lab.topology_certificate import canonical_nets,input_hashes
from .experiments import D

class CompiledGeometry:
 def __init__(self,coefficients,metadata,samples=512):
  c=np.array(coefficients,dtype=np.float32,copy=True);meta=dict(metadata);cert=json.loads(meta['certificate_json']);expected=input_hashes(c,meta['allocation'],meta['roles'],meta['parents'],meta['window_seconds'])
  if not cert.get('certified') or any(cert.get(k)!=v for k,v in expected.items()):raise ValueError('Uncertified or changed geometry/decoder')
  if not isinstance(samples,int) or not 8<=samples<=4096:raise ValueError('Bounded sample count required')
  self.start=meta['window_start_us'];self.duration=meta['window_seconds']*1e6
  point,ids=canonical_nets(c,meta['allocation']);self.rings=[]
  for ring,(m,role) in enumerate(zip(meta['allocation'],meta['roles'])):
   u=np.arange(samples)*m/samples;piece=np.floor(u).astype(int);z=u-piece;bs=np.c_[(1-z)**3,3*z*(1-z)**2,3*z*z*(1-z),z**3];net=np.einsum('ns,ntsc->ntc',bs,point[ids==ring][piece]);net.setflags(write=False);self.rings.append((net,role))
 def evaluate(self,timestamp_us):
  t=(float(timestamp_us)-self.start)/self.duration
  if not np.isfinite(t) or not 0<=t<=1:raise ValueError('No geometry extrapolation outside certified history')
  bt=np.array([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3]);return [np.einsum('t,ntc->nc',bt,net)[::-1] if role=='hole' else np.einsum('t,ntc->nc',bt,net) for net,role in self.rings]

class RevisionDataset:
 """Pilot optional geometry only. The finetuned M16 carrier stays a separate reader."""
 def __init__(self):self.cache=collections.OrderedDict();self.context_cache=collections.OrderedDict()
 def close(self):
  for rows,a in self.cache.values():a._mmap.close()
  self.cache.clear()
  for rows,arrays in self.context_cache.values():
   for a in arrays.values():a._mmap.close()
  self.context_cache.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def geometry(self,video_id,source,entity_track_id,window_end_us,protocol,variant,samples=512):
  from track1_finetune.data import split
  split(video_id)
  if video_id not in self.cache:
   if len(self.cache)>=2:self.cache.popitem(last=False)[1][1]._mmap.close()
   rows=pq.read_table(D/f'{video_id}.parquet').to_pylist();a=np.load(D/f'{video_id}.geometry.npy',mmap_mode='r');self.cache[video_id]=(rows,a)
  self.cache.move_to_end(video_id);rows,a=self.cache[video_id];matching=[r for r in rows if (r['source'],r['entity_track_id'],r['window_end_us'],r['protocol'],r['variant'])==(source,entity_track_id,window_end_us,protocol,variant)]
  if len(matching)!=1:raise KeyError('No unique pilot state for the requested withheld-target protocol')
  r=matching[0]
  if not r['certificate_valid'] or not r.get('source_complete',False):raise ValueError('Complete certificate-bound source geometry unavailable')
  return CompiledGeometry(a[r['row_index']],r,samples)

 def get_context_sequence(self,index):
  from track1_finetune.data import split
  from track1_lab.topology_certificate import evaluate_curves
  v=index['video_id']
  if split(v)!=index['split'] or index['model']!=16:raise ValueError('M16 carrier with participant split required')
  if v not in self.context_cache:
   if len(self.context_cache)>=2:
    for a in self.context_cache.popitem(last=False)[1][1].values():a._mmap.close()
   rows=pq.read_table(D/f'{v}.context_states.parquet').to_pylist();arrays={name:np.load(D/f'{v}.context_{name}.npy',mmap_mode='r') for name in ['coefficients','geometry','appearance','xi','kinematic_position','kinematic_velocity','kinematic_cov']};self.context_cache[v]=(rows,arrays)
  self.context_cache.move_to_end(v);rows,a=self.context_cache[v];selected=[rows[i] for i in index['row_indices']]
  if len(selected)!=index['sequence_length'] or index.get('source')!='stable_context_extension' or any(r['entity_track_id']!=index['entity_track_id'] or r['episode_id']!=index['episode_id'] for r in selected):raise ValueError('Context sequence identity mismatch')
  if not selected or any(not r['valid'] or not r['appearance_valid'] or r['model']!=16 for r in selected):raise ValueError('Unsupported context carrier')
  if [r['window_end_us'] for r in selected]!=index['end_times_us'] or len({r['epoch'] for r in selected})!=1 or len({r['episode_id'] for r in selected})!=1 or any(b['window_end_us']-c['window_end_us']!=250000 for c,b in zip(selected,selected[1:])):raise ValueError('Context gap or epoch boundary')
  ids=index['row_indices'];lookup={(r['model'],r['window_end_us']):r for r in rows};geometry=[];kinematics=[]
  for r in selected:
   other=lookup[(32,r['window_end_us'])];geometry.append(CompiledGeometry(a['geometry'][other['row_index']],other) if other['geometry_training_eligible'] else None);i=r['row_index'];valid=r['kinematics_valid'];kinematics.append(dict(position=np.array(a['kinematic_position'][i]) if valid else None,velocity=np.array(a['kinematic_velocity'][i]) if valid else None,covariance=np.array(a['kinematic_cov'][i]) if valid else None,support_valid=valid,measurement_availability='retrospective',covariance_calibrated=False))
  return dict(coefficients=np.array(a['coefficients'][ids,:,:35]),xi=np.array(a['xi'][ids,:,:68]),appearance=np.array(a['appearance'][ids]),appearance_mask=np.array([r['appearance_valid'] for r in selected]),kinematics=kinematics,geometry32=geometry,metadata=selected,semantic_ground_truth=None)
