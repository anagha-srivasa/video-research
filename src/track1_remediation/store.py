"""M16 continuity carrier with optional, explicitly scoped M32 geometry."""
import collections,json
import numpy as np
import pyarrow as pa,pyarrow.parquet as pq
from track1_lab.topology_certificate import input_hashes,evaluate_curves
from track1_retrain.store import Dataset,sequences
from . import *

class TieredDataset:
 def __init__(self,source='context'):
  if source not in SOURCES:raise ValueError('Unknown source')
  self.source=source;self.base=Dataset(root=SOURCES[source]);self.cache=collections.OrderedDict()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):
  self.base.close()
  for _,a in self.cache.values():a._mmap.close()
  self.cache.clear()
 def get_state(self,video_id,entity_track_id,window_end_us):
  core=self.base.get_state(video_id,entity_track_id,window_end_us,model=16);baseline=self.base.get_state(video_id,entity_track_id,window_end_us,model=32,require_functional=False);geometry=None
  if baseline['metadata']['geometry_training_eligible']:
   meta=baseline['metadata'];geometry=dict(coefficients=baseline['geometry_coefficients'],metadata=meta,kind='full_one_second_history')
  else:
   if video_id not in self.cache:
    if len(self.cache)>=2:self.cache.popitem(last=False)[1][1]._mmap.close()
    rows=pq.read_table(D/self.source/f'{video_id}.geometry.parquet').to_pylist();lookup={(r['entity_track_id'],r['window_end_us']):r for r in rows};a=np.load(D/self.source/f'{video_id}.coefficients.npy',mmap_mode='r');self.cache[video_id]=(lookup,a)
   self.cache.move_to_end(video_id);lookup,a=self.cache[video_id];r=lookup[(entity_track_id,window_end_us)]
   if r['valid']:geometry=dict(coefficients=np.array(a[r['row_index']]),metadata=dict(r),kind='latest_stable_topology_segment')
  if geometry is not None:self._verify_geometry(geometry)
  return dict(core16=core,geometry32=geometry,m32_available=geometry is not None,quality=dict(core_valid=True,core16_geometry_certified=core['metadata'].get('certified_geometry_valid',False),core16_geometry_complete_certified=core['metadata'].get('geometry_training_eligible',False),geometry32_source_complete=geometry is not None,geometry32_certificate_bound=geometry is not None,geometry32_history_seconds=geometry['metadata']['window_seconds'] if geometry else None,geometry32_epoch=geometry['metadata']['epoch'] if geometry else None,identity_verified=False,physical_motion_verified=False,geometry_covariance_calibrated=False,geometry_covariance=None,discrete_identity_probabilities=None,discrete_topology_probabilities=None),kinematics=core['kinematics'])
 @staticmethod
 def _verify_geometry(geometry):
  r=geometry['metadata'];certificate=json.loads(r['certificate_json']);expected=input_hashes(geometry['coefficients'],r['allocation'],r['roles'],r['parents'],r['window_seconds'])
  if not certificate or not certificate['certified'] or any(certificate.get(k)!=v for k,v in expected.items()):raise ValueError('Geometry certificate binding mismatch')
 @classmethod
 def geometry(cls,state,timestamp_us,samples=512):
  g=state['geometry32']
  if g is None:raise ValueError('No optional M32 geometry')
  cls._verify_geometry(g);r=g['metadata']
  if timestamp_us<r['window_start_us'] or timestamp_us>r['window_end_us']:raise ValueError('Outside geometry segment')
  phase=(timestamp_us-r['window_start_us'])/(r['window_seconds']*1e6)
  return evaluate_curves(g['coefficients'],r['allocation'],r['roles'],phase,samples)

def build(monitor):
 summary=[]
 schema=pa.schema([('video_id',pa.string()),('split',pa.string()),('entity_track_id',pa.string()),('model',pa.int64()),('sequence_length',pa.int64()),('row_indices',pa.list_(pa.int64())),('end_times_us',pa.list_(pa.int64())),('all_core16_geometry_complete_certified',pa.bool_()),('identity_verified',pa.bool_()),('geometry32_mask',pa.list_(pa.bool_())),('geometry32_epochs',pa.list_(pa.string())),('geometry32_history_seconds',pa.list_(pa.float64())),('geometry32_same_epoch_pairs',pa.list_(pa.bool_())),('all_compound_geometry_available',pa.bool_()),('single_compound_epoch',pa.bool_()),('geometry_epochs_may_change_explicitly',pa.bool_())])
 for source,base in SOURCES.items():
  states=[]
  for v in VIDEOS:
   core=pq.read_table(base/'functional'/f'{v}.m16.parquet').to_pylist();full=pq.read_table(base/'functional'/f'{v}.m32.parquet').to_pylist();segments=pq.read_table(D/source/f'{v}.geometry.parquet').to_pylist();lookup={(r['entity_track_id'],r['window_end_us']):r for r in full};piece={(r['entity_track_id'],r['window_end_us']):r for r in segments}
   for r in core:
    key=(r['entity_track_id'],r['window_end_us']);f=lookup[key];g=piece[key];usefull=f['geometry_training_eligible'];available=usefull or g['valid'];chosen=f if usefull else g;states.append(dict(r,m32_available=available,m32_kind='full_one_second_history' if usefull else 'latest_stable_topology_segment' if g['valid'] else None,m32_epoch=chosen['epoch'] if available else None,m32_history_seconds=chosen['window_seconds'] if available else None))
  save(D/source/'tiered_states.parquet',states);index_dir=D/source/'indices';index_dir.mkdir(exist_ok=True)
  bykey={(r['video_id'],r['entity_track_id'],r['window_end_us']):r for r in states}
  for length in [4,8,16]:
   seq=sequences(states,length)
   for row in seq:
    row['all_core16_geometry_complete_certified']=row.pop('all_geometry_valid');local=[bykey[(row['video_id'],row['entity_track_id'],t)] for t in row['end_times_us']];row['geometry32_mask']=[r['m32_available'] for r in local];row['geometry32_epochs']=[r['m32_epoch'] for r in local];row['geometry32_history_seconds']=[r['m32_history_seconds'] for r in local];row['geometry32_same_epoch_pairs']=[local[i]['m32_available'] and local[i-1]['m32_available'] and local[i]['m32_epoch']==local[i-1]['m32_epoch'] for i in range(1,length)];row['all_compound_geometry_available']=all(row['geometry32_mask']);row['single_compound_epoch']=row['all_compound_geometry_available'] and all(row['geometry32_same_epoch_pairs']);row['geometry_epochs_may_change_explicitly']=True
   for part in ['train','validation']:
    local=[r for r in seq if r['split']==part];path=index_dir/f'{part}.length{length}.parquet';temp=path.with_suffix('.tmp');pq.write_table(pa.Table.from_pylist(local,schema=schema),temp,compression=None);temp.replace(path)
    for v in [v for v in VIDEOS if split(v)==part]:
     rs=[r for r in local if r['video_id']==v];summary.append(dict(source=source,video_id=v,split=part,length=length,core_sequences=len(rs),all_compound_geometry_available=sum(r['all_compound_geometry_available'] for r in rs),single_compound_epoch_sequences=sum(r['single_compound_epoch'] for r in rs),source_contexts=len({bykey[(r['video_id'],r['entity_track_id'],r['end_times_us'][0])]['episode_id'] for r in rs}),source_tracks=len({r['entity_track_id'] for r in rs}),independence_verified=False,overlapping_windows=True))
  monitor.check()
  with TieredDataset(source) as ds:
   for r in states:
    if r['valid'] and r['m32_available']:
     state=ds.get_state(r['video_id'],r['entity_track_id'],r['window_end_us']);ds.geometry(state,r['window_end_us']);break
 save(R/'tiered_continuity.parquet',summary);atomic_json(R/'tiered.json',dict(continuity=summary,core='Existing M16 coefficients and separate kinematics unchanged',geometry='Full certified M32 where available, otherwise latest stable source-topology segment; never bridges a discrete event',geometry_missing_is_zero=False,identity_verified=False,scientific_acceptance=False,track2_started=False,finalized=False))
