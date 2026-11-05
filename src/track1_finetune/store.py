"""Refitted M16 histories, separate kinematics and explicitly optional geometry."""
import collections
import json
import numpy as np
import pyarrow.parquet as pq
from track1_retrain.store import Dataset as FunctionalDataset,sequences
from track1_lab.topology_certificate import input_hashes
from track1_lab.variants import collocation
from track1_lab.history_detail import detail_basis
from track1_remediation.store import TieredDataset
from .data import *

class FinetunedDataset:
 def __init__(self,require_ready=True):
  if require_ready and not json.loads((D/'status.json').read_text())['numerical_ingestion_ready']:raise ValueError('Refit not ready')
  self.core={s:FunctionalDataset(root=D/s) for s in BASES};self.old={s:FunctionalDataset(root=b) for s,b in BASES.items()};self.latest=collections.OrderedDict()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):
  for reader in list(self.core.values())+list(self.old.values()):reader.close()
  for lookup,a in self.latest.values():a._mmap.close()
  self.latest.clear()
 def geometry(self,source,v,track,t):
  state=self.core[source].get_state(v,track,t,model=32,require_functional=False);r=state['metadata']
  if r['geometry_training_eligible']:geometry=dict(coefficients=state['geometry_coefficients'],metadata=r,kind='finetuned_full_one_second_history')
  else:
   old=self.old[source].get_state(v,track,t,model=32,require_functional=False);meta=old['metadata']
   if meta['geometry_training_eligible']:geometry=dict(coefficients=old['geometry_coefficients'],metadata=meta,kind='preserved_baseline_full_history')
   else:
    key=(source,v)
    if key not in self.latest:
     if len(self.latest)>=2:self.latest.popitem(last=False)[1][1]._mmap.close()
     root=ROOT/f'derived/track1_remediation/{source}';rs=pq.read_table(root/f'{v}.geometry.parquet').to_pylist();a=np.load(root/f'{v}.coefficients.npy',mmap_mode='r');self.latest[key]=({(r['entity_track_id'],r['window_end_us']):r for r in rs},a)
    self.latest.move_to_end(key);lookup,a=self.latest[key];meta=lookup[(track,t)]
    geometry=dict(coefficients=np.array(a[meta['row_index']]),metadata=meta,kind='preserved_latest_stable_epoch') if meta['valid'] else None
  if geometry is not None:TieredDataset._verify_geometry(geometry)
  return geometry
 def get_sequence(self,index):
  source=index['source'];v=index['video_id'];times=index['end_times_us']
  if split(v)!=index['split'] or source not in self.core or len(times)!=index['sequence_length']:raise ValueError('Invalid sequence identity')
  if any(b-a!=250000 for a,b in zip(times,times[1:])):raise ValueError('Missing hop')
  states=[self.core[source].get_state(v,index['entity_track_id'],t,model=16) for t in times];meta=[s['metadata'] for s in states]
  if len({r['epoch'] for r in meta})!=1 or {r['episode_id'] for r in meta}!={index['episode_id']}:raise ValueError('Epoch/episode changed')
  return dict(coefficients=np.stack([s['coefficients'] for s in states]),coefficient_cov=np.stack([s['coefficient_cov'] for s in states]),xi=np.stack([s['xi'] for s in states]),xi_dot=np.stack([s['xi_dot'] for s in states]),appearance=np.stack([s['appearance_coefficients'] for s in states]),appearance_mask=np.array([r['appearance_valid'] for r in meta]),kinematics=[s['kinematics'] for s in states],geometry32=[self.geometry(source,v,index['entity_track_id'],t) for t in times],metadata=meta,semantic_ground_truth=None)

def build_indices():
 keys=prepared_keys();states=[]
 for source in BASES:
  for v in VIDEOS:
   rows=pq.read_table(D/source/'functional'/f'{v}.m16.parquet').to_pylist()
   for r in rows:
    if (r['entity_track_id'],r['window_end_us']) in keys[(source,v)]:states.append(dict(r,source=source))
 save(D/'states.parquet',states);counts=[]
 for length in [1,4,8,16]:
  for part in ['train','validation']:
   output=[]
   groups=collections.defaultdict(list)
   for r in states:
    if r['split']==part:groups[(r['source'],r['episode_id'])].append(r)
   for (source,episode),rs in groups.items():
    for row in sequences(rs,length):output.append(dict(row,source=source,episode_id=episode,semantic_ground_truth=False))
   if not output:raise ValueError('No supported sequences')
   save(D/'indices'/f'{part}.length{length}.parquet',output);counts.append(dict(split=part,length=length,sequences=len(output)))
 return counts

def verify_tensors(monitor):
 checks=[]
 for source in BASES:
  for v in VIDEOS:
   for n in CFG['models']:
    folder=D/source/'functional';rows=pq.read_table(folder/f'{v}.m{n}.parquet').to_pylist();arrays={p.name[len(f'{v}.m{n}.'):-4]:np.load(p,mmap_mode='r') for p in folder.glob(f'{v}.m{n}.*.npy')}
    for a in arrays.values():
     assert a.shape[0]==len(rows)
     for lo in range(0,len(a),128):assert np.isfinite(a[lo:lo+128]).all()
    certificates=0
    for r in rows:
     i=r['row_index']
     if not r['valid']:continue
     assert arrays['coefficients'][i].shape==(4,2*n+3)
     assert r['latest_input_us']<=r['window_end_us'] and not r['online_observation_allowed']
     cov=arrays['coefficient_cov'][i].astype(float);e=np.linalg.eigvalsh(cov);assert (e[...,0]>=-1e-5*np.maximum(np.abs(e).max(-1),1e-20)).all()
     decoded=collocation(arrays['coefficients'][i].astype(float),1.)
     np.testing.assert_allclose(decoded['xi'],arrays['xi'][i],rtol=1e-4,atol=2e-5);np.testing.assert_allclose(decoded['xi_dot'],arrays['xi_dot'][i],rtol=1e-4,atol=2e-4)
     assert np.max(abs(detail_basis(np.array([1.]))@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     assert np.max(abs(detail_basis(np.array([1.]),1)@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     if r['certified_geometry_valid']:
      cert=json.loads(r['certificate_json']);expected=input_hashes(arrays['geometry_coefficients'][i],r['allocation'],r['roles'],r['parents'],1.);assert cert['certified'] and all(cert[k]==val for k,val in expected.items());certificates+=1
     if r['kinematics_valid']:
      assert r['kinematics_latest_input_us']<=r['window_end_us'];e=np.linalg.eigvalsh(arrays['kinematic_cov'][i].astype(float));assert e[0]>=-1e-6*max(np.abs(e).max(),1e-20)
    checks.append(dict(source=source,video_id=v,model=n,attempts=len(rows),valid=sum(r['valid'] for r in rows),certificates=certificates,source_complete_certified=sum(r['geometry_training_eligible'] for r in rows),appearance_valid=sum(r['appearance_valid'] for r in rows)))
    for a in arrays.values():a._mmap.close()
    monitor.check()
   # Kinematics must be identical to the original estimator, across both model representations.
   for name in ['kinematic_position','kinematic_velocity','kinematic_cov']:
    a=np.load(D/source/'functional'/f'{v}.m16.{name}.npy',mmap_mode='r');b=np.load(BASES[source]/'functional'/f'{v}.m16.{name}.npy',mmap_mode='r');np.testing.assert_array_equal(a,b);a._mmap.close();b._mmap.close()
 return checks
