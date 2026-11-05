"""Explicit dataset-v2 training reader, numerical checks and gap-safe sequences."""
import collections,json,hashlib
import numpy as np
import pyarrow as pa,pyarrow.parquet as pq
from track1.numerics import basis
from track1_lab.variants import collocation
from track1_lab.topology_certificate import input_hashes,evaluate_curves
from track1_lab.history_detail import detail_basis
from .common import *

class Dataset:
 def __init__(self,root=None):self.cache=collections.OrderedDict();self.root=OUT if root is None else Path(root)
 def close(self):
  for _,arrays in self.cache.values():
   for a in arrays.values():a._mmap.close()
  self.cache.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def get_state(self,video_id,entity_track_id,window_end_us,model=16,require_geometry=False,require_functional=True):
  split(video_id)
  if model not in [16,32]:raise ValueError('Model must be 16 or 32')
  key=(video_id,model)
  if key not in self.cache:
   if len(self.cache)>=2:
    _,arrays=self.cache.popitem(last=False)[1]
    for a in arrays.values():a._mmap.close()
   rows=pq.read_table(self.root/'functional'/f'{video_id}.m{model}.parquet').to_pylist();lookup={(r['entity_track_id'],r['window_end_us']):r for r in rows};arrays={name:np.load(self.root/'functional'/f'{video_id}.m{model}.{name}.npy',mmap_mode='r') for name in ['coefficients','coefficient_cov','geometry_coefficients','xi','xi_dot','detail','detail_transform','appearance_coefficients','kinematic_position','kinematic_velocity','kinematic_cov']};self.cache[key]=(lookup,arrays)
  self.cache.move_to_end(key);lookup,arrays=self.cache[key];r=lookup[(entity_track_id,int(window_end_us))]
  if require_functional and not r['valid']:raise ValueError('Unsupported state: '+str(r['failure_reason']))
  if require_geometry and not r['geometry_training_eligible']:raise ValueError('Complete certified geometry unavailable')
  i=r['row_index'];state={name:np.array(a[i]) for name,a in arrays.items()};state['metadata']=dict(r);state['functional_endpoint_state']=basis(1.)@state['coefficients'];state['functional_endpoint_velocity']=2*basis(1.,1)@state['coefficients'];state['kinematics']=dict(position=state.pop('kinematic_position') if r['kinematics_valid'] else None,velocity=state.pop('kinematic_velocity') if r['kinematics_valid'] else None,covariance=state.pop('kinematic_cov') if r['kinematics_valid'] else None,support_valid=r['kinematics_valid'],covariance_calibrated=False,measurement_availability='retrospective')
  for key_ in ['kinematic_position','kinematic_velocity','kinematic_cov']:state.pop(key_,None)
  state['kinematics'].update(method='local_constant_velocity',failure_reason=r['kinematics_failure_reason'],observation_count=r['kinematics_observation_count'],latest_input_us=r['kinematics_latest_input_us'],acceleration=None,derivative_reliable=None)
  state['appearance_embedding']=basis(1.)@state['appearance_coefficients'][:,0];state['context_embedding']=basis(1.)@state['appearance_coefficients'][:,1]
  if require_geometry:
   cert=json.loads(r['certificate_json']);expected=input_hashes(state['geometry_coefficients'],r['allocation'],r['roles'],r['parents'],1.)
   if not cert or any(cert.get(k)!=v for k,v in expected.items()):raise ValueError('Certificate/decoder binding mismatch')
  return state
 def get_kinematics(self,video_id,entity_track_id,window_end_us,model=16):
  return self.get_state(video_id,entity_track_id,window_end_us,model=model,require_functional=False)['kinematics']
 def geometry(self,state,timestamp_us,samples=512):
  r=state['metadata'];cert=json.loads(r['certificate_json'])
  if not r['geometry_training_eligible']:raise ValueError('Complete certified geometry unavailable')
  expected=input_hashes(state['geometry_coefficients'],r['allocation'],r['roles'],r['parents'],1.)
  if not cert or not cert['certified'] or any(cert.get(k)!=v for k,v in expected.items()):raise ValueError('Certificate/decoder binding mismatch')
  phase=1+(timestamp_us-r['window_end_us'])/1e6
  return evaluate_curves(state['geometry_coefficients'],r['allocation'],r['roles'],phase,samples)

def sequences(rows,length):
 groups=collections.defaultdict(list);result=[]
 for r in rows:
  if r['valid']:groups[(r['video_id'],r['entity_track_id'],r['model'])].append(r)
 for (v,track,n),rs in groups.items():
  rs.sort(key=lambda r:r['window_end_us']);run=[]
  for r in rs:
   if run and (r['window_end_us']-run[-1]['window_end_us']!=250000 or r['epoch']!=run[-1]['epoch']):run=[]
   run.append(r)
   if len(run)>=length:
    chosen=run[-length:];result.append(dict(video_id=v,split=split(v),entity_track_id=track,model=n,sequence_length=length,row_indices=[x['row_index'] for x in chosen],end_times_us=[x['window_end_us'] for x in chosen],all_geometry_valid=all(x['geometry_training_eligible'] for x in chosen),identity_verified=False))
 return result

def selection_ledger(v):
 import duckdb
 with duckdb.connect() as db:
  db.execute('SET threads=1');db.execute("SET memory_limit='1GB'")
  table=db.execute('''SELECT m.video_id,m.mask_uid,m.source_kind,m.geometry_complete,m.partial_geometry_valid,
      s.row_index IS NOT NULL AS used_in_refit,
      CASE WHEN s.row_index IS NOT NULL THEN 'selected_provisional_source_association'
      WHEN NOT m.geometry_complete THEN 'incomplete_or_quarantined_source_geometry'
      ELSE 'not_in_inherited_source_selection_or_association' END AS selection_reason
      FROM read_parquet(?) m JOIN read_parquet(?) f USING(video_id,timestamp_us)
      LEFT JOIN read_parquet(?) s USING(mask_uid)''',[str(ROOT/f'canonical/v2/masks.{v}.parquet'),str(ROOT/'dataset/partition_v2/frames.parquet'),str(folder('spatial')/f'{v}.parquet')]).fetch_arrow_table()
 pq.write_table(table,REPORT/f'source_selection.{v}.parquet',compression=None)
 return dict(video_id=v,source_observations=len(table),used_observations=int(table.column('used_in_refit').to_numpy().sum()))

def verify(monitor):
 allrows=[];checks=[];sources=[]
 with Dataset() as ds:
  for v in VIDEOS:
   sources.append(selection_ledger(v))
   for n in [16,32]:
    meta=pq.read_table(folder('functional')/f'{v}.m{n}.parquet').to_pylist();allrows+=meta;arrays={name:np.load(folder('functional')/f'{v}.m{n}.{name}.npy',mmap_mode='r') for name in ['coefficients','coefficient_cov','geometry_coefficients','xi','xi_dot','detail','detail_transform','appearance_coefficients','kinematic_cov']};psd_bad=0;reconstruction_error=0.;certificate_checks=0
    for lo in range(0,len(meta),128):
     for a in arrays.values():assert np.isfinite(a[lo:lo+128]).all()
     eig=np.linalg.eigvalsh(arrays['coefficient_cov'][lo:lo+128].astype(float));scale=np.maximum(np.max(abs(eig),axis=-1),1e-20);psd_bad+=int(np.sum(eig[...,0]<-1e-5*scale));monitor.check()
    assert psd_bad==0,(v,n,psd_bad)
    for r in meta:
     i=r['row_index']
     if not r['valid']:continue
     assert r['latest_input_us']<=r['window_end_us'];assert r['online_observation_allowed'] is False
     c=arrays['coefficients'][i].astype(float);decoded=collocation(c,1.);difference=max(float(np.max(abs(decoded['xi']-arrays['xi'][i]))),float(np.max(abs(decoded['xi_dot']-arrays['xi_dot'][i]))));reconstruction_error=max(reconstruction_error,difference);np.testing.assert_allclose(decoded['xi'],arrays['xi'][i],rtol=1e-4,atol=2e-5);np.testing.assert_allclose(decoded['xi_dot'],arrays['xi_dot'][i],rtol=1e-4,atol=2e-4)
     assert np.max(abs(detail_basis(np.array([1.]))@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     assert np.max(abs(detail_basis(np.array([1.]),1)@arrays['detail_transform'][i]@arrays['detail'][i]))==0.
     if r['certified_geometry_valid']:
      cert=json.loads(r['certificate_json']);expected=input_hashes(arrays['geometry_coefficients'][i],r['allocation'],r['roles'],r['parents'],1.);assert cert['certified'] and all(cert[k]==value for k,value in expected.items());certificate_checks+=1
     if r['kinematics_valid']:
      assert r['kinematics_latest_input_us']<=r['window_end_us']
      eig=np.linalg.eigvalsh(arrays['kinematic_cov'][i].astype(float));assert eig[0]>=-1e-6*max(abs(eig).max(),1e-20)
    checks.append(dict(video_id=v,model=n,attempted=len(meta),valid=sum(r['valid'] for r in meta),certificate_checks=certificate_checks,max_collocation_absolute_error=reconstruction_error,psd_violations=psd_bad));monitor.check()
    for a in arrays.values():a._mmap.close()
 # The separate estimator must not inherit geometry-specific failure flags.
 for v in VIDEOS:
  a=pq.read_table(folder('functional')/f'{v}.m16.parquet').to_pylist();b=pq.read_table(folder('functional')/f'{v}.m32.parquet').to_pylist();assert [(r['entity_track_id'],r['window_end_us']) for r in a]==[(r['entity_track_id'],r['window_end_us']) for r in b]
  assert [r['kinematics_valid'] for r in a]==[r['kinematics_valid'] for r in b]
  for name in ['kinematic_position','kinematic_velocity','kinematic_cov']:
   x=np.load(folder('functional')/f'{v}.m16.{name}.npy',mmap_mode='r');y=np.load(folder('functional')/f'{v}.m32.{name}.npy',mmap_mode='r');assert np.array_equal(x,y);x._mmap.close();y._mmap.close()
 indices=folder('indices');counts={};schema=pa.schema([('video_id',pa.string()),('split',pa.string()),('entity_track_id',pa.string()),('model',pa.int64()),('sequence_length',pa.int64()),('row_indices',pa.list_(pa.int64())),('end_times_us',pa.list_(pa.int64())),('all_geometry_valid',pa.bool_()),('identity_verified',pa.bool_())])
 for length in [4,16]:
  seq=sequences(allrows,length)
  for part in ['train','validation']:
   values=[r for r in seq if r['split']==part];pq.write_table(pa.Table.from_pylist(values,schema=schema),indices/f'{part}.length{length}.parquet',compression=None);counts[f'{part}/length{length}']=len(values)
 atomic_json(REPORT/'verification.json',dict(checks=checks,source_selection=sources,sequences=counts,numerical_checks_passed=True,independent_scientific_acceptance=False,clean_full_rebuild_performed=False,code=code()))
 atomic_json(OUT/'status.json',dict(ready=True,ready_scope='numerical functional-state ingestion on available partition timelines; inspect model/geometry/kinematic flags',sequence16_available=counts['train/length16']>0,finalized=False,scientific_acceptance=False,online_observation_pipeline_ready=False,track2_started=False,models=[16,32],dataset_manifest_sha256=digest(ROOT/'reports/dataset_v2/manifest.json')))
 from .report import write
 write()
