"""Immutable M16 adapter and explicit future-target denominators."""
import collections,json,hashlib,random,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json,digest
from track1_retrain.common import CFG as T1_CFG,split
from track1_finetune.data import save

CFG=json.loads((ROOT/'configs/track2_feasibility.json').read_text())
DATA=ROOT/'derived/track2_feasibility';REPORT=ROOT/'reports/track2_feasibility';WORK=ROOT/'work/track2_feasibility'
VIDEOS=T1_CFG['train']+T1_CFG['validation'];BASE=ROOT/'derived/track1_context_v3'
INPUT_DIM=157;OUTPUT_DIM=144

def stable_id(sequence):return hashlib.sha256(json.dumps([sequence['video_id'],sequence['entity_track_id'],sequence['end_times_us']],sort_keys=True).encode()).hexdigest()[:24]

def safe_covariance_code(cov):
 cov=np.asarray(cov,float)
 if cov.shape!=(4,4) or not np.isfinite(cov).all():raise ValueError('Invalid kinematic covariance')
 scale=max(float(abs(cov).max()),1e-12)
 if np.max(abs(cov-cov.T))>1e-5*scale:raise ValueError('Asymmetric covariance')
 values,vectors=np.linalg.eigh((cov+cov.T)/2)
 if values[0]<-1e-5*scale:raise ValueError('Non-PSD covariance')
 floor=max(1e-12,scale*1e-7);fixed=(vectors*np.maximum(values,floor))@vectors.T;chol=np.linalg.cholesky((fixed+fixed.T)/2);indices=np.tril_indices(4);code=chol[indices];diag=indices[0]==indices[1];code[diag]=np.log(code[diag]);return code,dict(eigenvalue_floor=floor,repaired=bool(np.any(values<floor)))

def target_reason(lookup,video,track,origin,horizon):
 for delta in range(250000,horizon+1,250000):
  row=lookup.get((video,track,origin['window_end_us']+delta))
  if row is None:return 'missing_future_state'
  if row['episode_id']!=origin['episode_id']:return 'episode_boundary'
  if row['epoch']!=origin['epoch']:return 'source_epoch_boundary'
  if not row['valid']:return 'invalid_future_functional_state'
  if not row['kinematics_valid']:return 'invalid_future_kinematics'
  if row['latest_input_us']>row['window_end_us'] or row['kinematics_latest_input_us']>row['window_end_us']:return 'future_input_in_target_state'
  if row['kinematics_latest_input_us']<=origin['window_end_us']:return 'target_kinematics_has_no_future_observation'
 return None

def inventory(base):
 rows=[];indices=[]
 for v in VIDEOS:rows+=pq.read_table(base/'functional'/f'{v}.m16.parquet').to_pylist()
 for part in ['train','validation']:indices += [r for r in pq.read_table(base/'indices'/f'{part}.length4.parquet').to_pylist() if r['model']==16]
 return rows,indices

def prepare():
 DATA.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
 source={}
 for base in [ROOT/'derived/track1_dataset_v2',BASE]:
  source[str(base.relative_to(ROOT))+'/status.json']=digest(base/'status.json')
  for p in (base/'indices').glob('*.parquet'):source[str(p.relative_to(ROOT))]=digest(p)
  for p in (base/'functional').glob('*.m16.*'):source[str(p.relative_to(ROOT))]=digest(p)
 # Preserve prior frozen-v2 manifests and all their declared files through the experiment.
 for p in (ROOT/'reports/track1_refinement').glob('*manifest*.json'):source[str(p.relative_to(ROOT))]=digest(p)
 locked=dict(source_hashes=source,config_sha256=digest(ROOT/'configs/track2_feasibility.json'),report_sha256=digest(Path('/Users/anagha/Downloads/track 2 decision.md')),reported_public_commit='13e02ff092f610371c1f8d220c599bec7d54253b',local_commit_verified=False,full_track2_authorized=False)
 atomic_json(REPORT/'frozen_inputs.json',locked)
 audit=[]
 for name,base in [('base_v2',ROOT/'derived/track1_dataset_v2'),('amended_context',BASE)]:
  rows,indices=inventory(base);lookup={(r['video_id'],r['entity_track_id'],r['window_end_us']):r for r in rows}
  for s in indices:
   origin=lookup[(s['video_id'],s['entity_track_id'],s['end_times_us'][-1])];past=[lookup[(s['video_id'],s['entity_track_id'],t)] for t in s['end_times_us']];input_ok=all(r['valid'] and r['kinematics_valid'] and r['epoch']==origin['epoch'] and r['episode_id']==origin['episode_id'] for r in past)
   for horizon in CFG['horizons_us']:audit.append(dict(cohort=name,sequence_id=stable_id(s),video_id=s['video_id'],participant=s['video_id'].split('_')[0],split=s['split'],horizon_us=horizon,input_supported=input_ok,target_supported=input_ok and target_reason(lookup,s['video_id'],s['entity_track_id'],origin,horizon) is None,reason='input_support_or_boundary' if not input_ok else target_reason(lookup,s['video_id'],s['entity_track_id'],origin,horizon)))
 save(REPORT/'availability_audit.parquet',audit)
 rows,indices=inventory(BASE);lookup={(r['video_id'],r['entity_track_id'],r['window_end_us']):r for r in rows};byvideo={v:[r for r in rows if r['video_id']==v] for v in VIDEOS};samples=[];xs=[];ys=[];masks=[];ledger=[];jitter=[]
 for v in VIDEOS:
  fields=['coefficients','kinematic_position','kinematic_velocity','kinematic_cov'];arrays={key:np.load(BASE/'functional'/f'{v}.m16.{key}.npy',mmap_mode='r') for key in fields}
  def state(row):
   i=row['row_index'];c=np.array(arrays['coefficients'][i],np.float32).ravel();p=np.array(arrays['kinematic_position'][i],np.float32);vel=np.array(arrays['kinematic_velocity'][i],np.float32);cov,info=safe_covariance_code(arrays['kinematic_cov'][i]);return np.r_[c,p,vel,cov,np.ones(3)].astype(np.float32),np.r_[p,vel,c].astype(np.float32),info
  for seq in [s for s in indices if s['video_id']==v]:
   sid=stable_id(seq);past=[lookup[(v,seq['entity_track_id'],t)] for t in seq['end_times_us']];last=past[-1];reason=None;input_values=[];input_infos=[]
   if any(not r['valid'] or not r['kinematics_valid'] or r['epoch']!=last['epoch'] or r['episode_id']!=last['episode_id'] for r in past):reason='input_support_or_boundary'
   if reason is None and any(r['latest_input_us']>r['window_end_us'] or r['kinematics_latest_input_us']>r['window_end_us'] for r in past):reason='future_input_leakage'
   if any(b-a!=250000 for a,b in zip(seq['end_times_us'],seq['end_times_us'][1:])):reason='missing_input_hop'
   if reason is None:
    try:
     for r in past:
      x,y,info=state(r);input_values.append(x);input_infos.append(info)
    except ValueError as e:reason=str(e)
   target_values=np.zeros((3,OUTPUT_DIM),np.float32);mask=np.zeros(3,bool);target_ids=[]
   for hi,h in enumerate(CFG['horizons_us']):
    why=reason or target_reason(lookup,v,seq['entity_track_id'],last,h);target=lookup.get((v,seq['entity_track_id'],last['window_end_us']+h))
    if why is None:
     try:_,target_values[hi],_=state(target);mask[hi]=True
     except ValueError as e:why=str(e)
    target_ids.append(f"{v}:{seq['entity_track_id']}:{last['window_end_us']+h}")
    ledger.append(dict(sequence_id=sid,video_id=v,participant=v.split('_')[0],split=seq['split'],input_cutoff_us=last['window_end_us'],horizon_us=h,input_supported=reason is None,target_supported=bool(mask[hi]),reason=why,episode_id=last['episode_id'],epoch=last['epoch'],latest_input_us=max(r['latest_input_us'] or 0 for r in past),input_uses_appearance=False,input_uses_detail=False,input_uses_m32=False,source_labels_retrospective=True,online_rgb_forecast=False))
   if reason is not None:continue
   sample=dict(sequence_id=sid,video_id=v,participant=v.split('_')[0],split=seq['split'],entity_track_id=seq['entity_track_id'],episode_id=last['episode_id'],epoch=last['epoch'],input_times_us=seq['end_times_us'],target_state_ids=target_ids,input_state_ids=[f"{v}:{seq['entity_track_id']}:{t}" for t in seq['end_times_us']],row_index=len(samples));samples.append(sample);xs.append(input_values);ys.append(target_values);masks.append(mask);jitter.append(input_infos)
  for a in arrays.values():a._mmap.close()
 for name,values in [('x',xs),('y',ys),('mask',masks)]:np.save(DATA/f'{name}.npy',np.asarray(values,np.bool_ if name=='mask' else np.float32),allow_pickle=False)
 save(DATA/'samples.parquet',samples);save(REPORT/'target_ledger.parquet',ledger);atomic_json(REPORT/'covariance_repairs.json',jitter)
 summary=[]
 for cohort in ['base_v2','amended_context']:
  for part in ['train','validation']:
   local=[r for r in audit if r['cohort']==cohort and r['split']==part];summary.append(dict(cohort=cohort,split=part,m16_histories=len(local)//3,targets={str(h):sum(r['target_supported'] for r in local if r['horizon_us']==h) for h in CFG['horizons_us']}))
 atomic_json(REPORT/'adapter.json',dict(summary=summary,accepted_input_histories=len(samples),x_shape=list(np.array(xs).shape),y_shape=list(np.array(ys).shape),missing_target_values_are_masked_placeholders=True,required_covariance_input='conditional source-kinematic covariance, uncalibrated',scientific_acceptance=False,protocol_amendment=CFG['amendment']))
 # Every training participant and both exploratory validation participants must support every horizon.
 for v in VIDEOS:
  for h in CFG['horizons_us']:
   if not any(r['video_id']==v and r['horizon_us']==h and r['target_supported'] for r in ledger):raise ValueError('Insufficient future support for '+v+' / '+str(h))

class Data:
 def __init__(self):
  self.samples=pq.read_table(DATA/'samples.parquet').to_pylist();self.x=np.load(DATA/'x.npy');self.y=np.load(DATA/'y.npy');self.mask=np.load(DATA/'mask.npy');self.participants=np.array([r['participant'] for r in self.samples]);self.split=np.array([r['split'] for r in self.samples])
 def indices(self,part='train',exclude=None,only=None,usable=True):
  valid=self.split==part
  if exclude is not None:valid&=self.participants!=exclude
  if only is not None:valid&=self.participants==only
  if usable:valid&=self.mask.any(1)
  return np.flatnonzero(valid)

def normalization(data,indices):
 ids=set();x=[];target_ids=set();y=[]
 for i in indices:
  for j,key in enumerate(data.samples[i]['input_state_ids']):
   if key not in ids:ids.add(key);x.append(data.x[i,j,:154])
  for j,key in enumerate(data.samples[i]['target_state_ids']):
   if data.mask[i,j] and key not in target_ids:target_ids.add(key);y.append(data.y[i,j])
 x=np.array(x,float);y=np.array(y,float);xf=np.r_[np.full(140,.01),np.full(4,.001),np.full(10,.01)];yf=np.r_[np.full(4,.001),np.full(140,.01)]
 return dict(x_mean=x.mean(0).tolist(),x_scale=np.maximum(x.std(0),xf).tolist(),y_mean=y.mean(0).tolist(),y_scale=np.maximum(y.std(0),yf).tolist(),training_participants=[str(p) for p in sorted(set(data.participants[indices]))],unique_input_states=len(ids),unique_target_states=len(target_ids))

def check_frozen():
 for name,h in json.loads((REPORT/'frozen_inputs.json').read_text())['source_hashes'].items():
  if digest(ROOT/name)!=h:raise ValueError('Frozen Track 1 file changed: '+name)
