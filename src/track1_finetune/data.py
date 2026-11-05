"""Prepared opportunities, immutable inputs and train-only sufficient statistics."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_retrain.common import CFG as OLD_CFG, split
from track1_retrain.temporal import load, proposals, timing_eligible

CFG=json.loads((ROOT/'configs/track1_finetune.json').read_text())
D=ROOT/'derived/track1_finetuned'
R=ROOT/'reports/track1_finetune'
W=ROOT/'work/track1_finetune'
P=ROOT/'derived/track1_training_ready'
BASES={'v2':ROOT/'derived/track1_dataset_v2','context':ROOT/'derived/track1_context_v3'}
EPISODES={'v2':ROOT/'dataset/partition_v2/episodes.parquet','context':ROOT/'dataset/context_v3/episodes.parquet'}
VIDEOS=OLD_CFG['train']+OLD_CFG['validation']

def rank(value):return hashlib.sha256((str(CFG['seed'])+str(value)).encode()).hexdigest()
def save(path,rows):
 import pyarrow as pa
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 pq.write_table(pa.Table.from_pylist(rows),temp,compression=None);temp.replace(path)
def array(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.save(f,np.asarray(value,np.float32),allow_pickle=False)
 temp.replace(path)
def prepared_keys():
 result=collections.defaultdict(set)
 for r in pq.read_table(P/'states.parquet').to_pylist():result[(r['source'],r['video_id'])].add((r['entity_track_id'],r['window_end_us']))
 return result

def cases_for(video,n,limit=None):
 """Fixed source selection with round-robin tracks; fitting failures stay visible."""
 keys=prepared_keys();groups=collections.defaultdict(list);inputs={}
 for source,base in BASES.items():
  rows,data=load(video,n,base/'spatial');inputs[source]=(rows,data)
  episodes=[e for e in pq.read_table(EPISODES[source]).to_pylist() if e['video_id']==video]
  for c in proposals(video,rows,episodes):
   if (c['entity_track_id'],c['window_end_us']) in keys[(source,video)] and timing_eligible(c,rows):
    groups[(source,c['entity_track_id'])].append(dict(c,source=source))
 for rs in groups.values():rs.sort(key=lambda c:rank((c['source'],c['entity_track_id'],c['window_end_us'])))
 ordered=sorted(groups,key=rank);selected=[];level=0
 while ordered and (limit is None or len(selected)<limit):
  remaining=[]
  for key in ordered:
   if level<len(groups[key]):selected.append(groups[key][level]);remaining.append(key)
   if limit is not None and len(selected)>=limit:break
  ordered=remaining;level+=1
 return selected,inputs

def baseline():return json.loads((BASES['v2']/'calibration/model.json').read_text())

def check_inputs():
 status=json.loads((P/'status.json').read_text());assert status['training_ingestion_ready']
 manifest=json.loads((P/'manifest.json').read_text())
 for p,h in manifest['artifacts'].items():assert digest(ROOT/p)==h,('Prepared artifact changed',p)
 dependencies=json.loads((ROOT/'reports/track1_training_ready/verification.json').read_text())['tensor_files_manifest']
 for entry in dependencies:assert digest(ROOT/entry['path'])==entry['sha256'],entry['path']
 hashes={e['path']:e['sha256'] for e in dependencies}
 for base in BASES.values():
  for folder in ['spatial','appearance']:
   for p in (base/folder).glob('*'):
    if p.is_file():hashes[str(p.relative_to(ROOT))]=digest(p)
 atomic_json(R/'input_manifest.json',dict(prepared_manifest_sha256=digest(P/'manifest.json'),dependencies=hashes,weak_annotations_used_for_fitting=False))
 for source,base in BASES.items():
  root=D/source;root.mkdir(parents=True,exist_ok=True)
  link=root/'spatial'
  if not link.exists():link.symlink_to(base/'spatial',target_is_directory=True)
  for name in ['appearance','calibration','functional']:(root/name).mkdir(exist_ok=True)
  for v in VIDEOS:
   link=root/'appearance'/f'{v}.parquet'
   if not link.exists():link.symlink_to(base/'appearance'/f'{v}.parquet')


def train_pca(monitor):
 """Track-balanced PCA on unique training masks from both prepared sources."""
 from scipy.linalg import eigh
 from threadpoolctl import threadpool_limits
 unique={};counts=collections.Counter();masks=collections.defaultdict(set)
 # Only observations retained in the fixed prepared timelines are included.
 for r in pq.read_table(P/'states.parquet').to_pylist():masks[(r['source'],r['video_id'])].add((r['window_start_us'],r['window_end_us']))
 for source,base in BASES.items():
  for v in OLD_CFG['train']:
   rows=pq.read_table(base/'spatial'/f'{v}.parquet',columns=['mask_uid','entity_track_id','timestamp_us','prior','quality']).to_pylist()
   meta=pq.read_table(base/'appearance'/f'{v}.parquet',columns=['mask_uid','row_index']).to_pylist();indices={r['mask_uid']:r['row_index'] for r in meta}
   intervals=sorted(masks[(source,v)])
   # These stores already cover selected timelines. Dedup avoids counting shared masks twice.
   for r in rows:
    key=(v,r['mask_uid'])
    if key in unique or not any(a<=r['timestamp_us']<=b for a,b in intervals):continue
    unique[key]=dict(r,source=source,video_id=v,row_index=indices[r['mask_uid']]);counts[(v,r['entity_track_id'])]+=1
 sums=np.zeros((2,576));cross=np.zeros((2,576,576));total=0.
 for source,base in BASES.items():
  for v in OLD_CFG['train']:
   a=np.load(base/'appearance'/f'{v}.features.npy',mmap_mode='r');rs=[r for r in unique.values() if r['source']==source and r['video_id']==v]
   for lo in range(0,len(rs),512):
    chunk=rs[lo:lo+512];x=np.asarray(a[[r['row_index'] for r in chunk]],float)
    w=np.array([r['prior']**.5*r['quality']/counts[(v,r['entity_track_id'])] for r in chunk]);total+=w.sum();sums+=np.einsum('n,njc->jc',w,x)
    with threadpool_limits(limits=4):cross+=np.einsum('n,njc,njd->jcd',w,x,x,optimize=True)
    monitor.check()
   a._mmap.close()
 mean=sums/total;components=[];explained=[]
 with threadpool_limits(limits=4):
  for j in range(2):
   cov=cross[j]/total-np.outer(mean[j],mean[j]);values,vectors=eigh((cov+cov.T)/2);order=np.argsort(values)[::-1];c=vectors[:,order[:128]]
   for k in range(128):
    if c[np.argmax(abs(c[:,k])),k]<0:c[:,k]*=-1
   components.append(c);explained.append(float(values[order[:128]].sum()/max(values.sum(),1e-20)))
 components=np.asarray(components);array(D/'pca_mean.npy',mean);array(D/'pca_components.npy',components)
 for source,base in BASES.items():
  for v in VIDEOS:
   a=np.load(base/'appearance'/f'{v}.features.npy',mmap_mode='r');out=np.empty((len(a),2,128),np.float32)
   with threadpool_limits(limits=4):
    for lo in range(0,len(a),512):out[lo:lo+512]=np.einsum('njc,jcd->njd',np.asarray(a[lo:lo+512],float)-mean,components,optimize=True)
   assert np.isfinite(out).all();array(D/source/'appearance'/f'{v}.projected.npy',out);a._mmap.close();monitor.check()
 atomic_json(R/'pca.json',dict(unique_training_observations=len(unique),training_videos=OLD_CFG['train'],weights='track balanced times source prior^0.5 times geometry quality',explained_variance=explained,backbone_frozen=True,dimensions=128,validation_used_for_fitting=False))
