"""Full-partition frozen CNN extraction; balanced train-only PCA, object/context separate."""
import collections,json,time
import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import eigh
from .common import *

def fingerprint(v):
 return dict(inputs_sha256=digest(WORK/f'inputs.{v}.parquet'),extractor_sha256=digest(ROOT/'src/track1_retrain/appearance.py'),config_sha256=digest(ROOT/'configs/track1_retrain.json'),weights_sha256=digest(ROOT/'work/track1_weights/mobilenet_v3_small-047dcff4.pth'))

def run(monitor):
 import torch,cv2
 from track1.appearance import backbone
 from pipeline.geometry import rasterize
 from data_v2.store import Dataset
 torch.manual_seed(CFG['seed']);torch.set_num_threads(2);cv2.setNumThreads(1);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=backbone().to(device);out=folder('appearance');mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];summaries=[]
 def resize(a):
  scale=min(224/a.shape[1],224/a.shape[0]);w=max(1,round(a.shape[1]*scale));h=max(1,round(a.shape[0]*scale));canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[(224-h)//2:(224-h)//2+h,(224-w)//2:(224-w)//2+w]=cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA);return canvas
 with Dataset() as ds:
  for v in VIDEOS:
   marker=out/f'{v}.json';target=out/f'{v}.features.npy';fp=fingerprint(v)
   if marker.exists() and target.exists():
    old=json.loads(marker.read_text())
    if old['inputs']==fp and old['output_sha256']==digest(target):summaries.append(old);LOG.info('Reused appearance %s',v);continue
   rows=pq.read_table(WORK/f'inputs.{v}.parquet').to_pylist();lookup={r['mask_uid']:i for i,r in enumerate(rows)};bytime=collections.defaultdict(list)
   for r in rows:bytime[r['timestamp_us']].append(r)
   temp=target.with_suffix('.npy.tmp');features=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(rows),2,576));features[:]=0;batch=[];indices=[];done=0;started=time.monotonic()
   def flush():
    nonlocal done
    if not batch:return
    tensor=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).to(device,dtype=torch.float32)/255
    with torch.inference_mode():z=model((tensor-mean)/std).mean((-1,-2)).cpu().numpy().reshape(-1,2,576)
    if not np.isfinite(z).all():raise ValueError('Nonfinite CNN features')
    features[indices]=z;done+=len(indices);batch.clear();indices.clear();monitor.check()
   for frame_no,(timestamp,rs) in enumerate(sorted(bytime.items())):
    rgb=ds.get_frame(v,timestamp)
    for r in rs:
     mask=rasterize(json.loads(r['segments_raw']),r['annotation_width'],r['annotation_height']);ys,xs=np.nonzero(mask)
     sx=r['width']/r['annotation_width'];sy=r['height']/r['annotation_height'];x0=max(0,int(np.floor(xs.min()*sx))-8);x1=min(r['width'],int(np.ceil((xs.max()+1)*sx))+8);y0=max(0,int(np.floor(ys.min()*sy))-8);y1=min(r['height'],int(np.ceil((ys.max()+1)*sy))+8)
     # Crop the annotation mask before resizing: preserve global integer-grid mapping
     # exactly via remap, including native pixel-centre nearest-neighbour transport.
     xx=np.floor(np.arange(x0,x1)/sx).astype(int);yy=np.floor(np.arange(y0,y1)/sy).astype(int);native_mask=mask[np.ix_(np.clip(yy,0,mask.shape[0]-1),np.clip(xx,0,mask.shape[1]-1))].astype(bool);crop=rgb[y0:y1,x0:x1];obj=crop.copy();obj[~native_mask]=[124,116,104];batch.extend([resize(obj),resize(crop)]);indices.append(lookup[r['mask_uid']])
     if len(batch)>=CFG['appearance_batch']:flush()
    if frame_no%64==0:ds.close()
    if frame_no%256==0:
     features.flush();LOG.info('%s appearance frames=%d/%d observations=%d/%d elapsed=%.1fs',v,frame_no,len(bytime),done,len(rows),time.monotonic()-started)
   flush();features.flush();del features;temp.replace(target)
   # Export only stable metadata; no raw polygons copied into the training feature index.
   save(out/f'{v}.parquet',[{k:r[k] for k in ['video_id','split','mask_uid','entity_track_id','timestamp_us','source_kind','dependency_group']}|dict(row_index=i) for i,r in enumerate(rows)])
   result=dict(video_id=v,observations=len(rows),frames=len(bytime),seconds=time.monotonic()-started,device=str(device),inputs=fp,output_sha256=digest(target));atomic_json(marker,result);summaries.append(result)
 # Balance provisional track contributions, then provenance. Dense frame repetition
 # does not create additional independent identities or human supervision.
 total=0.;sums=np.zeros((2,576));cross=np.zeros((2,576,576));counts={};training_n=0
 for v in CFG['train']:
  rows=pq.read_table(out/f'{v}.parquet').to_pylist();a=np.load(out/f'{v}.features.npy',mmap_mode='r');pertrack=collections.Counter(r['entity_track_id'] for r in rows)
  weights=np.array([CFG['source_kind_priors'][r['source_kind']]/pertrack[r['entity_track_id']] for r in rows]);total+=weights.sum();training_n+=len(rows);counts[v]=len(pertrack)
  for lo in range(0,len(rows),512):
   x=a[lo:lo+512].astype(float);w=weights[lo:lo+512];sums+=np.einsum('n,njc->jc',w,x);cross+=np.einsum('n,njc,njd->jcd',w,x,x)
  del a;monitor.check()
 means=sums/total;components=[];variance=[]
 for j in range(2):
  cov=cross[j]/total-np.outer(means[j],means[j]);values,vectors=eigh((cov+cov.T)/2);order=np.argsort(values)[::-1];c=vectors[:,order[:128]]
  for k in range(128):
   if c[np.argmax(abs(c[:,k])),k]<0:c[:,k]*=-1
  components.append(c);variance.append(float(values[order[:128]].sum()/max(values.sum(),1e-20)))
 array(out/'mean.npy',means);array(out/'components.npy',components)
 for v in VIDEOS:
  a=np.load(out/f'{v}.features.npy',mmap_mode='r');p=np.empty((len(a),2,128),np.float32)
  for lo in range(0,len(a),512):p[lo:lo+512]=np.einsum('njc,jcd->njd',a[lo:lo+512]-means,np.asarray(components))
  array(out/f'{v}.projected.npy',p);del a,p
 atomic_json(REPORT/'appearance.json',dict(videos=summaries,training_observations=training_n,train_tracks=counts,training_weight_sum=float(total),train_explained_variance=variance,projection_dimensions=128,selection='all eligible partition observations; train-only track-balanced PCA',source_masks_privileged=True,object_and_context_separate=True,independent_identity_accuracy=None,device=str(device),code=code()))
