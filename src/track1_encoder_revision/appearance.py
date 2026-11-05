"""Bounded frozen-CNN extraction for new contexts; no RGB payload on disk."""
import io,json,struct,subprocess,sys
import numpy as np
import pyarrow.parquet as pq
from pipeline.geometry import rasterize
from track1.numerics import basis
from track1.appearance import backbone
from track1_reference.frame_checks import read_exact
from .experiments import ROOT,D,R,W,array,save

def run(monitor):
 import torch,cv2
 torch.set_num_threads(2);cv2.setNumThreads(1);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=backbone().to(device);mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];pca_mean=np.load(ROOT/'derived/track1_finetuned/pca_mean.npy');pca=np.load(ROOT/'derived/track1_finetuned/pca_components.npy');summary=[];all_sequences=[]
 def resize(a):
  factor=min(224/a.shape[1],224/a.shape[0]);w=max(1,round(a.shape[1]*factor));h=max(1,round(a.shape[0]*factor));canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[(224-h)//2:(224-h)//2+h,(224-w)//2:(224-w)//2+w]=cv2.resize(a,(w,h),interpolation=cv2.INTER_AREA);return canvas
 for context in json.loads((R/'contexts.json').read_text())['videos']:
  if not context['selected']:continue
  v=context['video_id'];inputs=pq.read_table(W/'contexts'/f'{v}.inputs.parquet').to_pylist();native={r['timestamp_us']:r for r in pq.read_table(D/f'{v}.context_frames.parquet').to_pylist()};frames=[native[r['timestamp_us']] for r in inputs];request=W/'contexts'/f'{v}.decode.json';request.write_text(json.dumps(dict(video_id=v,frames=frames)));proc=subprocess.Popen([sys.executable,'-m','track1_encoder_revision.source_stream',str(request)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT);features=np.zeros((len(inputs),2,576),np.float32);seen=set();batch=[];indices=[]
  def flush():
   if not batch:return
   tensor=torch.from_numpy(np.stack(batch)).permute(0,3,1,2).to(device,dtype=torch.float32)/255
   with torch.inference_mode():out=model((tensor-mean)/std).mean((-1,-2)).cpu().numpy().reshape(-1,2,576)
   features[indices]=out;batch.clear();indices.clear();monitor.check()
  try:
   while True:
    header=proc.stdout.read(8)
    if not header:break
    if len(header)!=8:raise ValueError('Truncated native RGB pipe')
    i,size=struct.unpack('<II',header)
    if size!=1920*1080*3 or not 0<=i<len(inputs) or i in seen:raise ValueError('Invalid native frame pipe')
    rgb=np.frombuffer(read_exact(proc.stdout,size),np.uint8).reshape(1080,1920,3);r=inputs[i];mask=rasterize(json.loads(r['segments_raw']),r['annotation_width'],r['annotation_height']);yy,xx=np.nonzero(mask);sx=1920/r['annotation_width'];sy=1080/r['annotation_height'];x0=max(0,int(np.floor(xx.min()*sx))-8);x1=min(1920,int(np.ceil((xx.max()+1)*sx))+8);y0=max(0,int(np.floor(yy.min()*sy))-8);y1=min(1080,int(np.ceil((yy.max()+1)*sy))+8);xs=np.clip(np.floor(np.arange(x0,x1)/sx).astype(int),0,mask.shape[1]-1);ys=np.clip(np.floor(np.arange(y0,y1)/sy).astype(int),0,mask.shape[0]-1);foreground=mask[np.ix_(ys,xs)].astype(bool);crop=rgb[y0:y1,x0:x1];obj=crop.copy();obj[~foreground]=[124,116,104];batch.extend([resize(obj),resize(crop)]);indices.append(i);seen.add(i)
    if len(batch)>=16:flush()
   error=proc.stderr.read().decode(errors='replace')
   if proc.wait()!=0:raise ValueError(error[-2000:])
   flush()
  finally:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  if len(seen)!=len(inputs) or not np.isfinite(features).all():raise ValueError('Incomplete feature extraction')
  projected=np.einsum('njc,jcd->njd',features-pca_mean,pca);array(D/f'{v}.context_features.npy',features);array(D/f'{v}.context_projected.npy',projected);states=pq.read_table(D/f'{v}.context_states.parquet').to_pylist();coeff=np.zeros((len(states),4,2,128),np.float32);lookup={r['mask_uid']:i for i,r in enumerate(inputs)}
  for state in states:
   if not state['valid']:continue
   ids=[lookup[u] for u in state['retained_mask_uids']];t=(np.array([inputs[i]['timestamp_us'] for i in ids])-state['window_start_us'])/1e6;phi=basis(2*t-1);y=projected[ids].reshape(len(ids),256);priors={'manual_sparse':1.,'dense_gt_endpoint':.9,'dense_interpolated':.6};w=np.array([priors[inputs[i]['source_kind']]**.5 for i in ids]);w*=len(w)/w.sum();coeff[state['row_index']]=np.linalg.solve(phi.T@(w[:,None]*phi)+np.diag([1e-8,1e-8,.01,.09]),phi.T@(w[:,None]*y)).reshape(4,2,128);state['appearance_valid']=True
  array(D/f'{v}.context_appearance.npy',coeff);save(D/f'{v}.context_states.parquet',states);from track1_retrain.store import sequences
  for length in [4,8,16]:
   for row in sequences([r for r in states if r['model']==16 and r['appearance_valid']],length):all_sequences.append(dict(row,episode_id=states[row['row_indices'][0]]['episode_id'],source='stable_context_extension'))
  summary.append(dict(video_id=v,source_observations=len(inputs),exact_native_frames=len(seen),backbone_frozen=True,pca_locked=True,appearance_states=sum(r['appearance_valid'] for r in states),new_rgb_bytes=0));monitor.check()
 from pipeline.common import atomic_json
 for part in ['train','validation']:
  for length in [4,8,16]:
   local=[r for r in all_sequences if r['split']==part and r['sequence_length']==length]
   if local:save(D/f'{part}.context_length{length}.parquet',local)
 atomic_json(R/'context_appearance.json',dict(videos=summary,device=str(device),new_rgb_bytes=0,pca_and_backbone_frozen=True,source_masks_privileged=True))
