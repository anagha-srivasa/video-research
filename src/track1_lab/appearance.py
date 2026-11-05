"""Native crop and feature RoI alignment ablations with the frozen MobileNet backbone."""
from collections import defaultdict
import json,subprocess,sys,time
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT,atomic_json,digest
from track1.artifacts import load_shard,save_shard,save_index
from track1.appearance import read_exact,backbone
from refinement.protocol import rank
from refinement.kernels import periodic_design
from .common import config,path,allowed,split,pool,report,LOG,identity
from .geometry_contract import normalize as normalize_rings,render as render_geometry


def stream(video,timestamps,width,height):
 import duckdb
 allowed([video])
 with duckdb.connect() as db:
  db.execute('SET threads=1');pts=dict(db.execute('SELECT pts_raw,timestamp_us FROM read_parquet(?) WHERE timestamp_us IN (SELECT unnest(?))',[str(ROOT/f'canonical/v1/frames.{video}.parquet'),list(map(int,timestamps))]).fetchall())
 assert len(pts)==len(set(timestamps));request=path('work')/f'decode.{video}.{width}.json';atomic_json(request,dict(path=str(ROOT/f'dataset/videos/{video}.MP4'),pts=pts,width=width,height=height));proc=subprocess.Popen([sys.executable,'-m','track1.decoder',str(request)],stdout=subprocess.PIPE)
 try:
  while True:
   line=proc.stdout.readline()
   if not line:break
   header=json.loads(line);yield int(header['timestamp_us']),np.frombuffer(read_exact(proc.stdout,header['bytes']),np.uint8).reshape(header['shape'])
  if proc.wait():raise RuntimeError('Hardware decoder failed '+video)
 finally:
  if proc.poll() is None:proc.terminate();proc.wait()


def select(rows):
 tracks=defaultdict(list)
 for r in rows:
  if r['source_kind']=='manual_sparse' and r['spatial_valid']:tracks[r['entity_track_id']].append(r)
 chosen={}
 for track in sorted(tracks,key=rank):
  rs=tracks[track]
  if len(rs)<3:continue
  for i in np.linspace(0,len(rs)-1,min(4,len(rs)),dtype=int):chosen[rs[i]['mask_uid']]=rs[i]
  if len(chosen)>=72:break
 remaining=sorted([r for rs in tracks.values() for r in rs],key=lambda r:(r['width']*r['height'],rank(r['mask_uid'])))
 # Uniformly hashed extras cover short tracks; explicit small objects added in worker.
 for r in sorted(remaining,key=lambda r:rank(r['mask_uid']))[:24]:chosen[r['mask_uid']]=r
 return chosen


def roi_masks(row,state,source,box,size=224):
 import cv2
 from scipy.special import expit
 i=row['row_index'];x0,y0,x1,y1=box;scale=np.array([size/(x1-x0),size/(y1-y0)]);curve=periodic_design(np.arange(128)/128)@state['controls'][i];p=(curve-[x0,y0])*scale;inside=np.zeros((size,size),np.uint8);cv2.fillPoly(inside,[np.rint(p).astype(np.int32)],1)
 rings=normalize_rings(source['contours_json']);native=np.array([row['width']/source['annotation_width'],row['height']/source['annotation_height']])
 transformed=[dict(ring,points=((np.asarray(ring['points'])*native-[x0,y0])*scale).tolist()) for ring in rings]
 target=render_geometry(transformed,(size,size));keep=np.ones_like(inside)
 for ring in sorted(transformed,key=lambda r:r['depth']):
  if ring['depth']>0 and len(ring['points'])>=3:cv2.fillPoly(keep,[np.rint(ring['points']).astype(np.int32)],0 if ring['role']=='hole' else 1)
 inside*=keep
 dist_in=cv2.distanceTransform(inside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);dist_out=cv2.distanceTransform(1-inside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);signed=dist_in-dist_out;kappa=json.loads((ROOT/'derived/track1_refined_v2/calibration/model.json').read_text())['kappa_cov'];sigma=float(np.sqrt(np.median(state['normal_variance'][i])*kappa)*np.sqrt(scale.prod()));soft=expit((signed-sigma)/max(float(np.sqrt(scale.prod())*row['width']/448),.5));hard=soft*inside
 # Area integration gives nonzero support to sub-cell objects instead of relying on centres.
 return hard.astype(np.float32),soft.astype(np.float32),target


def worker(v):
 import torch,cv2
 from torchvision.ops import roi_align
 from scipy.linalg import eigh
 torch.set_num_threads(2);cv2.setNumThreads(1)
 if not torch.backends.mps.is_available():raise RuntimeError('Metal unavailable')
 torch.mps.set_per_process_memory_fraction(.2);device='mps';model=backbone().to(device);sp=ROOT/'derived/track1_refined_v2/spatial';s=load_shard(sp/f'{v}.pt');rows=pq.read_table(sp/f'{v}.parquet').to_pylist();chosen=select(rows)
 manual=[r for r in rows if r['source_kind']=='manual_sparse' and r['spatial_valid']]
 for r in sorted(manual,key=lambda r:float(s['scale_native'][r['row_index']]))[:24]:chosen[r['mask_uid']]=r
 selected=list(chosen.values());source={r['mask_uid']:r for r in pq.read_table(ROOT/f'canonical/v1/masks.{v}.parquet').to_pylist() if r['mask_uid'] in chosen};bytime=defaultdict(list)
 for r in selected:bytime[r['timestamp_us']].append(r)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];features=defaultdict(list);metadata=[];sensitivity=[];crop_cache=defaultdict(list);begin=time.monotonic();full=np.load(ROOT/f'work/track1_refinement/features/{v}.npy',mmap_mode='r');width,height=rows[0]['width'],rows[0]['height'];images=[]
 for number,(ts,rgb) in enumerate(stream(v,bytime,width,height)):
  if number%20==0:LOG.info('%s ROI frames %d/%d',v,number,len(bytime))
  selected_now=bytime[ts];crops=[];masks=[];boxes=[];truth=[];softs=[]
  for r in selected_now:
   i=r['row_index'];curve=periodic_design(np.arange(128)/128)@s['controls'][i];lo=curve.min(0);hi=curve.max(0);side=max(float(np.max(hi-lo))*1.3,24);centre=(lo+hi)/2;x0=int(max(0,np.floor(centre[0]-side/2)));y0=int(max(0,np.floor(centre[1]-side/2)));x1=int(min(width,np.ceil(centre[0]+side/2)));y1=int(min(height,np.ceil(centre[1]+side/2)));box=(x0,y0,max(x0+1,x1),max(y0+1,y1));crop=cv2.resize(rgb[box[1]:box[3],box[0]:box[2]],(224,224),interpolation=cv2.INTER_AREA);hard,soft,target=roi_masks(r,s,source[r['mask_uid']],box);crops.append(crop);masks.append(hard);softs.append(soft);truth.append(target);boxes.append(box)
  batch=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).to(device,dtype=torch.float32)/255;fullrgb=cv2.resize(rgb,(448,256),interpolation=cv2.INTER_AREA);tensor=torch.from_numpy(fullrgb.copy()).permute(2,0,1)[None].to(device,dtype=torch.float32)/255
  masked=np.stack(crops).copy();spline_masked=masked.copy()
  for jj in range(len(crops)):
   masked[jj][truth[jj]==0]=127;spline_masked[jj][masks[jj]<=0]=127
  mb=torch.from_numpy(np.concatenate([masked,spline_masked])).permute(0,3,1,2).to(device,dtype=torch.float32)/255
  with torch.inference_mode():maps=model((batch-mean)/std).cpu();fm=model((tensor-mean)/std).cpu();masked_maps=model((mb-mean)/std).cpu().numpy()
  fh,fw=fm.shape[-2:];box_t=torch.tensor([[0,b[0]*fw/width,b[1]*fh/height,b[2]*fw/width,b[3]*fh/height] for b in boxes],dtype=torch.float32);aligned=roi_align(fm,box_t,output_size=(7,7),spatial_scale=1.,sampling_ratio=2,aligned=True)
  for j,r in enumerate(selected_now):
   hard=cv2.resize(masks[j],(7,7),interpolation=cv2.INTER_AREA);soft=cv2.resize(softs[j],(7,7),interpolation=cv2.INTER_AREA);fg=cv2.resize(truth[j].astype(np.float32),(7,7),interpolation=cv2.INTER_AREA);den=max(float(hard.sum()),1e-12);m=maps[j].numpy();am=aligned[j].numpy();features['native_crop_hard'].append(np.sum(m*hard[None],axis=(1,2))/den);features['native_crop_soft'].append(np.sum(m*soft[None],axis=(1,2))/max(float(soft.sum()),1e-12));features['feature_roi_align'].append(np.sum(am*hard[None],axis=(1,2))/den);features['fullframe_v2'].append(np.array(full[r['row_index']]));geometry=s['h'][r['row_index'],2:];features['geometry_only'].append(geometry);features['source_masked_crop'].append(np.sum(masked_maps[j]*hard[None],axis=(1,2))/den);features['spline_masked_crop'].append(np.sum(masked_maps[j+len(crops)]*hard[None],axis=(1,2))/den);crop_cache['rgb'].append(crops[j]);crop_cache['source_mask'].append(truth[j]);crop_cache['spline_mask'].append((masks[j]>0).astype(np.uint8));crop_cache['pool_weights'].append(hard.astype(np.float32))
   centre_fg=truth[j][(np.arange(7)*32+16)[:,None],(np.arange(7)*32+16)[None,:]];metadata.append(dict(video_id=v,split=split(v),mask_uid=r['mask_uid'],entity_track_id=r['entity_track_id'],entity_name=r['entity_name'],class_id=r['class_id'],timestamp_us=ts,row_index=len(metadata),scale_native=float(s['scale_native'][r['row_index']]),small_object=bool(s['scale_native'][r['row_index']]<32),crop_box=list(boxes[j]),core_mass=float(hard.sum()),foreground_area_precision=float(np.sum(hard*fg)/den),foreground_cell_centre_precision=float(np.sum(hard*centre_fg)/den),has_foreground_cells=bool(centre_fg.any())))
   if len(images)<6:images.append((crops[j],truth[j],r['entity_name'],float(s['scale_native'][r['row_index']])))
   if j==0 and number<12:
    # Background intervention keeps RGB foreground and the pooling geometry fixed.
    altered=crops[j].copy();altered[truth[j]==0]=np.array([127,127,127],np.uint8);a=torch.from_numpy(altered).permute(2,0,1)[None].to(device,dtype=torch.float32)/255
    with torch.inference_mode():mm=model((a-mean)/std).cpu().numpy()[0]
    pooled=np.sum(mm*hard[None],axis=(1,2))/den;base=features['native_crop_hard'][-1];cos=float(np.dot(base,pooled)/max(np.linalg.norm(base)*np.linalg.norm(pooled),1e-12));spline_alt=altered.copy();spline_alt[masks[j]<=0]=127;ab=torch.from_numpy(spline_alt).permute(2,0,1)[None].to(device,dtype=torch.float32)/255
    with torch.inference_mode():sm=model((ab-mean)/std).cpu().numpy()[0]
    spooled=np.sum(sm*hard[None],axis=(1,2))/den;original=features['spline_masked_crop'][-1];scos=float(np.dot(original,spooled)/max(np.linalg.norm(original)*np.linalg.norm(spooled),1e-12));sensitivity.append(dict(video_id=v,mask_uid=r['mask_uid'],small_object=bool(s['scale_native'][r['row_index']]<32),cosine_after_background_replacement=cos,spline_masked_cosine_after_background_replacement=scos,source_masked_input_identical=bool(np.array_equal(altered,masked[j]))))
  del maps,aligned,batch,fm,tensor,mb,masked_maps
 from PIL import Image,ImageDraw
 sheet=Image.new('RGB',(1344,270),(20,20,20));draw=ImageDraw.Draw(sheet)
 for i,(im,mask,name,scale) in enumerate(images):sheet.paste(Image.fromarray(im),(i*224,0));draw.text((i*224+4,230),name[:24],fill='white');draw.text((i*224+4,248),f'scale={scale:.1f}px',fill='white')
 sheet.save(path('reports')/f'appearance.{v}.jpg',quality=92);save_shard(path('derived')/f'appearance.{v}.pt',{k:np.stack(a).astype(np.float32) for k,a in features.items()});save_index(path('derived')/f'appearance.{v}.parquet',metadata);save_index(path('reports')/f'background_sensitivity.{v}.parquet',sensitivity)
 save_shard(path('work')/f'roi_cache.{v}.pt',{k:np.stack(a) for k,a in crop_cache.items()})
 return dict(video_id=v,observations=len(metadata),frames=len(bytime),seconds=time.monotonic()-begin)


def summarize(monitor):
 from scipy.linalg import eigh
 cfg=config();features=defaultdict(list);meta=[];sensitivity=[]
 for v in cfg['train']+cfg['validation']:
  shard=load_shard(path('derived')/f'appearance.{v}.pt')
  for k,a in shard.items():features[k].append(a)
  meta+=pq.read_table(path('derived')/f'appearance.{v}.parquet').to_pylist();sensitivity+=pq.read_table(path('reports')/f'background_sensitivity.{v}.parquet').to_pylist()
 train=np.array([r['split']=='train' for r in meta]);embeddings={};pca={}
 for method,arrays in features.items():
  a=np.concatenate(arrays).astype(float);mu=a[train].mean(0);centred=a-mu
  if method=='geometry_only':
   sd=np.maximum(a[train].std(0),.01);z=centred/sd;pca[method]=dict(dimension=z.shape[1],note='Geometry shape/scale only, standardized on training data; not an appearance model.')
  else:
   cov=centred[train].T@centred[train]/train.sum();values,vectors=eigh(cov);ix=np.argsort(values)[::-1];components=vectors[:,ix[:128]];z=centred@components;pca[method]=dict(dimension=128,training_observations=int(train.sum()),training_explained_variance=float(values[ix[:128]].sum()/max(values.sum(),1e-20)));save_shard(path('derived')/f'projection.{method}.pt',dict(mean=mu.astype(np.float32),components=components.astype(np.float32)))
  embeddings[method]=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
 ids=np.array([r['entity_track_id'] for r in meta]);videos=np.array([r['video_id'] for r in meta]);classes=np.array([r['class_id'] for r in meta]);times=np.array([r['timestamp_us'] for r in meta]);retrieval={}
 for method,z in embeddings.items():
  scores=[]
  for i,r in enumerate(meta):
   if r['split']!='validation':continue
   keep=(videos==videos[i])&(classes==classes[i])&(abs(times-times[i])>=500000);positive=keep&(ids==ids[i]);negative=keep&(ids!=ids[i])
   if not positive.any() or not negative.any():continue
   idx=np.flatnonzero(keep);sim=z[idx]@z[i];scores.append(dict(correct=bool(ids[idx[sim.argmax()]]==ids[i]),small=r['small_object'],positive_similarity=float(np.mean(z[positive]@z[i])),negative_similarity=float(np.mean(z[negative]@z[i]))))
  retrieval[method]=dict(queries=len(scores),provisional_chain_top1=float(np.mean([r['correct'] for r in scores])) if scores else None,same_chain_mean_cosine=float(np.mean([r['positive_similarity'] for r in scores])) if scores else None,different_chain_mean_cosine=float(np.mean([r['negative_similarity'] for r in scores])) if scores else None)
 groups={}
 for group in ['all','small','other']:
  rs=[r for r in meta if group=='all' or r['small_object']==(group=='small')];groups[group]=dict(observations=len(rs),empty_foreground_cells=sum(not r['has_foreground_cells'] for r in rs),mean_area_foreground_precision=float(np.mean([r['foreground_area_precision'] for r in rs])),mean_cell_centre_foreground_precision=float(np.mean([r['foreground_cell_centre_precision'] for r in rs])),usable_core_fraction=float(np.mean([r['core_mass']>1e-4 for r in rs])))
 report('appearance_ablation.json',dict(localization=groups,pca=pca,provisional_identity_retrieval=retrieval,background_replacement=dict(n=len(sensitivity),mean_cosine=float(np.mean([r['cosine_after_background_replacement'] for r in sensitivity])),spline_masked_mean_cosine=float(np.mean([r['spline_masked_cosine_after_background_replacement'] for r in sensitivity])),source_masked_inputs_identical_fraction=float(np.mean([r['source_masked_input_identical'] for r in sensitivity]))),limits=['Native ROI crops are resized before the frozen CNN; feature_roi_align instead resamples the low-resolution full-frame feature map using torchvision roi_align aligned=True.','Pooling uses spline interior and source-hole exclusion, never hard clipping to the evaluation foreground mask. Source mask supplies only localization evaluation and the controlled RGB background intervention.','Source-masked crops remove background using the source contour supplied to Track 1; this is an additional preprocessing variant, not a learned segmentation result. Spline-masked crops use only the represented spline plus source holes.',
        'Provisional source-chain retrieval is not independent identity truth. Separate chains may be the same physical entity; no IDF1/HOTA claim.','Train-only PCA is fitted anew for each appearance variant, using identical training samples. Geometry-only is a separate 33D baseline.'],state='experimental_not_finalized',test_videos_used=False,**identity()));monitor.check()


def run(monitor):
 cfg=config();out=pool(worker,cfg['train']+cfg['validation'],monitor,workers=3);report('appearance_extraction.json',dict(videos=out,**identity()));summarize(monitor)
