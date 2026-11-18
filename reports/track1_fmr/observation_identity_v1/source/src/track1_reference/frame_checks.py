"""Native-frame optical-flow diagnostics. Never infers contact or physical identity."""
import hashlib,json,struct,subprocess,sys,time
import cv2
import numpy as np
from . import ROOT,WORK

def read_exact(pipe,n):
 data=bytearray()
 while len(data)<n:
  chunk=pipe.read(n-len(data))
  if not chunk:raise EOFError('Incomplete exact-frame stream')
  data.extend(chunk)
 return bytes(data)

def motion(previous,current,box=None):
 mask=None
 if box is not None:
  mask=np.zeros(previous.shape,dtype=np.uint8);x0,y0,x1,y1=np.asarray(box,dtype=int);mask[max(0,y0):min(previous.shape[0],y1),max(0,x0):min(previous.shape[1],x1)]=255
 points=cv2.goodFeaturesToTrack(previous,maxCorners=400 if box is None else 100,qualityLevel=.015,minDistance=8,mask=mask)
 empty=dict(feature_count=0,retention=0.,median_flow_px=None,p95_flow_px=None,median_fb_px=None,transform=None,flow_valid=False)
 if points is None or len(points)<12:return empty
 q,valid,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
 if q is None:return empty
 back,back_valid,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3)
 if back is None:return empty
 p=points[:,0];out=q[:,0];fb=np.linalg.norm(back[:,0]-p,axis=1);good=(valid.ravel()>0)&(back_valid.ravel()>0)&np.isfinite(out).all(1)&np.isfinite(fb)&(fb<1.)&(out[:,0]>=0)&(out[:,0]<current.shape[1])&(out[:,1]>=0)&(out[:,1]<current.shape[0]);a=p[good];b=out[good];count=len(a)
 if count<12:return dict(empty,feature_count=len(points),retention=count/len(points))
 delta=np.linalg.norm(b-a,axis=1);cv2.setRNGSeed(0);affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=1000,confidence=.995)
 if affine is not None and not np.isfinite(affine).all():affine=None
 fraction=float(inliers.mean()) if inliers is not None else 0.
 return dict(feature_count=len(points),retention=count/len(points),median_flow_px=float(np.median(delta)),p95_flow_px=float(np.quantile(delta,.95)),median_fb_px=float(np.median(fb[good])),transform=affine.tolist() if affine is not None else None,inlier_fraction=fraction,flow_valid=bool(count/len(points)>=.6 and fraction>=.65),p95_residual_proxy_px=float(np.quantile(np.linalg.norm((a@affine[:,:2].T+affine[:,2])-b,axis=1),.95)) if affine is not None else None)

def scan(clip,anchors=None):
 cv2.setNumThreads(2);cv2.setRNGSeed(0);started=time.monotonic();command=[sys.executable,'-m','track1_reference.native_stream',clip];proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE);rows=[];previous=None;active={};anchors=anchors or {};source_frames=json.loads((WORK/'task_map.json').read_text())['clips'][clip]
 try:
  while True:
   head=proc.stdout.read(8)
   if not head:break
   if len(head)!=8:raise EOFError('Incomplete stream header')
   idx,size=struct.unpack('<II',head)
   if idx!=len(rows) or size!=1920*1080:raise ValueError('Unexpected native stream schema')
   raw=read_exact(proc.stdout,size);gray=np.frombuffer(raw,dtype=np.uint8).reshape(1080,1920);metrics=motion(previous,gray) if previous is not None else dict(flow_valid=False,initial_frame=True)
   metrics.update(gray_sha256=hashlib.sha256(raw).hexdigest(),blur_proxy=float(cv2.Laplacian(gray,cv2.CV_64F).var()),mean_luma=float(gray.mean()),dark_fraction=float(np.mean(gray<15)))
   if previous is not None:metrics['mean_abs_luma_change']=float(cv2.absdiff(previous,gray).mean())
   tracks={}
   for group,state in list(active.items()):
    if source_frames[idx]['timestamp_us']-source_frames[state['anchor_index']]['timestamp_us']>500_000:active.pop(group);continue
    local=motion(previous,gray,state['box']) if previous is not None else dict(flow_valid=False)
    valid=local['flow_valid'];newbox=None
    if valid:
     x0,y0,x1,y1=state['box'];corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])@np.array(local['transform'])[:,:2].T+np.array(local['transform'])[:,2];lo=corners.min(0);hi=corners.max(0);valid=bool(np.all(lo>=0) and hi[0]<=1920 and hi[1]<=1080 and np.all(hi-lo>=16))
     if valid:newbox=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])];state['box']=newbox
    tracks[group]=dict(attention_box_native=newbox,feature_consistency=local,anchor_index=state['anchor_index'],valid=valid,semantic_labels_propagated=False,identity_verified=False)
    if not valid:active.pop(group)
   for group,box in anchors.get(idx,[]):
    active[group]=dict(box=box,anchor_index=idx);tracks[group]=dict(attention_box_native=box,anchor_index=idx,valid=True,anchor_only=True,semantic_labels_propagated=False,identity_verified=False)
   metrics['attention_tracks']=tracks
   rows.append((idx,metrics));previous=gray
  error=proc.stderr.read().decode(errors='replace');code=proc.wait()
  if code:raise RuntimeError(error[-2000:])
  return clip,rows,time.monotonic()-started
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
