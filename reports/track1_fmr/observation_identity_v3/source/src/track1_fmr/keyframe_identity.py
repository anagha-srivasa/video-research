"""Adaptive detector keyframes and image-correspondence carriers, with provisional lineage."""
from dataclasses import dataclass, field
import json
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pipeline.common import ROOT, digest, atomic_json
from .observation_identity import DetectorObserver, bbox_cloud, Observation
from .dorsal_flux import measure, extract_proposal_lab_moments
from .belief_filter import Belief, advance, condition
from .appearance_reid import VisualReID, crop_region

PROMPTS=('person','hand','arm','glove','cup','mug','bowl','plate','pan','pot','knife','spoon','fork','bottle','jar','tool','sponge','towel','box','bag','phone','book','ball','vehicle')

PROMPTS_V3=('person','hand','human hand','arm','glove','cup','mug','bowl','plate','pan','pot','kettle','knife','spoon','fork','bottle','jar','can','lid','box','bag','packet','towel','cloth','sponge','brush','tool','scissors','cutting board','sink','faucet','phone','laptop','book','paper','pen','chair','table','ball','car','bicycle','dog','cat','horse')


class KeyframeDetector(DetectorObserver):
    """Two local models; text embeddings are computed once and content-addressed."""
    def __init__(self, device='cpu', imgsz=None, revision='v2'):
        super().__init__(checkpoint=ROOT/'work/yoloe-26s-seg.pt' if revision=='v3' else None,device=device)
        if revision not in ('v2','v3'):raise ValueError('Unknown revision')
        self.revision=revision;self.prompts=PROMPTS_V3 if revision=='v3' else PROMPTS
        self.imgsz=imgsz or (480 if revision=='v3' else 640); self.text_model=None; self.warmup_seconds=0

    def warmup(self):
        if self.text_model is not None:return
        import os
        import torch
        os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/track1_fmr_config'))
        os.environ.setdefault('YOLO_AUTOINSTALL','false')
        from ultralytics import YOLOE
        start=time.perf_counter()
        self.model=YOLOE(str(self.checkpoint)) if self.revision=='v2' else None
        self.text_model=YOLOE(str(ROOT/'work/yoloe-26s-seg.pt'))
        weights=ROOT/'work/mobileclip2_b.ts'
        signature=dict(prompts=self.prompts,checkpoint=digest(ROOT/'work/yoloe-26s-seg.pt'),text_encoder=digest(weights))
        import hashlib
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        cache=ROOT/'work/track1_fmr_text_cache'/key;cache.mkdir(parents=True,exist_ok=True)
        if (cache/'embedding.npy').is_file():
            if json.loads((cache/'manifest.json').read_text())['embedding_sha256']!=digest(cache/'embedding.npy'):raise ValueError('Text cache changed')
            embedding=torch.from_numpy(np.load(cache/'embedding.npy',allow_pickle=False))
        else:
            from ultralytics.nn.text_model import MobileCLIPTS
            self.text_model.model.clip_model=MobileCLIPTS(torch.device(self.device),weight=str(weights))
            embedding=self.text_model.model.get_text_pe(list(self.prompts),cache_clip_model=True).detach().cpu()
            np.save(cache/'embedding.npy',embedding.numpy())
            atomic_json(cache/'manifest.json',dict(**signature,embedding_sha256=digest(cache/'embedding.npy')))
            del self.text_model.model.clip_model
        self.text_model.set_classes(list(self.prompts),embedding)
        self.cache_manifest=cache/'manifest.json'
        for model in ([self.model,self.text_model] if self.revision=='v2' else [self.text_model]):
            model.predict(np.zeros((self.imgsz,self.imgsz,3),np.uint8),device=self.device,conf=.1,imgsz=self.imgsz,max_det=40,verbose=False)
        self.warmup_seconds=time.perf_counter()-start

    def __call__(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('uint8 RGB required')
        self.warmup();output=[]
        for model in ([self.model,self.text_model] if self.revision=='v2' else [self.text_model]):
            result=model.predict(np.ascontiguousarray(rgb[...,::-1]),device=self.device,conf=.1,imgsz=self.imgsz,max_det=40,verbose=False)[0]
            if result.boxes is None:continue
            polygons=[] if result.masks is None else result.masks.xyn
            for i,(box,confidence) in enumerate(zip(result.boxes.xyxyn.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                box=np.clip(box.astype(float),0,1)
                if np.any(box[2:]<=box[:2]):continue
                mask=np.zeros((160,160),np.uint8)
                if i<len(polygons) and len(polygons[i])>=3:
                    cv2.fillPoly(mask,[np.clip(np.rint(np.asarray(polygons[i])*160),0,159).astype(np.int32)],1)
                valid=mask.sum()>=3
                if not valid:
                    lo=np.floor(box[:2]*160).astype(int);hi=np.ceil(box[2:]*160).astype(int);mask[lo[1]:hi[1],lo[0]:hi[0]]=1
                lab=extract_proposal_lab_moments(rgb,box,mask.astype(bool))
                cloud=measure(None,mask,lab_moments=lab) if valid else bbox_cloud(box,lab)
                output.append(Observation(box,cloud,float(confidence),'detector_mask' if valid else 'detector_bbox',mask.astype(bool) if valid else None))
        # Suppress duplicate masks across heads, while retaining nested hand/person supports.
        kept=[]
        for obs in sorted(output,key=lambda o:-o.confidence):
            duplicate=False
            for old in kept:
                if obs.mask is not None and old.mask is not None:
                    intersection=(obs.mask&old.mask).sum();union=(obs.mask|old.mask).sum()
                    if intersection/max(1,union)>.8:duplicate=True;break
            if not duplicate:kept.append(obs)
            if len(kept)>=40:break
        self.calls+=1
        return kept


@dataclass
class Entity:
    track_id:int
    belief:Belief
    box:np.ndarray
    observation:Observation
    anchors:np.ndarray=field(default_factory=lambda:np.empty((0,2),np.float32))
    evidence:str='detector'
    hits:int=1
    last_seen:float=0
    parent_ids:tuple=()
    displacement_old:np.ndarray=field(default_factory=lambda:np.empty((0,2)))
    displacement_new:np.ndarray=field(default_factory=lambda:np.empty((0,2)))
    fb_errors:np.ndarray=field(default_factory=lambda:np.empty(0))
    measurement:np.ndarray|None=None
    measurement_covariance:np.ndarray|None=None
    previous_box:np.ndarray|None=None
    anchor_patches:np.ndarray=field(default_factory=lambda:np.empty((0,5,5),np.float32))
    ncc_scores:np.ndarray=field(default_factory=lambda:np.empty(0))
    motion_delta:np.ndarray=field(default_factory=lambda:np.zeros(2))
    motion_dt:float=0.

    @property
    def observed(self):return self.evidence in ('detector','fb_lk')


class KeyframeObservationIdentityTracker:
    """No reference input. Failed anchors are discarded, never advected from predictions."""
    def __init__(self,keyframe_stride=6,measurement_floor=None,acceleration_noise=2.40508678,revision='v2'):
        if keyframe_stride<1:raise ValueError('Positive keyframe stride required')
        if revision not in ('v2','v3'):raise ValueError('Unknown revision')
        self.revision=revision;self.q_lk=.015
        self.stride=keyframe_stride;self.floor=(.012 if revision=='v3' else .00979414048) if measurement_floor is None else measurement_floor;self.q=acceleration_noise
        self.entities={};self.dormant={};self.next_id=0;self.frame=0;self.last_keyframe=-keyframe_stride
        self.encoder=VisualReID();self.patches={};self.embeddings={};self.last_rescue=-10
        self.gray=None;self.time=None;self.lineage=[];self.last_telemetry={};self.last_reasons=[]

    @staticmethod
    def transport(point,old,new):
        if len(old)<4:return np.asarray(point).copy()
        distances=np.linalg.norm(old-point,axis=1)
        weights=1/np.maximum(distances,1e-4)**2;weights/=weights.sum()
        return np.asarray(point)+(weights[:,None]*(new-old)).sum(0)

    def transport_entity(self,point,entity):
        if len(entity.displacement_old)>=4:
            return self.transport(point,entity.displacement_old,entity.displacement_new)
        if entity.evidence=='detector' and entity.previous_box is not None:
            old=entity.previous_box;extent=old[2:]-old[:2]
            if np.all(extent>1e-6):return entity.box[:2]+(np.asarray(point)-old[:2])/extent*(entity.box[2:]-entity.box[:2])
        return np.asarray(point).copy()

    @staticmethod
    def prepare_shared_gray_320(rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        while min(gray.shape)//2>=320:gray=cv2.pyrDown(gray)
        return gray if gray.shape==(320,320) else cv2.resize(gray,(320,320))

    @staticmethod
    def anchor_patches(gray,anchors):
        return np.asarray([cv2.getRectSubPix(gray,(5,5),tuple(map(float,p*320))) for p in anchors],np.float32).reshape(-1,5,5)

    @staticmethod
    def patch_ncc(reference,current):
        a=reference.reshape(-1,25).astype(float);b=current.reshape(-1,25).astype(float)
        a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
        norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        return np.divide((a*b).sum(1),norm,out=np.full(len(a),-1.),where=norm>1e-4)

    def _anchors(self,gray,obs):
        if self.revision=='v2':
            mask=np.zeros_like(gray)
            if obs.mask is not None:mask=cv2.resize(obs.mask.astype(np.uint8),gray.shape[::-1],interpolation=cv2.INTER_NEAREST)
            else:
                lo=(obs.box[:2]*320).astype(int);hi=np.ceil(obs.box[2:]*320).astype(int);mask[lo[1]:hi[1],lo[0]:hi[0]]=1
            distance=cv2.distanceTransform(mask,cv2.DIST_L2,3);interior=(distance>max(1.,.3*distance.max())).astype(np.uint8)
            anchors=cv2.goodFeaturesToTrack(gray,8,.01,5,mask=interior)
            return np.empty((0,2),np.float32) if anchors is None else anchors[:,0]/320
        mask=np.zeros((160,160),np.uint8)
        if obs.mask is not None:mask=obs.mask.astype(np.uint8)
        else:
            lo=(obs.box[:2]*160).astype(int);hi=np.ceil(obs.box[2:]*160).astype(int);mask[lo[1]:hi[1],lo[0]:hi[0]]=1
        if self.revision=='v3':mask=cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
        mask=cv2.resize(mask,gray.shape[::-1],interpolation=cv2.INTER_NEAREST)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,3)
        interior=(distance>max(1.,.3*distance.max())).astype(np.uint8)
        anchors=cv2.goodFeaturesToTrack(gray,12 if self.revision=='v3' else 8,.01,5,mask=interior)
        return np.empty((0,2),np.float32) if anchors is None else anchors[:,0]/320

    def _flow(self,gray):
        owners=[];points=[];references=[]
        for tid,e in self.entities.items():
            e.displacement_old=np.empty((0,2));e.displacement_new=np.empty((0,2));e.fb_errors=np.empty(0);e.ncc_scores=np.empty(0)
            if self.revision=='v3':references.extend(e.anchor_patches)
            owners.extend([tid]*len(e.anchors));points.extend(e.anchors)
        if self.gray is None or not points:return {}
        p=np.asarray(points,np.float32)[:,None,:]*320
        q,forward,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,p,None,winSize=(9,9) if self.revision=='v3' else (21,21),maxLevel=3)
        if q is None:return {}
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,q,None,winSize=(9,9) if self.revision=='v3' else (21,21),maxLevel=3)
        if back is None:return {}
        error=np.linalg.norm(back[:,0]-p[:,0],axis=1)
        good=(forward[:,0]>0)&(reverse[:,0]>0)&(error<(1.2 if self.revision=='v3' else 1.5))&np.isfinite(q[:,0]).all(1)&(q[:,0]>=0).all(1)&(q[:,0]<320).all(1)
        ncc=np.ones(len(p))
        if self.revision=='v3':
            current=self.anchor_patches(gray,np.nan_to_num(q[:,0]/320,nan=0,posinf=0,neginf=0))
            ncc=self.patch_ncc(np.asarray(references),current);good &= ncc>=.65
        result={};owners=np.asarray(owners)
        for tid,e in self.entities.items():
            take=good&(owners==tid);old=p[take,0]/320;new=q[take,0]/320
            e.anchors=new.copy()
            if self.revision=='v3':e.anchor_patches=np.asarray(references)[take].copy()
            if len(new)>=4:
                e.displacement_old=old;e.displacement_new=new;e.fb_errors=error[take];e.ncc_scores=ncc[take]
                delta=new-old;r=np.cov(delta,rowvar=False,bias=True)+np.eye(2)*(self.floor**2+1/320**2)
                result[tid]=(np.median(delta,axis=0),r)
        return result

    def step(self,rgb,time,detector,shared_gray_320=None):
        if self.time is not None and time<=self.time:raise ValueError('Increasing source time required')
        gray=(self.prepare_shared_gray_320(rgb) if self.revision=='v3' else cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(320,320))) if shared_gray_320 is None else shared_gray_320
        if gray.shape!=(320,320) or gray.dtype!=np.uint8:raise ValueError('uint8 320x320 gray buffer required')
        previously_trackable={tid for tid,e in self.entities.items() if len(e.anchors)>=4}
        if self.time is None or time-self.time<=.1:flow=self._flow(gray)
        else:
            flow={}
            for e in self.entities.values():
                e.anchors=np.empty((0,2),np.float32);e.anchor_patches=np.empty((0,5,5),np.float32)
                e.displacement_old=np.empty((0,2));e.displacement_new=np.empty((0,2));e.fb_errors=np.empty(0);e.ncc_scores=np.empty(0)
        reasons=[]
        if not self.entities:reasons.append('empty')
        if self.frame-self.last_keyframe>=self.stride:reasons.append('stride')
        confirmed={tid for tid,e in self.entities.items() if e.hits>=2}
        active=previously_trackable or confirmed
        if self.revision=='v2' and active and not (active & set(flow)):reasons.append('flow_loss')
        if self.revision=='v2' and self.gray is not None:
            unexplained=cv2.absdiff(gray,self.gray)>35
            for e in self.entities.values():
                lo=np.clip((e.box[:2]*320).astype(int),0,320);hi=np.clip(np.ceil(e.box[2:]*320).astype(int),0,320)
                unexplained[lo[1]:hi[1],lo[0]:hi[0]]=False
            if unexplained.mean()>.04:reasons.append('unexplained_flux')
        for e in self.entities.values():e.previous_box=e.box.copy()
        prior={tid:e.belief for tid,e in self.entities.items()}
        predictions={tid:advance(e.belief,time,self.q) for tid,e in self.entities.items()}
        # Cap only process uncertainty, never shape; record this heuristic explicitly in protocol.
        for tid,p in (predictions.items() if self.revision=='v2' else []):
            values,vectors=np.linalg.eigh(p.covariance)
            predictions[tid]=Belief(time,p.mean,(vectors*np.minimum(values,.25))@vectors.T,p.shape_scatter,'PREDICTED')
        observations=detector(rgb) if reasons else []
        if reasons:
            self.last_keyframe=self.frame
            for tid,e in list(self.dormant.items()):
                if time-e.last_seen<=10.:
                    self.entities[tid]=e;prior[tid]=e.belief;predictions[tid]=advance(e.belief,time,self.q)
                self.dormant.pop(tid)
        ids=list(self.entities);cost=np.full((len(ids),len(observations)),1e6)
        for i,tid in enumerate(ids):
            e=self.entities[tid];p=predictions[tid]
            for j,obs in enumerate(observations):
                d=obs.cloud.centroid-p.mean[:2];spatial=float(d@np.linalg.solve(p.covariance[:2,:2]+obs.cloud.shape_scatter,d))
                appearance=np.linalg.norm(e.observation.cloud.algebraic_sig[4:7]-obs.cloud.algebraic_sig[4:7])/255
                # Appearance rescue uses the CV prior, not an LK-shifted gating centre.
                if spatial<16 or (appearance<.08 and np.linalg.norm(d)<.35):cost[i,j]=min(spatial,16)+3*appearance
        # Lazy SAVPE rescue of otherwise rejected, spatially plausible pairs. Strong
        # image appearance is an alternative gate; LK never shifts its CV prior.
        rescue=[]
        if self.frame-self.last_rescue>=10:
            for i,tid in enumerate(ids):
                if tid not in self.patches or self.entities[tid].evidence!='predicted' or (self.revision=='v2' and np.any(cost[i]<1e6)):continue
                candidates=[(np.linalg.norm(obs.cloud.centroid-predictions[tid].mean[:2]),j) for j,obs in enumerate(observations)]
                if candidates:
                    distance,j=min(candidates)
                    if distance<(min(.45,.25+2*np.sqrt(np.trace(predictions[tid].covariance[:2,:2]))) if self.revision=='v3' else .35):rescue.append((i,j,tid))
            rescue=rescue[:4]
        if rescue:
            self.last_rescue=self.frame
            missing=sorted(set(tid for _,_,tid in rescue if tid not in self.embeddings))
            js=sorted(set(j for _,j,_ in rescue));patches=[crop_region(rgb,observations[j].cloud) for j in js]
            encoded=self.encoder.encode_many([self.patches[tid] for tid in missing]+patches)
            for tid,vector in zip(missing,encoded):self.embeddings[tid]=vector
            vectors=dict(zip(js,encoded[len(missing):]))
            for i,j,tid in rescue:
                similarity=float(self.embeddings[tid]@vectors[j])
                if similarity>=(.64 if self.revision=='v3' else .8):cost[i,j]=(4*(1-similarity) if self.revision=='v3' else 8+4*(1-similarity))
        assigned={};used=set()
        if cost.size:
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if cost[i,j]<1e6:assigned[ids[i]]=j;used.add(j)
        for tid,e in list(self.entities.items()):
            e.measurement=None;e.measurement_covariance=None;e.motion_delta=np.zeros(2);e.motion_dt=0.
            if tid in assigned:
                obs=observations[assigned[tid]];r=obs.cloud.obs_cov_c+np.eye(2)*self.floor**2
                e.belief=condition(predictions[tid],obs.cloud.centroid,r,obs.cloud.shape_scatter)
                e.box=obs.box.copy();e.observation=obs;e.evidence='detector';e.hits+=1;e.last_seen=time
                e.measurement=obs.cloud.centroid.copy();e.measurement_covariance=obs.cloud.obs_cov_c.copy()
                e.anchors=self._anchors(gray,obs)
                if self.revision=='v3':e.anchor_patches=self.anchor_patches(gray,e.anchors)
                self.patches[tid]=crop_region(rgb,obs.cloud)
            elif tid in flow:
                delta,r=flow[tid];z=prior[tid].mean[:2]+delta
                if self.revision=='v3':
                    dt=time-prior[tid].time;p=predictions[tid];cov=p.covariance.copy()
                    diffusion=r-np.eye(2)*self.floor**2+np.eye(2)*self.q_lk**2*dt
                    cov[:2,:2]+=diffusion
                    e.belief=Belief(time,np.r_[z,delta/dt],cov,p.shape_scatter,'OBSERVED')
                    e.motion_delta=delta.copy();e.motion_dt=dt
                else:e.belief=condition(predictions[tid],z,r)
                e.box=np.clip(e.box+np.tile(delta,2),0,1);e.evidence='fb_lk';e.hits+=1;e.last_seen=time
                e.measurement=z;e.measurement_covariance=r-np.eye(2)*self.floor**2
            else:
                e.belief=predictions[tid];e.evidence='predicted';e.anchors=np.empty((0,2),np.float32);e.anchor_patches=np.empty((0,5,5),np.float32)
                if time-e.last_seen>1.:
                    self.dormant[tid]=e;del self.entities[tid]
                    if len(self.dormant)>64:self.dormant.pop(min(self.dormant,key=lambda k:self.dormant[k].last_seen))
        while len(self.entities)>40:
            candidates=[tid for tid,e in self.entities.items() if not e.observed]
            if not candidates:break
            tid=min(candidates,key=lambda k:self.entities[k].last_seen);self.dormant[tid]=self.entities.pop(tid)
        for j,obs in enumerate(observations):
            if j in used or len(self.entities)>=40:continue
            parents=[]
            for tid,old in prior.items():
                if tid not in self.entities or self.entities[tid].hits<3:continue
                d=obs.cloud.centroid-old.mean[:2]
                if float(d@np.linalg.solve(old.shape_scatter,d))<=2.25**2:parents.append(tid)
            tid=self.next_id;self.next_id+=1
            r=obs.cloud.obs_cov_c+np.eye(2)*self.floor**2;cov=np.zeros((4,4));cov[:2,:2]=r;cov[2:,2:]=np.eye(2)*.01
            e=Entity(tid,Belief(time,np.r_[obs.cloud.centroid,[0.,0.]],cov,obs.cloud.shape_scatter,'OBSERVED'),obs.box.copy(),obs,
                self._anchors(gray,obs),last_seen=time,parent_ids=tuple(parents),measurement=obs.cloud.centroid.copy(),measurement_covariance=obs.cloud.obs_cov_c.copy())
            if self.revision=='v3':e.anchor_patches=self.anchor_patches(gray,e.anchors)
            self.entities[tid]=e
            self.patches[tid]=crop_region(rgb,obs.cloud)
            for parent in parents:self.lineage.append(dict(time=time,parent=parent,child=tid,kind='split_hypothesis'))
        # Merge hypotheses preserve a causal edge to the assigned active support, without certifying identity.
        for tid,e in self.entities.items():
            if tid in assigned:continue
            if tid not in prior:continue
            for target in assigned:
                obs=observations[assigned[target]];d=prior[tid].mean[:2]-obs.cloud.centroid
                if float(d@np.linalg.solve(obs.cloud.shape_scatter,d))<=2.25**2 and prior[tid].time<time:
                    edge=dict(time=time,parent=tid,child=target,kind='merge_hypothesis')
                    if not any(x['parent']==tid and x['child']==target for x in self.lineage[-80:]):self.lineage.append(edge)
        while len(self.dormant)>64:
            self.dormant.pop(min(self.dormant,key=lambda k:self.dormant[k].last_seen))
        for tid in list(self.dormant):
            if time-self.dormant[tid].last_seen>10.:self.dormant.pop(tid)
        for mapping in (self.patches,self.embeddings):
            for tid in set(mapping)-(set(self.entities)|set(self.dormant)):mapping.pop(tid,None)
        self.last_reasons=reasons;self.last_telemetry=dict(keyframe=bool(reasons),reasons=reasons,detector=len(observations),fb_lk=sum(e.evidence=='fb_lk' for e in self.entities.values()),predicted=sum(not e.observed for e in self.entities.values()))
        self.gray=gray;self.time=time;self.frame+=1
        return self.entities

    def follow(self,tid,point):
        """Choose a spatially supported descendant using only the previously predicted query point."""
        root=self.entities.get(tid)
        transported=np.asarray(point) if root is None else self.transport_entity(point,root)
        candidates=[tid]+[x['child'] for x in self.lineage if x['parent']==tid and x['time']==self.time]
        valid=[]
        for candidate in candidates:
            e=self.entities.get(candidate)
            if e is None or not e.observed:continue
            p=transported
            if np.all(p>=e.box[:2]) and np.all(p<=e.box[2:]):
                valid.append((float(np.prod(e.box[2:]-e.box[:2])),candidate))
        return min(valid)[1] if valid else tid
