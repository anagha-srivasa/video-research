"""Detector-owned observations and identity; current moments never require a contour fit."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from pipeline.common import ROOT
from .contracts import CloudMeasure
from .belief_filter import Belief, BeliefTracker
from .dorsal_flux import measure, extract_proposal_lab_moments
from .appearance_reid import VisualReID, crop_region


@dataclass(frozen=True)
class Observation:
    box: np.ndarray
    cloud: CloudMeasure
    confidence: float
    source: str
    mask: np.ndarray | None = None
    partial: bool = False  # Only supplied evidence may assert partial visibility.

    def __post_init__(self):
        box=np.asarray(self.box,float).copy()
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>1) or np.any(box[2:]<=box[:2]):
            raise ValueError('Normalized nonempty observation box required')
        if not np.isfinite(self.confidence) or not 0<=self.confidence<=1:raise ValueError('Invalid detector confidence')
        if type(self.partial) is not bool:raise ValueError('Partial visibility must be explicit boolean evidence')
        if self.source not in ('detector_mask','detector_bbox'):raise ValueError('Direct detector evidence required')
        box.setflags(write=False);object.__setattr__(self,'box',box)
        if self.mask is not None:
            mask=np.asarray(self.mask,bool).copy()
            if mask.ndim!=2 or mask.sum()<3:raise ValueError('Nonempty support mask required')
            mask.setflags(write=False);object.__setattr__(self,'mask',mask)


def bbox_cloud(box, lab):
    box=np.asarray(box,float);extent=box[2:]-box[:2];centre=(box[:2]+box[2:])/2
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>1) or np.any(extent<=0):
        raise ValueError('Valid normalized rectangle required')
    lab=np.asarray(lab,float)
    if lab.shape!=(6,) or not np.isfinite(lab).all():raise ValueError('Six Lab moments required')
    scatter=np.diag(extent**2/12)+np.eye(2)*1e-9
    theta=np.arange(36)*2*np.pi/36
    support=(extent[0]*np.abs(np.cos(theta))+extent[1]*np.abs(np.sin(theta)))/2
    fourier=np.r_[support.mean(),[2*np.mean(support*np.cos(k*theta)) for k in range(1,5)],
                  [2*np.mean(support*np.sin(k*theta)) for k in range(1,5)]]
    signature=np.zeros(24,np.float32);signature[:4]=np.hypot(fourier[1:5],fourier[5:9]);signature[4:10]=lab
    signature[10:12]=np.diag(scatter)
    # Quantization proxy plus a separately configured tracker localization floor, not calibration.
    return CloudMeasure(float(extent.prod()),centre,scatter,fourier,np.eye(2)*1e-6,signature,0.)


class DetectorObserver:
    """One native-image detector pass; absent/invalid masks fall back to observed boxes."""
    def __init__(self,checkpoint=None,device='cpu',max_detections=40):
        self.checkpoint=Path(checkpoint or ROOT/'work/yoloe-26s-seg-pf.pt').resolve()
        if not self.checkpoint.is_file() or not self.checkpoint.is_relative_to(ROOT):raise FileNotFoundError(self.checkpoint)
        if max_detections<1:raise ValueError('Positive detection limit required')
        self.device=device;self.max_detections=max_detections;self.model=None;self.calls=0

    def __call__(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('uint8 RGB required')
        if self.model is None:
            import os
            os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/track1_fmr_config'))
            os.environ.setdefault('YOLO_AUTOINSTALL','false')
            from ultralytics import YOLOE
            self.model=YOLOE(str(self.checkpoint))
        result=self.model.predict(source=np.ascontiguousarray(rgb[...,::-1]),device=self.device,
            conf=.25,imgsz=640,max_det=self.max_detections,verbose=False)[0]
        self.calls+=1;output=[]
        if result.boxes is None:return output
        boxes=result.boxes.xyxyn.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
        polygons=[] if result.masks is None else result.masks.xyn
        for index,(box,score) in enumerate(zip(boxes,scores)):
            box=np.clip(box.astype(float),0,1)
            if np.any(box[2:]<=box[:2]):continue
            mask=np.zeros((160,160),np.uint8)
            if index<len(polygons):
                points=np.asarray(polygons[index],float)
                if points.ndim==2 and points.shape[1]==2 and len(points)>=3 and np.isfinite(points).all():
                    cv2.fillPoly(mask,[np.clip(np.rint(points*160),0,159).astype(np.int32)],1)
            lo=np.floor(box[:2]*160).astype(int);hi=np.ceil(box[2:]*160).astype(int)
            has_mask=mask.sum()>=3 and mask[lo[1]:hi[1],lo[0]:hi[0]].any()
            if not has_mask:
                lo=np.floor(box[:2]*160).astype(int);hi=np.ceil(box[2:]*160).astype(int)
                mask[lo[1]:hi[1],lo[0]:hi[0]]=1
            lab=extract_proposal_lab_moments(rgb,box,mask.astype(bool))
            rectangle=bbox_cloud(box,lab)
            observed=measure(None,mask,lab_moments=lab) if has_mask else rectangle
            output.append((Observation(box,rectangle,float(score),'detector_bbox'),
                           Observation(box,observed,float(score),'detector_mask' if has_mask else 'detector_bbox',mask if has_mask else None)))
        return output


@dataclass(frozen=True)
class Carrier:
    track_id: int
    belief: Belief
    box: np.ndarray
    mode: str
    observation: Observation | None

    @property
    def observed(self):return self.observation is not None


class ObservationIdentityCarrier:
    """Fresh detector evidence owns observation; flow only predicts. IDs remain provisional."""
    def __init__(self,mode='appearance',representation='mask',max_tracks=40):
        if mode not in ('detector','motion','appearance','flow') or representation not in ('bbox','mask'):
            raise ValueError('Unknown ablation')
        self.mode=mode;self.representation=representation
        self.tracker=BeliefTracker(measurement_floor=.02,acceleration_noise=.01,max_tracks=max_tracks,
            use_appearance=mode in ('appearance','flow'),enable_reid=mode in ('appearance','flow'),use_containment=False)
        self.last_observations={};self.last_time=None;self.previous_gray=None;self.previous_support={}
        self.regions={};self.next_detection_id=0;self.encoder=VisualReID();self.reference_rgb={};self.embeddings={}
        self.frame=0;self.last_embedding_frame=-10;self.last_telemetry={};self.image_shape=None

    def _flow(self,gray,time):
        if self.previous_gray is None or time-self.last_time>.1:return {}
        points=[];owners=[]
        for tid,observation in self.previous_support.items():
            mask=np.zeros_like(gray)
            if observation.mask is not None:mask=cv2.resize(observation.mask.astype(np.uint8),gray.shape[::-1],interpolation=cv2.INTER_NEAREST)
            else:
                lo=np.floor(observation.box[:2]*gray.shape[::-1]).astype(int);hi=np.ceil(observation.box[2:]*gray.shape[::-1]).astype(int)
                mask[lo[1]:hi[1],lo[0]:hi[0]]=1
            found=cv2.goodFeaturesToTrack(self.previous_gray,8,.02,3,mask=mask)
            if found is not None:points.extend(found[:,0]);owners.extend([tid]*len(found))
        if not points:return {}
        p=np.asarray(points,np.float32)[:,None,:]
        q,forward,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,p,None,winSize=(21,21),maxLevel=3)
        if q is None:return {}
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,q,None,winSize=(21,21),maxLevel=3)
        if back is None:return {}
        good=(forward[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<=1.)
        good &= np.isfinite(q[:,0]).all(1)&(q[:,0]>=0).all(1)&(q[:,0]<np.array(gray.shape[::-1])).all(1)
        displacement=(q[:,0]-p[:,0])/np.array(gray.shape[::-1]);owners=np.asarray(owners)
        result={}
        for tid in set(owners):
            selected=good&(owners==tid)
            if selected.sum()>=3 and tid in self.tracker.tracks:
                vectors=displacement[selected];cov=np.cov(vectors,rowvar=False,bias=True)+np.eye(2)*1e-5
                result[int(tid)]=(np.median(vectors,axis=0),cov)
        return result

    def step(self,rgb,time,detections):
        if not np.isfinite(time) or (self.last_time is not None and time<=self.last_time):raise ValueError('Increasing source time required')
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('uint8 RGB required')
        if self.image_shape is not None and rgb.shape!=self.image_shape:raise ValueError('Reset between image geometries')
        self.image_shape=rgb.shape
        observations=[pair[0 if self.representation=='bbox' else 1] for pair in detections]
        self.frame+=1;gray=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(256,256)) if self.mode=='flow' else None
        clouds=[item.cloud for item in observations];flow=self._flow(gray,time) if self.mode=='flow' else {}
        if self.mode=='detector':
            carriers={}
            for observation in observations:
                cloud=observation.cloud;cov=np.zeros((4,4));cov[:2,:2]=cloud.obs_cov_c+np.eye(2)*.02**2;cov[2:,2:]=np.eye(2)*.01
                tid=self.next_detection_id;self.next_detection_id+=1
                carriers[tid]=Carrier(tid,Belief(time,np.r_[cloud.centroid,[0.,0.]],cov,cloud.shape_scatter,'OBSERVED'),observation.box,'OBSERVED',observation)
            self.last_telemetry=dict(births=len(carriers),coasted=0,redetected=0,lost=0,rejected=0,flow_predictions=0)
        else:
            before=set(self.tracker.tracks)
            def compare(ids,index):
                if self.frame-self.last_embedding_frame<10 or any(tid not in self.reference_rgb for tid in ids):return {}
                patch=crop_region(rgb,clouds[index])
                if patch is None:return {}
                missing=[tid for tid in ids if tid not in self.embeddings]
                vectors=self.encoder.encode_many([self.reference_rgb[tid] for tid in missing]+[patch]);self.last_embedding_frame=self.frame
                for tid,vector in zip(missing,vectors):self.embeddings[tid]=vector
                return {tid:float(self.embeddings[tid]@vectors[-1]) for tid in ids}
            result=self.tracker.update(time,clouds,flow_advection=flow,
                reid_compare=compare if self.mode in ('appearance','flow') else None)
            by_cloud={id(observation.cloud):observation for observation in observations};carriers={}
            for tid,state in result['tracks'].items():
                cloud=result['clouds'].get(tid);observation=None if cloud is None else by_cloud[id(cloud)]
                if observation is not None:
                    # Current visible geometry can deform freely; localization uncertainty stays separate.
                    state=Belief(time,state.mean,state.covariance,cloud.shape_scatter,'OBSERVED');self.tracker.tracks[tid]=state
                    self.regions[tid]=observation
                    patch=crop_region(rgb,cloud)
                    if patch is not None:self.reference_rgb[tid]=patch;self.embeddings.pop(tid,None)
                prior=self.regions.get(tid)
                if prior is None:continue
                shift=state.mean[:2]-prior.cloud.centroid
                box=observation.box if observation is not None else np.clip(prior.box+np.tile(shift,2),0,1)
                mode=('REDETECTED' if tid in result['resurrected'] else 'PARTIAL' if observation.partial else 'OBSERVED') if observation is not None else 'COASTED'
                carriers[tid]=Carrier(tid,state,box,mode,observation)
            retained=set(self.tracker.tracks)|set(self.tracker.dormant_entities)
            for mapping in (self.regions,self.reference_rgb,self.embeddings):
                for tid in set(mapping)-retained:mapping.pop(tid,None)
            self.last_telemetry=dict(births=sum(tid not in before and tid not in result['resurrected'] for tid in result['clouds']),
                coasted=sum(not c.observed for c in carriers.values()),redetected=len(result['resurrected']),
                lost=len(self.tracker.dormant_entities),rejected=result['rejected_capacity'],flow_predictions=len(flow),
                savpe_forwards=self.encoder.calls)
        self.last_observations={tid:c.observation for tid,c in carriers.items() if c.observed}
        self.previous_support=self.last_observations.copy();self.previous_gray=gray;self.last_time=float(time)
        return carriers
