"""Lazy local YOLOE SAVPE embeddings for ambiguous archived-entity comparisons."""
from pathlib import Path
import cv2
import numpy as np
from pipeline.common import ROOT


class VisualReID:
    def __init__(self):
        self.checkpoint=ROOT/'work/yoloe-26s-seg.pt'
        self.model=None
        self.calls=0

    def encode(self,rgb):
        return self.encode_many([rgb])[0]

    def encode_many(self, images):
        import os
        import torch
        if self.model is None:
            if not self.checkpoint.is_file():raise FileNotFoundError(self.checkpoint)
            os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/track1_fmr_config'))
            os.environ.setdefault('YOLO_AUTOINSTALL','false')
            from ultralytics import YOLOE
            self.model=YOLOE(str(self.checkpoint));self.model.model.cpu().eval()
        image=torch.from_numpy(np.stack([cv2.resize(rgb,(320,320)).transpose(2,0,1) for rgb in images])).float()/255.
        prompt=torch.ones((len(images),1,40,40))
        with torch.inference_mode():embedding=self.model.get_visual_pe(image,prompt).reshape(len(images),-1).cpu().numpy()
        if not np.isfinite(embedding).all() or np.any(np.linalg.norm(embedding,axis=1)<1e-8):raise ValueError('Invalid SAVPE embedding')
        self.calls+=1
        return embedding/np.linalg.norm(embedding,axis=1,keepdims=True)


def crop_region(rgb, cloud):
    height,width=rgb.shape[:2];centre=cloud.centroid*[width,height]
    extent=np.maximum(4,3*np.sqrt(np.diag(cloud.shape_scatter))*[width,height])
    left,top=np.maximum(0,(centre-extent).astype(int))
    right,bottom=np.minimum([width,height],np.ceil(centre+extent).astype(int))
    if right<=left or bottom<=top:return None
    return cv2.resize(rgb[top:bottom,left:right],(96,96))
