"""Projection audit, distortion correction and nonmetric geometry."""
import json
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from pipeline.common import ROOT,atomic_json,digest
from pipeline import Dataset

def intrinsics(camera,width,height):
    p=np.asarray(camera['params']);sx=width/camera['width'];sy=height/camera['height']
    return np.array([[p[0]*sx,0,p[2]*sx],[0,p[1]*sy,p[3]*sy],[0,0,1.]]),p[4:8]

def undistort(points,k,distortion):
    return cv2.undistortPoints(np.asarray(points,np.float64).reshape(-1,1,2),k,distortion).reshape(-1,2)

def rotation(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

def projection_audit(video):
    """Test candidate resize by static-point reprojection and shifted negative controls.

    Photometric scores are diagnostics, not metric calibration or dynamic-object depth.
    Visual review is required before enabling the transform.
    """
    raw=ROOT/f'dataset/camera/{video}.json';data=json.loads(raw.read_text());points=np.asarray(data['points'],float)
    # Deterministic bounded sample; no point-cloud duplication on disk.
    if len(points)>50000:points=points[np.linspace(0,len(points)-1,50000,dtype=int)]
    with Dataset() as ds:
        poses=ds.rows('SELECT * FROM camera_frames WHERE video_id=? AND pose_valid ORDER BY timestamp_us',[video])
        choice=[poses[int((len(poses)-1)*f)] for f in [.15,.5,.85]]
        sheets=[];scores=[]
        for row in choice:
            rgb=ds.get_frame(video,row['timestamp_us']);height,width=rgb.shape[:2]
            k,dist=intrinsics(data['camera'],width,height)
            transform=np.asarray(row['T_world_to_camera']).reshape(4,4)
            xyz=points[:,:3]@transform[:3,:3].T+transform[:3,3]
            uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),k,dist);uv=uv[:,0]
            visible=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            uv=uv[visible];colors=points[visible,3:6];depth=xyz[visible,2]
            # Z-buffer at a four-pixel grid to reduce occluded static points.
            cell=(uv[:,1]//4).astype(int)*(width//4+1)+(uv[:,0]//4).astype(int)
            order=np.lexsort((depth,cell));_,first=np.unique(cell[order],return_index=True);keep=order[first]
            uv=uv[keep];colors=colors[keep]
            errors=[]
            blurred=cv2.GaussianBlur(rgb,(7,7),0)
            for dx,dy in [(0,0),(-80,0),(80,0),(0,-80),(0,80)]:
                xy=np.rint(uv+[dx,dy]).astype(int);good=(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
                errors.append(float(np.median(np.linalg.norm(blurred[xy[good,1],xy[good,0]].astype(float)-colors[good],axis=1))))
            small=np.asarray(Image.fromarray(rgb).resize((640,360))).copy()
            for xy,color in zip(uv,colors):cv2.circle(small,tuple(np.rint(xy*[640/width,360/height]).astype(int)),1,tuple(map(int,color)),-1)
            panel=Image.new('RGB',(640,420),(20,20,20));panel.paste(Image.fromarray(small));draw=ImageDraw.Draw(panel)
            draw.text((6,365),f'{video} | t={row["timestamp_us"]/1e6:.3f}s | {len(uv)} static projections',fill='white')
            draw.text((6,383),f'RGB median error actual={errors[0]:.1f} shifted={np.median(errors[1:]):.1f}',fill='white')
            draw.text((6,401),'Candidate pure resize; coloured dots = static reconstruction',fill='white')
            sheets.append(panel);scores.append(dict(timestamp_us=row['timestamp_us'],projected_points=len(uv),rgb_error=errors[0],shifted_controls=errors[1:]))
    sheet=Image.new('RGB',(1920,420))
    for i,p in enumerate(sheets):sheet.paste(p,(640*i,0))
    image=ROOT/f'reports/track1_camera.{video}.jpg';sheet.save(image,quality=92)
    report=dict(video_id=video,source_sha256=digest(raw),image_sha256=digest(image),samples=scores,transform_valid=False,visual_review='pending',
                candidate='pure_resize_without_crop',geometry_scale='reconstruction_units',metric_3d_enabled=False)
    atomic_json(ROOT/f'reports/track1_camera.{video}.json',report)
    return report

def require_validated(video):
    r=json.loads((ROOT/f'reports/track1_camera.{video}.json').read_text())
    if not r['transform_valid'] or r['source_sha256']!=digest(ROOT/f'dataset/camera/{video}.json'):raise ValueError('Camera resize/crop is not verified: '+video)
    return r
