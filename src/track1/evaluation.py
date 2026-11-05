"""Manual-mask fidelity, uncertainty reports, seam diagnostics and sequence indexing."""
from collections import defaultdict
import json,logging
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import pyarrow.parquet as pq
from pipeline import Dataset
from pipeline.common import ROOT,atomic_json,digest
from .artifacts import output_dir,load_shard,save_index,manifest,identity
from .numerics import spline_basis
from .spatial import soft_masks

LOG=logging.getLogger('track1')

def geometry_metrics(source,fit):
    source=source.astype(np.uint8);fit=fit.astype(np.uint8)
    intersection=np.sum((source>0)&(fit>0));union=np.sum((source>0)|(fit>0))
    bs=source-cv2.erode(source,np.ones((3,3),np.uint8));bf=fit-cv2.erode(fit,np.ones((3,3),np.uint8))
    ds=cv2.distanceTransform(1-bs,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);df=cv2.distanceTransform(1-bf,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    precision=float(np.mean(ds[bf>0]<=2)) if bf.any() else 0.;recall=float(np.mean(df[bs>0]<=2)) if bs.any() else 0.
    distance=float((ds[bf>0].mean()+df[bs>0].mean())/2) if bf.any() and bs.any() else float('inf')
    return dict(iou=float(intersection/max(union,1)),boundary_fscore_2px=2*precision*recall/max(precision+recall,1e-12),symmetric_boundary_distance_px=distance)

def evaluate_video(cfg,choice):
    v=choice['video_id'];out=output_dir(cfg,'spatial');s=load_shard(out/f'{v}.pt');rows=pq.read_table(out/f'{v}.parquet').to_pylist();phi=spline_basis()
    calibration=json.loads((output_dir(cfg,'calibration')/'model.json').read_text());metrics=[]
    manual=[r for r in rows if r['spatial_valid'] and r['source_kind']=='manual_sparse']
    with Dataset() as ds:
        for i in np.linspace(0,len(manual)-1,min(24,len(manual)),dtype=int):
            r=manual[i];idx=r['row_index'];target=ds.get_mask(r['mask_uid']);height,width=target.shape
            curve=phi@s['controls'][idx];annotation=curve*[width/r['width'],height/r['height']]
            fitted=np.zeros_like(target);cv2.fillPoly(fitted,[np.rint(annotation).astype(np.int32)],1)
            result=geometry_metrics(target,fitted)
            n,labels,stats,_=cv2.connectedComponentsWithStats(target.astype(np.uint8),connectivity=8)
            dominant=(labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))).astype(np.uint8) if n>1 else target
            dom=geometry_metrics(dominant,fitted)
            distances=cKDTree(curve).query(s['boundary'][idx])[0]
            metrics.append(dict(mask_uid=r['mask_uid'],component_count=r['component_count'],spatial_rmse_px=r['spatial_rmse_px'],contour_nearest_rmse_px=float(np.sqrt(np.mean(distances**2))),full_mask=result,dominant_component=dom))
        chosen=[]
        for kind in cfg['provenance_priors']:
            subset=[r for r in rows if r['spatial_valid'] and r['source_kind']==kind]
            if subset:chosen.extend([subset[0],subset[len(subset)//2]])
        chosen=chosen[:6];sheet=Image.new('RGB',(1920,1040),(20,20,20));draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(chosen):
            idx=r['row_index'];rgb=ds.get_frame(v,r['timestamp_us']);image=np.asarray(Image.fromarray(rgb).resize((640,360))).copy()
            curve=phi@s['controls'][idx];source=s['boundary'][idx];std=np.sqrt(s['normal_variance'][idx]*calibration['kappa_cov'])
            scale=np.array([640/r['width'],360/r['height']]);p=np.rint(curve*scale).astype(np.int32);y=np.rint(source*scale).astype(np.int32)
            cv2.polylines(image,[y],True,(255,90,70),1);cv2.polylines(image,[p],True,(50,255,230),2)
            for k in range(0,len(curve),4):
                n=s['normals'][idx,k]*std[k]*1.96
                a,b=np.rint((curve[k]+np.array([-n,n]))*scale).astype(np.int32);cv2.line(image,tuple(a),tuple(b),(180,230,60),1)
            x=j%3*640;y0=j//3*520;sheet.paste(Image.fromarray(image),(x,y0))
            lines=[f'{v} | {r["timestamp_us"]/1e6:.6f}s | {r["entity_name"]}',f'{r["source_kind"]} | track={r["entity_track_id"][:14]}',
                f'RMSE={r["spatial_rmse_px"]:.2f} native px | seam shift={r["seam_shift"]}',
                f'components={r["component_count"]} holes={r["hole_count"]} | pose={r["pose_valid"]}',
                'Red: observed dominant contour; cyan: cubic spline','Yellow: calibrated 95% normal ribbon; no metric depth',r['identity_evidence']]
            for k,line in enumerate(lines):draw.text((x+6,y0+368+k*20),line[:100],fill='white')
        image_path=ROOT/f'reports/track1_spatial.{v}.jpg';sheet.save(image_path,quality=92)
    adjacent=[];first_delta=[]
    for prev,current in zip(rows,rows[1:]):
        if prev['entity_track_id']==current['entity_track_id'] and current['timestamp_us']-prev['timestamp_us']<=350000:
            a=s['controls'][prev['row_index']];b=s['controls'][current['row_index']];delta=np.linalg.norm(a-b,axis=1)
            adjacent.append(float(np.median(delta)));first_delta.append(float(delta[0]))
    ratio=np.asarray(first_delta)/np.maximum(adjacent,1e-6)
    result=dict(video_id=v,manual_geometry_samples=metrics,manual_median_full_mask_iou=float(np.median([m['full_mask']['iou'] for m in metrics])),
        manual_median_dominant_iou=float(np.median([m['dominant_component']['iou'] for m in metrics])),
        median_seam_to_neighbour_motion_ratio=float(np.median(ratio)) if len(ratio) else None,
        large_first_control_motion_proxy=int(np.sum(ratio>4)),seam_metric_limit='Proxy for indexing issues; not annotated physical phase jumps.',
        visual_review='pending',image_sha256=digest(image_path),spatial_shard_sha256=digest(out/f'{v}.pt'))
    atomic_json(ROOT/f'reports/track1_evaluation.{v}.json',result);return result

def sequence_indices(cfg,choices,length=16):
    output=ROOT/cfg['output']/'training_v1';output.mkdir(exist_ok=True);splits=defaultdict(list);all_rows=[]
    for choice in choices:
        v=choice['video_id'];rows=pq.read_table(output_dir(cfg,'functional')/f'{v}.parquet').to_pylist();all_rows.extend(rows);by_track=defaultdict(list)
        for r in rows:by_track[r['entity_track_id']].append(r)
        for track,rs in by_track.items():
            for begin in range(len(rs)-length+1):
                sequence=rs[begin:begin+length]
                if not all(r['valid'] for r in sequence):continue
                if not all(b['window_end_us']-a['window_end_us']==250000 for a,b in zip(sequence,sequence[1:])):continue
                r=sequence[0];splits[r['engineering_split']].append(dict(video_id=v,entity_track_id=track,row_indices=[s['row_index'] for s in sequence],sequence_length=length,
                    start_us=r['window_start_us'],end_us=sequence[-1]['window_end_us'],action_split=r['action_split'],visor_split=r['visor_split'],engineering_split=r['engineering_split'],
                    source_compatible_segmentation_training=r['visor_split']=='train',source_compatible_action_training=r['action_split']=='train'))
    for split in ['train','validation','test']:save_index(output/f'{split}_index.parquet',splits[split])
    save_index(output/'functional_index.parquet',all_rows)
    # Scalar metadata is consolidated; tensor shards stay per video and mmap-loadable.
    for stage in ['spatial','functional']:
        paths=[output_dir(cfg,stage)/f'{c["video_id"]}.parquet' for c in choices]
        pq.write_table(__import__('pyarrow').concat_tables([pq.read_table(p) for p in paths]),output_dir(cfg,stage)/'index.parquet',compression='zstd')
    report=dict(sequence_length=length,sequence_counts={k:len(v) for k,v in splits.items()},window_count=len(all_rows),valid_windows=sum(r['valid'] for r in all_rows),
        split_policy='Whole video/participant before windows; official action and VISOR assignments remain independent. Engineering split is not an official benchmark.',
        train_videos=cfg['engineering_splits']['train'],validation_videos=cfg['engineering_splits']['validation'],test_videos=cfg['engineering_splits']['test'])
    atomic_json(output/'manifest.json',dict(**identity(cfg),**report));return report
