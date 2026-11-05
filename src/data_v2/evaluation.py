"""Matched source-contour reconstruction after correcting ring roles/hierarchy."""
import collections,json,time
import numpy as np
import pyarrow.parquet as pq
from pipeline.geometry import rasterize
from track1.spatial import resample
from track1_lab.architecture import general_fit
from track1.evaluation import geometry_metrics
from refinement.kernels import periodic_design
from refinement.data import spatial_cases
from refinement.protocol import rank as old_rank
from .geometry import normalize,select,render
from .common import *

def fit_rings(rings):
    allocation=[32] if len(rings)==1 else [24,8] if len(rings)==2 else [16,8,8]
    fitted=[];valid=True;controls=[]
    for ring,m in zip(rings,allocation):
        f=general_fit(resample(ring['points'],128),m=m);valid &= f['valid']
        points=(periodic_design(np.arange(512)/512,m=m)@f['controls']).tolist()
        fitted.append(dict(ring,points=points));controls.append(f['controls'].astype(np.float32).tolist())
    return fitted,controls,valid

def worker(v):
    import cv2
    cv2.setNumThreads(1);split(v)
    # Same 64 per-video manual probes as the historical architecture experiment.
    paired=sorted([c for c in spatial_cases([v]) if c['source_kind']=='manual_sparse'],key=lambda c:old_rank('architecture'+c['mask_uid']))[:64]
    wanted={r['mask_uid'] for r in paired};source={}
    hole_extra=[]
    for batch in pq.ParquetFile(OUT/f'masks.{v}.parquet').iter_batches(batch_size=256):
        for r in batch.to_pylist():
            if r['mask_uid'] in wanted:source[r['mask_uid']]=r
            if r['source_kind']=='manual_sparse' and r['hole_count'] and r['geometry_complete']:
                hole_extra.append(r)
    for r in sorted(hole_extra,key=lambda r:rank(r['mask_uid']))[:32]:source[r['mask_uid']]=r
    records=[];fit_records=[]
    for uid,s in source.items():
        guard();shape=(s['annotation_height'],s['annotation_width']);target=rasterize(json.loads(s['segments_raw']),shape[1],shape[0]);rings=normalize(s['contours_json'])
        if not rings:continue
        main=max((r for r in rings if r['role']=='outer' and len(r['points'])>=3),key=lambda r:abs(r['signed_area']))
        # Historical implementation: rank extra raw rings and paint only outer roles.
        legacy=[main]+sorted([r for r in rings if r is not main and len(r['points'])>=3],key=lambda r:-abs(r['signed_area']))[:2]
        corrected=select([dict(r) for r in rings],3)
        for variant,chosen in [('legacy_role_mismatch',legacy),('geometry_contract_v2',corrected)]:
            fitted,controls,valid=fit_rings(chosen)
            if variant=='legacy_role_mismatch':
                mask=np.zeros(shape,np.uint8)
                for r in fitted:
                    if r['role']=='outer':cv2.fillPoly(mask,[np.rint(r['points']).astype(np.int32)],1)
            else:mask=render(fitted,shape)
            gm=geometry_metrics(target,mask)
            records.append(dict(video_id=v,split=split(v),mask_uid=uid,cohort='matched_manual' if uid in wanted else 'additional_hole_manual',variant=variant,
                                source_holes=s['hole_count'],source_components=s['component_count'],represented_rings=len(chosen),source_rings=len(rings),fit_valid=bool(valid),**gm))
            if variant=='geometry_contract_v2':fit_records.append(dict(video_id=v,mask_uid=uid,roles=[r['role'] for r in fitted],parents=[r['parent_ring'] for r in fitted],controls_json=json.dumps(controls),fit_valid=bool(valid)))
    save(REPORT/f'geometry.{v}.parquet',records);save(OUT/f'geometry_fits.{v}.parquet',fit_records)
    return records

def run():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as ex:
        records=[r for rows in ex.map(worker,VIDEOS) for r in rows]
    save(REPORT/'geometry.parquet',records);summary=[]
    for part in ['train','validation']:
        for cohort in ['matched_manual','additional_hole_manual']:
            for variant in ['legacy_role_mismatch','geometry_contract_v2']:
                rs=[r for r in records if r['split']==part and r['cohort']==cohort and r['variant']==variant]
                if not rs:continue
                summary.append(dict(split=part,cohort=cohort,variant=variant,masks=len(rs),hole_masks=sum(r['source_holes']>0 for r in rs),
                                    iou=float(np.mean([r['iou'] for r in rs])),boundary_fscore_2px=float(np.mean([r['boundary_fscore_2px'] for r in rs])),
                                    aggregation='pooled masks; per-video evidence retained',invalid_fits=sum(not r['fit_valid'] for r in rs)))
    atomic_json(REPORT/'geometry.json',dict(metrics=summary,limits=['Retrospective contour reconstruction only. No new causal motion or segmentation benchmark.',
        'Corrected bounded ring hierarchy/role fitting; incomplete source representation when >3 rings.',
        'No new temporal state, appearance PCA, calibration or continuous certificate is promoted. Those old artifacts remain superseded where schema-dependent.'],code=identity()))
