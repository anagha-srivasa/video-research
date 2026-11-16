"""Independent manual-mask support diagnostic; semantic hand roles, not object IDF1."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_belief import data
from track1_fmr.pipeline import Pipeline


def raster(row, size=160):
    mask=np.zeros((size,size),np.uint8)
    for polygon in json.loads(row['segments_raw']):
        points=np.asarray(polygon)/[row['annotation_width'],row['annotation_height']]*size
        if len(points)>=3:cv2.fillPoly(mask,[np.clip(np.rint(points),0,size-1).astype(np.int32)],1)
    return mask.astype(bool)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',default='stage2_visor_v3');ap.add_argument('--ncc-threshold',type=float,default=.78);ap.add_argument('--adaptation-rate',type=float,default=.5);args=ap.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    report=ROOT/'reports/track1_fmr'/args.tag;report.mkdir(exist_ok=False)
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    episodes=[];hashes={}
    # Disjoint development/validation participants. Both are historically consumed, not fresh holdouts.
    for video,split in [('P02_122','dev'),('P08_09','validation'),('P06_108','reserved_sequence'),('P21_01','reserved_sequence')]:
        mp=ROOT/f'canonical/v1/masks.{video}.parquet';fp=ROOT/f'canonical/v1/frames.{video}.parquet'
        hashes[str(mp.relative_to(ROOT))]=digest(mp);hashes[str(fp.relative_to(ROOT))]=digest(fp)
        rows=[r for r in pq.read_table(mp).to_pylist() if r['source_kind']=='manual_sparse' and r['entity_name_open']=='right hand']
        rows.sort(key=lambda r:r['timestamp_us']);frames=pq.read_table(fp).to_pylist()
        selected=[]
        for a,b in zip(rows,rows[1:]):
            if not 0<b['timestamp_us']-a['timestamp_us']<=6e6:continue
            if json.loads(a['entity_scope'])[1]!=json.loads(b['entity_scope'])[1]:continue
            clip=[f for f in frames if a['timestamp_us']<=f['timestamp_us']<=b['timestamp_us']]
            if not clip or clip[0]['timestamp_us']!=a['timestamp_us'] or clip[-1]['timestamp_us']!=b['timestamp_us']:continue
            if split=='reserved_sequence' and not selected:
                selected.append(None);continue
            selected.append(dict(video_id=video,split=split,frames=clip,references=[a,b]))
            break
        episodes.extend(e for e in selected if e is not None)
    if len(episodes)!=4:raise ValueError('Need four multi-timestamp manual sequences')
    atomic_json(report/'protocol.json',dict(episodes=episodes,input_hashes=hashes,
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),*sorted((ROOT/'src/track1_fmr').glob('*.py'))]},
        source='VISOR manual_sparse, https://epic-kitchens.github.io/VISOR/site',
        ncc_threshold=args.ncc_threshold,adaptation_rate=args.adaptation_rate,identity_scope='Unique right-hand semantic role within one external annotation subsequence; no inferred canonical entity IDs',
        independence='External human masks withheld from inference; reference used once for evaluator-only binding',
        split_scope='P02 development and P08 validation; P06/P21 second feasible subsequences reserved before this run. All participants historically consumed; not fresh participant holdout'))
    protocol=json.loads((report/'protocol.json').read_text())
    for rel in protocol['code_hashes']:
        destination=report/'source'/rel;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/rel).read_bytes())
    results=[];data.W=ROOT/'work/track1_fmr'
    for ep in episodes:
        pipeline=Pipeline(dorsal_grid=160,ncc_threshold=args.ncc_threshold,adaptation_rate=args.adaptation_rate);bytime={r['timestamp_us']:r for r in ep['references']};fixed=None;rows=[]
        for index,rgb,decode_seconds in data.native_frames(ep['video_id'],ep['frames'],ep['video_id']+'.visor_stage2'):
            timestamp=ep['frames'][index]['timestamp_us'];pipeline.step(rgb,timestamp/1e6)
            if timestamp not in bytime:continue
            ref=bytime[timestamp];mask=raster(ref);yy,xx=np.nonzero(mask)
            centre=np.array([xx.mean()/160,yy.mean()/160])
            if index==0:fixed=pipeline.bind_seed(centre)
            state=pipeline.tracker.tracks.get(fixed);observed=state is not None and state.status=='OBSERVED'
            support=np.zeros((160,160),np.uint8)
            if observed:
                from track1_fmr.belief_filter import _ellipse_polygon
                polygon,_area=_ellipse_polygon(state.mean[:2],state.shape_scatter)
                cv2.fillPoly(support,[np.rint(polygon*160).astype(np.int32)],1)
            covered=int((mask & support.astype(bool)).sum())
            rows.append(dict(timestamp_us=timestamp,mask_uid=ref['mask_uid'],fixed_id=fixed,observed=observed,
                initialization_frame=index==0,reference_pixels=int(mask.sum()),covered_pixels=covered,
                support_recall=covered/int(mask.sum()),measurement_source=getattr(pipeline,'last_measurement_sources',{}).get(fixed),
                centroid_error_px=None if not observed else float(np.linalg.norm((state.mean[:2]-centre)*[1920,1080]))))
        pipeline.close();results.append(dict(video_id=ep['video_id'],split=ep['split'],frames=len(ep['frames']),references=rows))
        atomic_json(report/'progress.json',results);print(ep['video_id'],rows,flush=True)
    subsequent=[r['support_recall'] for e in results for r in e['references'] if not r['initialization_frame']]
    atomic_json(report/'result.json',dict(episodes=results,subsequent_support_recalls=subsequent,support_gate_passed=bool(subsequent) and all(v>=.5 for v in subsequent),independent_human_masks=True,identity_idf1=None,
        stage2_promoted=False,qualification_scope='Two sparse hand-role sequences only; ellipse support is not predicted segmentation'))


if __name__=='__main__':main()
