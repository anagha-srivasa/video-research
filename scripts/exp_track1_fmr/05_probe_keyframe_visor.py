"""Detector-only sparse VISOR probes, kept separate from causal keyframe ingestion."""
import argparse
import importlib.util
import json
import time
import cv2
import numpy as np
from pipeline.common import ROOT,atomic_json,digest
from track1_fmr.keyframe_identity import KeyframeDetector
from track1_belief import data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    out=ROOT/'reports/track1_fmr'/args.tag
    if (out/'visor_detector_probes.json').exists():raise FileExistsError('Probe already exists')
    selection_path=ROOT/'reports/track1_fmr/stage2_visor_v3/protocol.json';selection=json.loads(selection_path.read_text())
    for rel,sha in selection['input_hashes'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('Reference changed')
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    spec=importlib.util.spec_from_file_location('reference',ROOT/'scripts/exp_track1_fmr/02a_eval_visor_sparse.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    protocol=json.loads((out/'protocol.json').read_text())
    detector=KeyframeDetector(revision=protocol.get('revision','v2'));detector.warmup();data.W=ROOT/'work/track1_fmr';rows=[];arrays={}
    for ep in selection['episodes']:
        bytime={r['timestamp_us']:r for r in ep['references']};frames=[f for f in ep['frames'] if f['timestamp_us'] in bytime]
        for index,rgb,_ in data.native_frames(ep['video_id'],frames,ep['video_id']+'.keyframe_probe'):
            ts=frames[index]['timestamp_us'];tick=time.perf_counter();observations=detector(rgb);seconds=time.perf_counter()-tick
            masks=[]
            for obs in observations:
                if obs.mask is not None:mask=obs.mask
                else:
                    mask=np.zeros((160,160),bool);lo=np.floor(obs.box[:2]*160).astype(int);hi=np.ceil(obs.box[2:]*160).astype(int);mask[lo[1]:hi[1],lo[0]:hi[0]]=True
                masks.append(mask)
            truth=reference.raster(bytime[ts]);best=max([float((m&truth).sum()/max(1,(m|truth).sum())) for m in masks] or [0.])
            key=f"{ep['video_id']}_{ts}";arrays[key]=np.asarray(masks,bool).reshape(-1,160,160)
            rows.append(dict(video_id=ep['video_id'],timestamp_us=ts,array_key=key,best_mask_iou=best,detector_seconds=seconds))
    np.savez_compressed(out/'visor_detector_probes.npz',**arrays)
    atomic_json(out/'visor_detector_probes.json',dict(rows=rows,source_sha256=digest(__file__),adapter_sha256=digest(ROOT/'src/track1_fmr/keyframe_identity.py'),reference_selection_sha256=digest(selection_path),arrays_sha256=digest(out/'visor_detector_probes.npz'),scope='Detector-only probes at locked sparse evaluation timestamps. Never injected into causal tracker, never included in ingestion timing. Previously consumed references, not fresh holdouts.'))
    print(json.dumps(rows))


if __name__=='__main__':main()
