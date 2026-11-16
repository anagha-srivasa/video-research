"""Observed-only engineering cohort diagnostic; independent NEES/IDF1 require adjudication."""
import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_belief import data
from track1_fmr.pipeline import Pipeline
from track1_fmr.ledger import write_tubes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='stage2_cohort_v1')
    parser.add_argument('--acceleration-noise',type=float,default=.001)
    parser.add_argument('--measurement-floor',type=float,default=0.)
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    report=ROOT/'reports/track1_fmr'/args.tag
    report.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    metadata_path=ROOT/'reports/track1_belief/anchor_data.json'
    metadata=json.loads(metadata_path.read_text())['episodes']
    frame_index=ROOT/'dataset/context_v3/frames.parquet'
    frames_all=pq.read_table(frame_index).to_pylist()
    ref_paths=[ROOT/'reports/track1_reference/independent_evaluation.json',
               ROOT/'reports/track1_lab/independent_reference_validation.json',
               ROOT/'reports/track1_remediation/independent_evaluation.json']
    protocol=dict(scope='20-episode engineering diagnostic; source-centroid proxy is not adjudicated identity truth',
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),*sorted((ROOT/'src/track1_fmr').glob('*.py'))]},
                  episode_ids=[m['episode_id'] for m in metadata],source_ledgers={m['path']:m['sha256'] for m in metadata},
                  metadata_sha256=digest(metadata_path),frame_index_sha256=digest(frame_index),
                  reference_readiness_hashes={str(p.relative_to(ROOT)):digest(p) for p in ref_paths},
                  acceleration_noise=args.acceleration_noise,measurement_floor=args.measurement_floor,
                  covariance_calibrated=False,frame_policy='Decode >=seed; observed coverage denominators use strictly >seed',
                  proxy_initialization='At first valid source annotation, one closest OBSERVED ID within 50 native pixels; never reassign',
                  max_process_tree_rss_gib=18,memory_sampling_interval_ms=50,
                  independent_identity_idf1=None,independent_mean_2d_nees=None)
    atomic_json(report/'protocol.json',protocol)
    for relative in protocol['source_hashes']:
        dest=report/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    root=psutil.Process();monitored=[root];lock=threading.Lock();stop=threading.Event();resource=dict(peak_rss=0,failure=None)
    def monitor():
        while not stop.is_set():
            try:
                with lock:processes=list(monitored)
                rss=0
                for process in processes:
                    try:rss+=process.memory_info().rss
                    except psutil.NoSuchProcess:pass
                resource['peak_rss']=max(resource['peak_rss'],rss)
                if rss>18*2**30:resource['failure']='18 GiB guard exceeded'
            except Exception as exc:resource['failure']=repr(exc)
            stop.wait(.05)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    results=[];started=time.perf_counter();data.W=ROOT/'work/track1_fmr'
    try:
        for meta in metadata:
            source=ROOT/meta['path']
            if digest(source)!=meta['sha256']:raise ValueError('Frozen point ledger changed')
            with np.load(source,allow_pickle=False) as z:
                timestamps=z['timestamp_us'];indices=np.flatnonzero(timestamps>=meta['manual_seed_us'])
                centres=z['centroid_positions'].copy();valid=z['centroid_valid'].copy()
            requested=set(timestamps[indices].tolist())
            frames=sorted([f for f in frames_all if f['video_id']==meta['video_id'] and f['timestamp_us'] in requested],key=lambda f:f['timestamp_us'])
            if [f['timestamp_us'] for f in frames]!=timestamps[indices].tolist():raise ValueError('Exact source PTS mismatch')
            pipeline=Pipeline(dorsal_grid=160,acceleration_noise=args.acceleration_noise,measurement_floor=args.measurement_floor)
            rows=[];attempted=False;fixed_id=None;success=0;eligible=0;observed_errors=[];observations=[]
            for n,rgb,_ in data.native_frames(meta['video_id'],frames,meta['episode_id']+'.stage2'):
                processes=[root]+root.children(recursive=True)
                with lock:monitored[:]=processes
                if resource['failure']:raise MemoryError(resource['failure'])
                index=int(indices[n]);timestamp=int(timestamps[index]);t=timestamp/1e6
                tick=time.perf_counter();value=pipeline.step(rgb,t);ingestion_ms=(time.perf_counter()-tick)*1000
                if not attempted and valid[index]:
                    attempted=True
                    candidates=[(float(np.linalg.norm(state.mean[:2]*[1920,1080]-centres[index])),i)
                                for i,state in pipeline.tracker.tracks.items() if state.status=='OBSERVED']
                    if candidates:
                        distance,candidate=min(candidates)
                        if distance<=50:fixed_id=candidate
                state=pipeline.tracker.tracks.get(fixed_id)
                observed=bool(state is not None and state.status=='OBSERVED')
                distance=None if state is None or not valid[index] else float(np.linalg.norm(state.mean[:2]*[1920,1080]-centres[index]))
                post=timestamp>meta['manual_seed_us']
                if post and valid[index]:
                    eligible+=1
                    success+=int(observed and distance is not None and distance<=50)
                    if observed:observed_errors.append(distance)
                rows.append(dict(timestamp_us=timestamp,post_seed=post,source_centroid_valid=bool(valid[index]),
                                 fixed_id=fixed_id,fixed_id_observed=observed,source_centroid_error_px=distance,
                                 ingestion_ms=ingestion_ms,**value))
                for track_id,cloud in pipeline.last_observations.items():
                    belief=pipeline.tracker.tracks[track_id]
                    observations.append([t,float(track_id),*cloud.centroid,*cloud.obs_cov_c.ravel(),*belief.mean,*belief.covariance.ravel()])
                if (n+1)%100==0:print(f"{meta['episode_id']} {n+1}/{len(frames)} frames elapsed={time.perf_counter()-started:.1f}s",flush=True)
            tubes=pipeline.close()
            ledger=report/(meta['episode_id']+'.tubes.parquet')
            write_tubes(ledger,tubes,video_id=meta['video_id'],episode_id=meta['episode_id'])
            obs_path=report/(meta['episode_id']+'.observations.npz')
            np.savez_compressed(obs_path,columns=np.array(['time','track_id','obs_x','obs_y','obs_cov_xx','obs_cov_xy','obs_cov_yx','obs_cov_yy',
                'belief_x','belief_y','belief_vx','belief_vy',*[f'belief_cov_{i}{j}' for i in range(4) for j in range(4)]]),
                values=np.asarray(observations,dtype=np.float64).reshape(-1,28))
            rows_path=report/(meta['episode_id']+'.frames.json');atomic_json(rows_path,rows)
            results.append(dict(episode_id=meta['episode_id'],split=meta['inputs']['episode']['split'],decoded_frames=len(rows),
                post_seed_frames=sum(row['post_seed'] for row in rows),source_reference_eligible=eligible,
                initialized_fixed_id=fixed_id,observed_only_proxy_successes=success,
                observed_only_proxy_coverage=success/eligible if eligible else None,
                conditional_observed_rmse_px=float(np.sqrt(np.mean(np.square(observed_errors)))) if observed_errors else None,
                emitted_tubes=len(tubes),observed_segments=sum(t.epistemic_tag.value=='OBSERVED' for t in tubes),
                predicted_segments=sum(t.epistemic_tag.value=='PREDICTED' for t in tubes),
                observations_sha256=digest(obs_path),frames_sha256=digest(rows_path),ledger_sha256=digest(ledger)))
            atomic_json(report/'progress.json',dict(completed=len(results),total=len(metadata),elapsed_seconds=time.perf_counter()-started,episodes=results))
            print(f"{len(results)}/{len(metadata)} episodes; fixed_id={fixed_id}; proxy observed-only={success}/{eligible}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        if resource['failure']:raise MemoryError(resource['failure'])
        total=sum(e['source_reference_eligible'] for e in results);success=sum(e['observed_only_proxy_successes'] for e in results)
        atomic_json(report/'result.json',dict(protocol_sha256=digest(report/'protocol.json'),episodes=results,
             decoded_frames=sum(e['decoded_frames'] for e in results),post_seed_frames=sum(e['post_seed_frames'] for e in results),
             fixed_ids_initialized=sum(e['initialized_fixed_id'] is not None for e in results),
             source_reference_eligible=total,observed_only_proxy_successes=success,observed_only_proxy_coverage=success/total if total else None,
             sampled_process_tree_peak_rss_gib=resource['peak_rss']/2**30,elapsed_seconds=time.perf_counter()-started,
             independent_identity_idf1=None,mean_2d_nees=None,joint_95_coverage=None,covariance_calibrated=False,
             calibration_status='Blocked: adjudicated position/identity reference ledgers have no usable labels; source centroids remain a proxy, not NEES ground truth',
             stage2_promoted=False))
    except Exception as exc:
        atomic_json(report/'failure.json',dict(error=repr(exc),resource=resource,episodes=results));raise
    finally:
        stop.set();thread.join()


if __name__=='__main__':main()
