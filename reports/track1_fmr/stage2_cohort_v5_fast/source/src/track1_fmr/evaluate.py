"""Small matched-source engineering pilot; not an independent identity benchmark."""
import argparse
import json
import time
import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_belief.common import save
from track1_belief import data
from .dorsal_flux import flux, measure
from .belief_filter import BeliefTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--mode', choices=('seeded', 'blind'), default='seeded')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--coasting-prior', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.frames <= 360:
        raise ValueError('Frame count must be between 2 and 360')
    work = ROOT / 'work/track1_fmr'; report = ROOT / 'reports/track1_fmr'
    work.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    episodes = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes']
    if not 0 <= args.episode < len(episodes):
        raise ValueError('Episode index out of bounds')
    meta = episodes[args.episode]
    path = ROOT / meta['path']
    if digest(path) != meta['sha256']:
        raise ValueError('Preserved point ledger changed')
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    times = old['timestamp_us']
    frames = [x for x in pq.read_table(ROOT / 'dataset/context_v3/frames.parquet').to_pylist()
              if x['video_id'] == meta['video_id'] and x['timestamp_us'] in set(times)]
    frames.sort(key=lambda x: x['timestamp_us'])
    if [f['timestamp_us'] for f in frames] != times.tolist():
        raise ValueError('Source frame timestamps differ from baseline')
    indices = np.flatnonzero(times >= meta['manual_seed_us'])[:args.frames]
    frames = [frames[i] for i in indices]
    data.W = work
    cv2.setNumThreads(4)
    tracker, previous, records = BeliefTracker(), None, []
    if args.mode == 'blind':
        from .pipeline import Pipeline
        pipeline = Pipeline(carry_prior=args.coasting_prior)
    peak = 0
    for n, rgb, _ in data.native_frames(meta['video_id'], frames, meta['episode_id'] + '.fmr'):
        rss = psutil.Process().memory_info().rss
        available = psutil.virtual_memory().available
        peak = max(peak, rss)
        if rss > 18 * 2**30 or available < 6 * 2**30:
            raise MemoryError('Resource guard stopped FMR pilot')
        index, timestamp = indices[n], times[indices[n]] / 1e6
        start = time.perf_counter()
        if args.mode == 'blind':
            result = pipeline.step(rgb, timestamp)
            records.append(dict(frame=int(index), timestamp_us=int(times[index]),
                                baseline_observed=bool(old['valid'][index]),
                                proposed_observed=result['observed'], proposed_coasting=result['coasting'],
                                candidate_count=result['tracks'], rejected_capacity=result['rejected_capacity'],
                                elapsed_seconds=time.perf_counter() - start))
            if n % 10 == 0:
                print(f'{n + 1}/{len(indices)} frames', flush=True)
            continue
        if previous is None:
            points = old['point_initial'][old['point_roles'] == 0].astype(np.int32)
            if len(points) < 3:
                raise ValueError('Insufficient seed points')
            mask = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
            cloud = measure(rgb, mask)
            current = tracker.update(timestamp, [cloud])
            supported, reason = True, 'manual_seed_point_hull'
        else:
            result = flux(previous, rgb)
            clouds = [c['measure'] for c in result['clouds']]
            current = tracker.update(timestamp, clouds)
            supported, reason = result['supported'], result['reason']
        state = current['tracks'].get(0)
        records.append(dict(frame=int(index), timestamp_us=int(times[index]), flux_supported=bool(supported),
                            flux_failure=reason if not supported else None,
                            baseline_observed=bool(old['valid'][index]),
                            track_observed=bool(state is not None and state.status == 'OBSERVED'),
                            track_predicted=bool(state is not None and state.status == 'PREDICTED'),
                            candidate_count=len(current['tracks']), rejected_capacity=current['rejected_capacity'],
                            elapsed_seconds=time.perf_counter() - start))
        previous = rgb.copy()
        if n % 10 == 0:
            print(f'{n + 1}/{len(indices)} frames', flush=True)
    suffix = (('native_coasting_' if args.coasting_prior else 'native_blind_') + meta['episode_id']) if args.mode == 'blind' else 'native_pilot'
    if args.mode == 'blind':
        tubes = pipeline.close()
        from .ledger import write_tubes
        prefix = 'tubes_coasting' if args.coasting_prior else 'tubes'
        ledger = ROOT / 'derived/track1_fmr' / f"{prefix}.{meta['episode_id']}.parquet"
        write_tubes(ledger, tubes, video_id=meta['video_id'], episode_id=meta['episode_id'])
        save(report / (suffix + '.parquet'), records)
        atomic_json(report / (suffix + '.json'), dict(video=meta['video_id'], episode_id=meta['episode_id'],
                    frames=len(records), emitted_tubes=len(tubes), proposals_per_frame=float(np.mean([r['proposed_observed'] for r in records])),
                    tube_ledger=str(ledger.relative_to(ROOT)), tube_ledger_sha256=digest(ledger),
                    capacity_rejections=sum(r['rejected_capacity'] for r in records),
                    identity_verified=False, source_target_coverage_measured=False,
                    source_timestamps='same native PTS as baseline', peak_process_rss_gib=peak / 2**30))
        return
    save(report / 'native_pilot.parquet', records)
    atomic_json(report / 'native_pilot.json', dict(video=meta['video_id'], episode_id=meta['episode_id'],
                frames=len(records), baseline_observed=sum(r['baseline_observed'] for r in records),
                candidate_observed=sum(r['track_observed'] for r in records),
                candidate_coasting=sum(r['track_predicted'] for r in records),
                identity_verified=False, scientific_coverage_validated=False,
                seed='convex hull of privileged original foreground points',
                scope='Single source episode, no independent identity truth; track 0 may drift or swap',
                peak_process_rss_gib=peak / 2**30))


if __name__ == '__main__':
    main()
