"""Stage 1 paired validation. Predictions are never scored as observed coverage."""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json
from track1_belief import data
from track1_fmr.pipeline import Pipeline
from track1_fmr.dorsal_flux import PromptFreeHeartbeat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tag', default='stage1_v1')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Tag must be alphanumeric with underscores')
    cv2.setNumThreads(4)
    import torch
    torch.set_num_threads(4)
    episodes = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes']
    selected = [m for m in episodes if m['inputs']['episode']['split'] == args.split]
    # Training smoke uses one episode; validation evaluates all five existing validation episodes.
    if args.split == 'train':
        selected = selected[:1]
    report = ROOT / 'reports/track1_fmr' / args.tag
    report.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'work/yoloe-26s-seg-pf.pt'
    protocol = dict(split=args.split, episode_ids=[m['episode_id'] for m in selected], frames_per_episode=30,
                    device=args.device, min_area_frac=.0005, max_area_frac=.4, radius_frac=.012,
                    max_proposals=15, heartbeat_seconds=2., rss_cap_gib=18,
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                   [Path(__file__).resolve(), *sorted((ROOT / 'src/track1_fmr').glob('*.py'))]},
                    checkpoint_sha256=sha(checkpoint),
                    source_ledger_sha256=sha(ROOT / 'reports/track1_belief/anchor_data.json'),
                    split_limitation='Existing validation participants were used in earlier exploratory audits; not a fresh unseen holdout',
                    gate='Every episode: proposal ratio [0.85,1.15], nonzero tracks; grid mean ingestion <=15ms; sampled tree RSS <=18GiB')
    atomic_json(report / 'protocol.json', protocol)
    stop = threading.Event()
    resources = dict(peak_rss=0, failure=None)

    def monitor():
        while not stop.is_set():
            try:
                root = psutil.Process()
                rss = 0
                for process in [root] + root.children(recursive=True):
                    try:
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                resources['peak_rss'] = max(resources['peak_rss'], rss)
                if rss > 18 * 2**30:
                    resources['failure'] = '18 GiB process-tree cap exceeded'
            except Exception as exc:
                resources['failure'] = repr(exc)
            stop.wait(.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    frames_table = pq.read_table(ROOT / 'dataset/context_v3/frames.parquet').to_pylist()
    heartbeat = PromptFreeHeartbeat(device=args.device)
    data.W = ROOT / 'work/track1_fmr'
    results = []
    try:
        # Cold loading/warmup is separately recorded, never silently included in warm throughput.
        start = time.perf_counter()
        heartbeat(np.zeros((640, 640, 3), np.uint8))
        warmup_seconds = time.perf_counter() - start
        for meta in selected:
            if resources['failure']:
                raise MemoryError(resources['failure'])
            path = ROOT / meta['path']
            if sha(path) != meta['sha256']:
                raise ValueError('Frozen source ledger changed')
            with np.load(path, allow_pickle=False) as z:
                times = z['timestamp_us']
            times = times[times >= meta['manual_seed_us']][:30].tolist()
            frames = sorted((f for f in frames_table if f['video_id'] == meta['video_id']
                             and f['timestamp_us'] in times), key=lambda f: f['timestamp_us'])
            if [f['timestamp_us'] for f in frames] != times:
                raise ValueError('Native PTS mismatch')
            pipelines = {'native': Pipeline(heartbeat=heartbeat),
                         'grid160': Pipeline(dorsal_grid=160, heartbeat=heartbeat)}
            rows, rgb_hash, pixels = [], hashlib.sha256(), 0
            for n, rgb, _ in data.native_frames(meta['video_id'], frames, meta['episode_id'] + '.invariance'):
                if resources['failure']:
                    raise MemoryError(resources['failure'])
                rgb_hash.update(rgb.tobytes())
                pixels += rgb.shape[0] * rgb.shape[1]
                row = {'timestamp_us': times[n]}
                for name in (list(pipelines) if n % 2 == 0 else list(pipelines)[::-1]):
                    start = time.perf_counter()
                    value = pipelines[name].step(rgb, times[n] / 1e6)
                    row[name] = dict(value, ingestion_ms=(time.perf_counter() - start) * 1000)
                rows.append(row)
            summary = {}
            for name in pipelines:
                latency = [r[name]['ingestion_ms'] for r in rows]
                summary[name] = dict(proposals=sum(r[name]['proposals'] for r in rows),
                                     flux_proposals=sum(r[name]['flux_proposals'] for r in rows),
                                     heartbeat_proposals=sum(r[name]['heartbeat_proposals'] for r in rows),
                                     mean_ingestion_ms=float(np.mean(latency)), median_ingestion_ms=float(np.median(latency)),
                                     p95_ingestion_ms=float(np.percentile(latency, 95)),
                                     final_tracks=rows[-1][name]['tracks'], tentative=None, confirmed=None)
            native, grid = summary['native'], summary['grid160']
            ratio = grid['proposals'] / native['proposals'] if native['proposals'] else None
            passed = (ratio is not None and .85 <= ratio <= 1.15 and
                      native['final_tracks'] > 0 and grid['final_tracks'] > 0 and grid['mean_ingestion_ms'] <= 15)
            results.append(dict(episode_id=meta['episode_id'], video_id=meta['video_id'],
                                rgb_sha256=rgb_hash.hexdigest(), decoded_pixels=pixels, summary=summary,
                                proposal_ratio=ratio, local_gate_passed=passed, records=rows))
            atomic_json(report / 'progress.json', dict(completed=len(results), total=len(selected), episodes=results))
            print(f"{len(results)}/{len(selected)} {meta['video_id']}: ratio={ratio}, grid mean={grid['mean_ingestion_ms']:.2f}ms, pass={passed}", flush=True)
        if resources['failure']:
            raise MemoryError(resources['failure'])
        atomic_json(report / 'result.json', dict(protocol_sha256=sha(report / 'protocol.json'),
                    warmup_seconds=warmup_seconds, heartbeat_calls=heartbeat.calls,
                    sampled_process_tree_peak_rss_gib=resources['peak_rss'] / 2**30,
                    memory_sampling_interval_ms=50, episodes=results,
                    local_gate_passed=all(r['local_gate_passed'] for r in results),
                    promotion_qualified=False, lifecycle_implemented=False,
                    qualification_limitation='Reused validation cohort and no independent identity truth; Stage 2 not started'))
    except Exception as exc:
        atomic_json(report / 'failure.json', dict(error=repr(exc), episodes=results, resources=resources))
        raise
    finally:
        stop.set()
        thread.join()


if __name__ == '__main__':
    main()
