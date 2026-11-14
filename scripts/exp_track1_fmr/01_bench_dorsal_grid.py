"""Matched native/grid ingestion diagnostic; no identity or promotion claim."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json
from track1_belief import data
from track1_fmr.pipeline import Pipeline


def main():
    cv2.setNumThreads(4)
    meta = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes'][0]
    ledger = ROOT / meta['path']
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == meta['sha256']
    with np.load(ledger, allow_pickle=False) as z:
        times = z['timestamp_us']
    selected = times[times >= meta['manual_seed_us']][:30].tolist()
    frames = sorted((f for f in pq.read_table(ROOT / 'dataset/context_v3/frames.parquet').to_pylist()
                     if f['video_id'] == meta['video_id'] and f['timestamp_us'] in selected),
                    key=lambda f: f['timestamp_us'])
    assert [f['timestamp_us'] for f in frames] == selected
    data.W = ROOT / 'work/track1_fmr'
    pipelines = {'native': Pipeline(dorsal_grid=None), 'grid160': Pipeline(dorsal_grid=160)}
    records, peak, pixels = [], 0, 0
    digest = hashlib.sha256()
    for n, rgb, _ in data.native_frames(meta['video_id'], frames, meta['episode_id'] + '.grid'):
        processes = [psutil.Process()] + psutil.Process().children(recursive=True)
        rss = 0
        for process in processes:
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass  # Decoder may exit between enumeration and sampling.
        peak = max(peak, rss)
        if rss > 18 * 2**30 or psutil.virtual_memory().available < 6 * 2**30:
            raise MemoryError('Resource guard stopped diagnostic')
        digest.update(rgb.tobytes())
        pixels += rgb.shape[0] * rgb.shape[1]
        row = {'timestamp_us': selected[n]}
        # Alternate order to reduce systematic warm-cache ordering bias.
        for name in (list(pipelines) if n % 2 == 0 else list(pipelines)[::-1]):
            start = time.perf_counter()
            result = pipelines[name].step(rgb, selected[n] / 1e6)
            row[name] = dict(result, ingestion_ms=(time.perf_counter() - start) * 1000)
        records.append(row)
        print(f'{n + 1}/{len(frames)} matched frames', flush=True)
    summary = {}
    for name, pipeline in pipelines.items():
        latency = [r[name]['ingestion_ms'] for r in records]
        summary[name] = dict(median_ingestion_ms=float(np.median(latency)),
                             p95_ingestion_ms=float(np.percentile(latency, 95)),
                             capacity_rejections=sum(r[name]['rejected_capacity'] for r in records),
                             final_tracks=records[-1][name]['tracks'],
                             retained_previous_rgb_bytes=pipeline.previous.nbytes)
    atomic_json(ROOT / 'reports/track1_fmr/dorsal_grid_diagnostic.json', dict(
        scope='30-frame engineering diagnostic; no independent identity truth or promotion',
        episode=meta['episode_id'], decoded_rgb_sha256=digest.hexdigest(), decoded_pixels=pixels,
        sampled_combined_process_tree_peak_rss_gib=peak / 2**30,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), *sorted((ROOT / 'src/track1_fmr').glob('*.py'))]},
        summary=summary, records=records, identity_verified=False, promotion_qualified=False))


if __name__ == '__main__':
    main()
