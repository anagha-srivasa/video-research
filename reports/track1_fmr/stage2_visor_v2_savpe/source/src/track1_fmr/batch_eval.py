"""Checkpointed blind proposal audit across the frozen native-PTS cohort."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
from pipeline.common import ROOT, atomic_json, digest

REPORT = ROOT / 'reports/track1_fmr'
WORK = ROOT / 'work/track1_fmr'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--coasting-prior', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True, parents=True); WORK.mkdir(exist_ok=True, parents=True)
    metadata = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes']
    protocol = dict(scope='blind proposal engineering audit; no independently verified source identity',
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src/track1_fmr').glob('*.py'))},
                    source_ledger_sha256=digest(ROOT / 'reports/track1_belief/anchor_data.json'),
                    baseline_sources={m['path']: m['sha256'] for m in metadata},
                    workers=args.workers, frame_limit=360, max_tree_rss_gib=18, min_available_gib=6,
                    coasting_prior=args.coasting_prior)
    prefix = 'coasting_batch' if args.coasting_prior else 'batch'
    protocol_path = REPORT / f'{prefix}_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Batch source or protocol changed; preserve this run before starting another')
    atomic_json(protocol_path, protocol)
    protocol_hash = digest(protocol_path)
    prior_dir = WORK / 'audit_v2'
    prior_protocol_path = prior_dir / 'batch_protocol.json'
    reusable = False
    if prior_protocol_path.exists() and not args.coasting_prior:
        prior_protocol = json.loads(prior_protocol_path.read_text())
        unchanged = {k: v for k, v in prior_protocol['source_hashes'].items()
                     if not k.endswith(('/evaluate.py', '/batch_eval.py'))}
        old_evaluate = (prior_dir / 'source/evaluate.py').read_text()
        expected = old_evaluate.replace('2 <= args.frames <= 300', '2 <= args.frames <= 360').replace(
            'Frame count must be between 2 and 300', 'Frame count must be between 2 and 360')
        reusable = (all((ROOT / k).exists() and digest(ROOT / k) == v for k, v in unchanged.items())
                    and (ROOT / 'src/track1_fmr/evaluate.py').read_text() == expected
                    and prior_protocol['source_ledger_sha256'] == protocol['source_ledger_sha256'])
        if not reusable:
            raise ValueError('Prior run is not safely reusable after the frame-limit amendment')
    atomic_json(REPORT / f'{prefix}_status.json', dict(state='running', completed=0, total=len(metadata),
                protocol_sha256=protocol_hash))
    guard_stop = threading.Event()
    children, lock = {}, threading.Lock()
    resource = dict(max_tree_rss_gib=0., minimum_available_gib=psutil.virtual_memory().available / 2**30, failure=None)

    def monitor():
        while not guard_stop.wait(2):
            try:
                root = psutil.Process()
                processes = [root] + root.children(recursive=True)
                rss = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                rss /= 2**30
                available = psutil.virtual_memory().available / 2**30
                resource['max_tree_rss_gib'] = max(resource['max_tree_rss_gib'], rss)
                resource['minimum_available_gib'] = min(resource['minimum_available_gib'], available)
                if rss > 18 or available < 6:
                    resource['failure'] = f'resource limit: tree RSS {rss:.2f} GiB, available {available:.2f} GiB'
                    with lock:
                        for child in children.values(): child.terminate()
                    return
            except (psutil.Error, OSError) as exc:
                resource['failure'] = f'cannot verify process-tree memory: {exc}'
                with lock:
                    for child in children.values(): child.terminate()
                return

    def run_episode(index):
        meta = metadata[index]; uid = meta['episode_id']
        marker = REPORT / f'{prefix}_{uid}.json'
        output_prefix = 'native_coasting' if args.coasting_prior else 'native_blind'
        output = REPORT / f'{output_prefix}_{uid}.json'
        parquet = REPORT / f'{output_prefix}_{uid}.parquet'
        if marker.exists():
            previous = json.loads(marker.read_text())
            ledger = ROOT / previous['tube_ledger']
            if previous['protocol_sha256'] != protocol_hash or not output.exists() or not parquet.exists() or not ledger.exists() or previous['output_sha256'] != digest(output) or previous['parquet_sha256'] != digest(parquet) or previous['tube_ledger_sha256'] != digest(ledger):
                raise ValueError('Existing episode checkpoint changed')
            return previous
        prior_marker = prior_dir / marker.name
        if reusable and prior_marker.exists():
            previous = json.loads(prior_marker.read_text())
            with np.load(ROOT / meta['path'], allow_pickle=False) as data:
                available = int(np.count_nonzero(data['timestamp_us'] >= meta['manual_seed_us']))
            ledger = ROOT / previous['tube_ledger']
            if (available <= 300 and previous['frames'] == available
                    and previous['protocol_sha256'] == digest(prior_protocol_path)
                    and output.exists() and parquet.exists() and ledger.exists()
                    and digest(output) == previous['output_sha256']
                    and digest(parquet) == previous['parquet_sha256']
                    and digest(ledger) == previous['tube_ledger_sha256']):
                previous.update(protocol_sha256=protocol_hash, reused_from_frame_limit=300,
                                reuse_reason='entire post-seed episode contains at most 300 frames')
                atomic_json(marker, previous)
                return previous
        if resource['failure']:
            raise MemoryError(resource['failure'])
        command = [sys.executable, '-u', '-m', 'track1_fmr.evaluate', '--frames', '360', '--mode', 'blind', '--episode', str(index)]
        if args.coasting_prior:
            command.append('--coasting-prior')
        log_name = f'{uid}.coasting.log' if args.coasting_prior else f'{uid}.log'
        with (WORK / log_name).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            with lock: children[uid] = child
            code = child.wait()
            with lock: children.pop(uid, None)
        if code or resource['failure']:
            raise RuntimeError(f'{uid} failed: return={code}, guard={resource["failure"]}')
        row = dict(split=meta['split'],
                   protocol_sha256=protocol_hash, output_sha256=digest(output), parquet_sha256=digest(parquet),
                   **json.loads(output.read_text()))
        atomic_json(marker, row)
        return row

    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_episode, i) for i in range(len(metadata))]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    resource['failure'] = resource['failure'] or 'episode failure'
                    with lock:
                        for child in children.values(): child.terminate()
                    for other in futures: other.cancel()
                    raise
                atomic_json(REPORT / f'{prefix}_status.json', dict(state='running', completed=len(results), total=len(metadata), resource=resource))
                print(f'{len(results)}/{len(metadata)} episodes complete', flush=True)
        atomic_json(REPORT / f'{prefix}_status.json', dict(state='completed', completed=len(results), total=len(metadata),
                    resource=resource, frames=sum(r['frames'] for r in results),
                    rejected_capacity=sum(r['capacity_rejections'] for r in results),
                    emitted_tubes=sum(r['emitted_tubes'] for r in results),
                    scientific_coverage_validated=False, identity_verified=False))
    except Exception as exc:
        with lock:
            for child in children.values(): child.terminate()
        atomic_json(REPORT / f'{prefix}_status.json', dict(state='failed', completed=len(results), total=len(metadata),
                    error=str(exc), resource=resource))
        raise
    finally:
        guard_stop.set(); watcher.join()


if __name__ == '__main__':
    main()
