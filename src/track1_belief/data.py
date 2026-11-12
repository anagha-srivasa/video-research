"""Persistent-point experiment on existing native context RGB; no RGB copies.

Source centroids and disjoint RGB-point targets are diagnostics, not independent
physical reference labels. Initial manual masks are privileged retrospective
initialization; only the subsequent tracking recursion has a past-only contract.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
import pyarrow.parquet as pq

from pipeline.common import ROOT, digest, atomic_json
from .common import C, D, R, W, arrays, save, guard, split, LOG
from .anchors import AnchorConfig, PersistentAnchors


def episodes():
    values = pq.read_table(ROOT/'dataset/context_v3/episodes.parquet').to_pylist()
    selected = []
    for video in C['train'] + C['validation']:
        selected.extend(sorted((r for r in values if r['video_id'] == video),
                               key=lambda r: r['episode_id'])[:C['episodes_per_video']])
    for row in selected:
        if row['split'] != split(row['video_id']):
            raise ValueError('Cohort split mismatch')
    return selected


def source_rows(episode):
    """Read source geometry only for seed selection and a separate target ledger."""
    import duckdb
    video = episode['video_id']
    split(video)
    with duckdb.connect() as db:
        db.execute('SET threads=1')
        db.execute("SET memory_limit='512MB'")
        cursor = db.execute('''SELECT m.mask_uid,m.timestamp_us,m.source_kind,m.centroid_x,m.centroid_y,
            m.annotation_width,m.annotation_height,m.geometry_complete,m.segments_raw,m.dependency_group,
            m.available_at_us,m.online_observation_allowed
            FROM read_parquet(?) m JOIN read_parquet(?) s USING(mask_uid)
            WHERE s.entity_track_id=? AND m.timestamp_us>=? AND m.timestamp_us<?
            ORDER BY m.timestamp_us,m.mask_uid''',
            [str(ROOT/f'canonical/v2/masks.{video}.parquet'),
             str(ROOT/f'canonical/v2/spatial_metadata.{video}.parquet'),
             episode['entity_track_id'], episode['start_us'], episode['end_us']])
        names = [x[0] for x in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def load_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def native_frames(video, frames, episode_id):
    """Read exact source PTS through the existing isolated decoder, one RGB at a time.

    Some published context mmap payloads have been evicted. Decoding the immutable
    MP4 avoids rebuilding gigabytes of RGB while retaining identical PTS selection.
    """
    from track1_reference.frame_checks import read_exact
    W.mkdir(parents=True, exist_ok=True)
    request = W/f'{episode_id}.decode.json'
    atomic_json(request, dict(video_id=video, frames=frames))
    process = subprocess.Popen([sys.executable, '-m', 'track1_encoder_revision.source_stream', str(request)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    seen = set()
    try:
        while True:
            tick = time.monotonic()
            header = process.stdout.read(8)
            if not header:
                break
            if len(header) != 8:
                raise EOFError('Truncated native frame header')
            index, size = struct.unpack('<II', header)
            if index != len(seen) or size != 1920*1080*3 or index >= len(frames):
                raise ValueError('Native PTS pipe schema/order violation')
            rgb = np.frombuffer(read_exact(process.stdout, size), np.uint8).reshape(1080,1920,3)
            seen.add(index)
            yield index, rgb, time.monotonic()-tick
        error = process.stderr.read().decode(errors='replace')
        if process.wait() != 0:
            raise RuntimeError('Native RGB decoder failed: ' + error[-2000:])
        if len(seen) != len(frames):
            raise ValueError('Missing requested exact native frames')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _worker(job):
    episode, destination = job
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    video, uid = episode['video_id'], episode['episode_id']
    cfg = AnchorConfig()
    output = destination/f'{uid}.npz'
    marker = destination/f'{uid}.json'
    dependencies = [ROOT/'src/track1_belief/anchors.py', ROOT/'src/track1_belief/data.py',
                    ROOT/'src/track1_encoder_revision/source_stream.py',
                    ROOT/'dataset/context_v3/episodes.parquet', ROOT/'dataset/context_v3/frames.parquet',
                    ROOT/f'canonical/v2/masks.{video}.parquet',
                    ROOT/f'canonical/v2/spatial_metadata.{video}.parquet',
                    ROOT/f'dataset/context_v3/{video}.json', ROOT/f'dataset/partition_v2/{video}.json']
    receipt = {str(p.relative_to(ROOT)): digest(p) for p in dependencies}
    inputs = dict(episode=episode, config=asdict(cfg), dependencies=receipt)
    if marker.exists() and output.exists():
        result = json.loads(marker.read_text())
        if result['inputs'] != inputs or result['sha256'] != digest(output):
            raise ValueError('Anchor checkpoint inputs/output changed: ' + uid)
        return result
    from pipeline.geometry import rasterize
    import cv2
    cv2.setNumThreads(1)
    metadata = [r for r in pq.read_table(ROOT/'dataset/context_v3/frames.parquet').to_pylist()
                if r['video_id'] == video and episode['start_us'] <= r['timestamp_us'] < episode['end_us']]
    metadata.sort(key=lambda r: r['timestamp_us'])
    times = np.array([r['timestamp_us'] for r in metadata], np.int64)
    if len(np.unique(times)) != len(times):
        raise ValueError('Duplicate native frame timestamps')
    source = source_rows(episode)
    by_time = {}
    for row in source:
        by_time.setdefault(row['timestamp_us'], row)
    seeds = [r for r in source if r['source_kind'] == 'manual_sparse' and r['geometry_complete']
             and r['timestamp_us'] in set(times)]
    seed = seeds[0] if seeds else None
    n = len(times)
    values = dict(timestamp_us=times, times=(times-times[0])/1e6 if n else np.empty(0),
                  centroid_positions=np.zeros((n, 2)), centroid_valid=np.zeros(n, bool),
                  centroid_groups=np.full(n, '', dtype='U64'), source_kind=np.full(n, '', dtype='U32'),
                  anchor_positions=np.zeros((n, 2)), valid=np.zeros(n, bool),
                  covariance_local=np.zeros((n, 2, 2)), covariance_shared=np.zeros((n, 2, 2)),
                  proxy_positions=np.zeros((n, 2)), proxy_valid=np.zeros(n, bool),
                  camera_relative_positions=np.zeros((n, 2)), camera_relative_covariance=np.zeros((n, 2, 2)),
                  camera_valid=np.zeros(n, bool), coherence=np.zeros(n), deformation_rmse=np.zeros(n),
                  latest_evidence_us=np.full(n, -1, np.int64), groups=np.full(n, uid, dtype='U64'),
                  reason=np.full(n, 'before_initial_manual_seed', dtype='U80'),
                  surviving_fit=np.zeros(n, np.int32), surviving_proxy=np.zeros(n, np.int32),
                  surviving_background=np.zeros(n, np.int32), frame_seconds=np.zeros(n))
    for i, t in enumerate(times):
        row = by_time.get(int(t))
        if row is None:
            continue
        point = np.array([row['centroid_x'], row['centroid_y']], float)
        if np.isfinite(point).all() and row['geometry_complete']:
            values['centroid_positions'][i] = point * [1920/row['annotation_width'], 1080/row['annotation_height']]
            values['centroid_valid'][i] = True
        values['centroid_groups'][i] = row['dependency_group'] or row['mask_uid']
        values['source_kind'][i] = row['source_kind']
    started = time.monotonic()
    point_positions, point_valid = [], []
    tracker = None
    chosen = [i for i,t in enumerate(times) if seed is not None and t >= seed['timestamp_us']]
    stream = native_frames(video, [metadata[i] for i in chosen], uid) if chosen else iter(())
    try:
        for local_index, rgb, decode_seconds in stream:
            i = chosen[local_index]
            t = times[i]
            tick = time.monotonic()
            if rgb.shape != (1080, 1920, 3):
                raise ValueError('Native RGB shape contract')
            if tracker is None:
                mask = rasterize(json.loads(seed['segments_raw']), seed['annotation_width'], seed['annotation_height'])
                mask = cv2.resize(mask.astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST)
                tracker = PersistentAnchors(rgb, mask, int(t), cfg)
                point_positions = np.zeros((n, len(tracker.points), 2), np.float32)
                point_valid = np.zeros((n, len(tracker.points)), bool)
                result = tracker.last
            else:
                result = tracker.update(rgb, int(t))
            values['frame_seconds'][i] = time.monotonic() - tick + decode_seconds
            mapping = {'anchor_positions':'position', 'covariance_local':'covariance_local',
                       'covariance_shared':'covariance_shared', 'valid':'valid',
                       'proxy_positions':'proxy_position', 'proxy_valid':'proxy_valid',
                       'camera_relative_positions':'camera_relative_position',
                       'camera_relative_covariance':'camera_relative_covariance', 'camera_valid':'camera_valid',
                       'coherence':'coherence', 'deformation_rmse':'deformation_rmse',
                       'latest_evidence_us':'latest_evidence_us', 'reason':'reason',
                       'surviving_fit':'surviving_fit', 'surviving_proxy':'surviving_proxy',
                       'surviving_background':'surviving_background'}
            for target, name in mapping.items():
                values[target][i] = result[name]
            point_positions[i], point_valid[i] = result['point_positions'], result['point_valid']
            if i % 16 == 0:
                guard()
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if seed is None:
        values['reason'][:] = 'no_complete_manual_seed_in_context'
    values['point_positions'] = np.asarray(point_positions) if tracker is not None else np.empty((n, 0, 2))
    values['point_valid'] = np.asarray(point_valid) if tracker is not None else np.empty((n, 0), bool)
    values['point_roles'] = tracker.roles if tracker is not None else np.empty(0, np.int8)
    values['point_initial'] = tracker.origin if tracker is not None else np.empty((0, 2))
    values['seed_reference'] = tracker.reference if tracker is not None else np.zeros(2)
    if not all(np.isfinite(v).all() for v in values.values() if v.dtype.kind in 'if'):
        raise ValueError('Non-finite anchor output')
    if np.any(values['latest_evidence_us'][values['valid']] > times[values['valid']]):
        raise ValueError('Noncausal RGB evidence')
    arrays(output, **values)
    result = dict(path=str(output.relative_to(ROOT)), video_id=video, split=split(video),
                  episode_id=uid, entity_track_id=episode['entity_track_id'], entity_name=episode['entity_name'],
                  frames=n, supported=int(values['valid'].sum()), proxy_supported=int(values['proxy_valid'].sum()),
                  matched_supported=int(np.sum(values['valid'] & values['proxy_valid'] & values['centroid_valid'])),
                  camera_supported=int(values['camera_valid'].sum()),
                  manual_seed_uid=seed['mask_uid'] if seed else None,
                  manual_seed_us=seed['timestamp_us'] if seed else None,
                  conditional_on_privileged_manual_initialization=True, fully_online_object_discovery=False,
                  subsequent_rgb_past_only=True, independent_reference=False,
                  source_centroid_retrospective=True, covariance_calibrated=False,
                  reference_kind='disjoint_point_similarity_same_RGB_LK_shared_seed_and_algorithm',
                  backend='native_OpenCV_LK_forward_backward_RANSAC_similarity',
                  physical_identity_verified=False, reacquisition_supported=False,
                  seconds=time.monotonic()-started, sha256=digest(output), inputs=inputs)
    atomic_json(marker, result)
    return result


def build_dataset(output_dir=None, reports_dir=None, config=None, monitor=None):
    """All selected contexts remain in the ledger, including unsupported ones."""
    if config is not None and config != C:
        raise ValueError('Use the locked project config; ad hoc data selection is forbidden')
    destination = Path(output_dir or D/'anchors')
    reports = Path(reports_dir or R)
    for p in (destination, reports):
        if not p.resolve().is_relative_to(ROOT):
            raise ValueError('Output outside project root')
        p.mkdir(parents=True, exist_ok=True)
    jobs = [(e, str(destination)) for e in episodes()]
    results = []
    executor = ProcessPoolExecutor(max_workers=C['rgb_workers'], mp_context=multiprocessing.get_context('spawn'))
    try:
        pending = {executor.submit(_worker, job): job[0]['episode_id'] for job in jobs}
        while pending:
            completed, _ = wait(pending, timeout=3, return_when=FIRST_COMPLETED)
            if monitor is not None:
                monitor.check()
            else:
                guard()
            for future in completed:
                uid = pending.pop(future)
                result = future.result()
                results.append(result)
                LOG.info('Native anchors %s: %d/%d supported; %.1fs', uid, result['supported'], result['frames'], result['seconds'])
    except BaseException:
        # Stop workers and their decoder children before propagating resource errors.
        import psutil
        for process in executor._processes.values():
            try:
                children = psutil.Process(process.pid).children(recursive=True)
                for child in children:
                    child.terminate()
            except psutil.Error:
                pass
            process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown()
    results.sort(key=lambda r:(r['video_id'], r['episode_id']))
    save(reports/'anchor_episodes.parquet', [{k:v for k,v in r.items() if k != 'inputs'} for r in results])
    atomic_json(reports/'anchor_data.json', dict(episodes=results, proposed=len(jobs),
        independent_reference_count=0, new_rgb_bytes=0, native_resolution=[1920,1080],
        conditional_manual_initialization=True, validation_used_for_selection=False,
        neural_tracker_used=False, neural_tracker_status='No preinstalled point-tracker checkpoint; this is an explicit LK baseline'))
    return results
