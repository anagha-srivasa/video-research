"""Source-backed Track 1 ingestion release; machine evidence is never ground truth."""
import collections
import contextlib
import concurrent.futures
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .common import ROOT, VIDEOS, Monitor, split
from .store import sequences
from pipeline.common import atomic_json, digest

OUT = ROOT / 'derived/track1_training_ready'
REPORT = ROOT / 'reports/track1_training_ready'
BASES = {'v2': ROOT / 'derived/track1_dataset_v2',
         'context': ROOT / 'derived/track1_context_v3'}
LOG = logging.getLogger(__name__)


def save(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    pq.write_table(pa.Table.from_pylist(rows), temp, compression=None)
    temp.replace(path)


def masked_annotation(value, flags, shared=False):
    """Only explicit opt-in weak targets; physical/identity/event supervision forbidden."""
    clean = bool(value.get('found') and value.get('visibility') == 'visible'
                 and not value.get('uncertainty') and not shared
                 and not any('disagreement' in f or 'contradict' in f for f in flags))
    projected = value.get('projected', 'unknown')
    return dict(projected_label={'gap': 0, 'meeting': 1, 'overlap': 2}.get(projected, -1),
                projected_weak_mask=clean and projected != 'unknown',
                physical_contact_mask=False, boundary_mask=False, identity_mask=False,
                velocity_mask=False, event_mask=False, human_reference_eligible=False,
                retrospective_context=True, shared_cue_not_verified_identity=shared)


def snapshot_reviews():
    source = ROOT / 'derived/track1_qwen_review_v3/reviews.sqlite'
    path = OUT / 'reviews.sqlite'
    if path.exists():
        raise FileExistsError('Release already exists; use --verify, never replace a snapshot')
    src = sqlite3.connect('file:' + str(source) + '?mode=ro', uri=True)
    if src.execute("SELECT count(*) FROM review_plan WHERE status='pending'").fetchone()[0]:
        raise RuntimeError('Review still has pending jobs')
    dst = sqlite3.connect(path)
    src.backup(dst)
    assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert src.execute('SELECT * FROM reviews ORDER BY task_id,idx').fetchall() == dst.execute('SELECT * FROM reviews ORDER BY task_id,idx').fetchall()
    dst.close()
    src.close()


def reviews():
    from track1_reference.adaptive_review import group_key
    tasks = json.loads((ROOT / 'references/track1/tasks.json').read_text())
    private = json.loads((ROOT / 'work/track1_reference/task_map.json').read_text())
    db = sqlite3.connect('file:' + str(OUT / 'reviews.sqlite') + '?mode=ro', uri=True)
    observed = {(t, i): (json.loads(v), json.loads(f), b) for t, i, v, f, b in
                db.execute('SELECT task_id,idx,value,flags,batch_id FROM reviews')}
    plans = {(g, i): s for g, i, s in db.execute('SELECT group_id,idx,status FROM review_plan')}
    groups = collections.defaultdict(list)
    for t in tasks:
        groups[group_key(t)].append(t['task_id'])
    evidence, ledger = [], []
    for t in tasks:
        mapping = private['tasks'][t['task_id']]
        video = mapping['video_id']
        for i, frame in enumerate(private['clips'][t['clip_id']]):
            match = observed.get((t['task_id'], i))
            shared = False
            if match is None:
                for other in groups[group_key(t)]:
                    if (other, i) in observed:
                        match = observed[other, i]
                        shared = True
                        break
            plan = plans.get((group_key(t), i), 'not_selected')
            common = dict(task_id=t['task_id'], group_id=group_key(t), kind=t['kind'],
                          video_id=video, split=split(video), clip_id=t['clip_id'],
                          frame_index=i, timestamp_us=frame['timestamp_us'],
                          native_frame_uid=frame['native_frame_uid'], pts_raw=frame['pts_raw'])
            masks = masked_annotation({}, [], shared)
            if match:
                value, flags, batch = match
                shared |= bool(value.get('shared_cue_is_not_identity')) or any('shared_cue' in f for f in flags)
                masks = masked_annotation(value, flags, shared)
                evidence.append(dict(common, **masks, cue=t['cue'], batch_id=batch,
                                     annotation_json=json.dumps(value), flags_json=json.dumps(flags),
                                     source_association_verified=False,
                                     # Context source chains and class-cue reviews are not instance matches.
                                     training_track_id=None))
            ledger.append(dict(common, **masks, has_machine_observation=match is not None,
                               plan_status=plan, supervised_contact_eligible=False))
    checks = [dict(clip_id=c, frame_index=i, timestamp_us=t, native_frame_uid=u,
                   diagnostics_json=v) for c, i, t, u, v in
              db.execute('SELECT clip,idx,timestamp_us,uid,value FROM frame_checks ORDER BY clip,idx')]
    save('machine_observations.parquet', evidence)
    save('review_coverage.parquet', ledger)
    save('frame_diagnostics.parquet', checks)
    plan_counts = dict(db.execute('SELECT status,count(*) FROM review_plan GROUP BY status'))
    db.close()
    return dict(case_frame_opportunities=len(ledger), stored_review_rows=len(observed),
                available_case_frame_observations=len(evidence), native_checks=len(checks),
                selected_question_status=plan_counts,
                opt_in_projected_weak_targets=sum(r['projected_weak_mask'] for r in evidence),
                physical_contact_targets=0, independent_references=0)


def frame_index():
    frames = {}
    for name in ['partition_v2', 'context_v3']:
        for r in pq.read_table(ROOT / f'dataset/{name}/frames.parquet').to_pylist():
            key = (r['video_id'], r['timestamp_us'])
            if key in frames:
                assert frames[key]['pts_raw'] == r['pts_raw']
                continue
            frames[key] = dict(video_id=r['video_id'], split=split(r['video_id']),
                               timestamp_us=r['timestamp_us'], pts_raw=r['pts_raw'],
                               native_frame_uid=r['native_frame_uid'],
                               source_path=f"dataset/videos/{r['video_id']}.MP4",
                               native_width=1920, native_height=1080)
    rows = sorted(frames.values(), key=lambda r: (r['video_id'], r['timestamp_us']))
    save('frames.parquet', rows)
    return frames


def build_indices(frames, monitor):
    allrows, counts = [], []
    # Deduplicate identical state opportunities across old/context timelines.
    # Context wins only on presence, never by validation reconstruction score.
    seen = set()
    for source in ['context', 'v2']:
        states = pq.read_table(ROOT / f'derived/track1_remediation/{source}/tiered_states.parquet').to_pylist()
        for v in VIDEOS:
            for r in states:
                if r['video_id'] != v:
                    continue
                key = (v, r['entity_track_id'], r['window_end_us'])
                if key in seen:
                    continue
                seen.add(key)
                exact_rgb = (v, r['window_end_us']) in frames
                allrows.append(dict(source=source, video_id=v, split=split(v),
                                    entity_track_id=r['entity_track_id'], model=16,
                                    row_index=r['row_index'], window_end_us=r['window_end_us'],
                                    window_start_us=r['window_start_us'], epoch=r['epoch'],
                                    episode_id=r['episode_id'], valid=r['valid'],
                                    failure_reason=r['failure_reason'], appearance_valid=r['appearance_valid'],
                                    kinematics_valid=r['kinematics_valid'], exact_endpoint_rgb=exact_rgb,
                                    geometry_training_eligible=r['geometry_training_eligible'],
                                    m32_available=r['m32_available'], identity_verified=False))
        monitor.check()
    save('states.parquet', allrows)
    for length in [1, 4, 8, 16]:
        for part in ['train', 'validation']:
            local = []
            # Avoid source, episode, epoch and missing-hop boundaries.
            cohorts = collections.defaultdict(list)
            for r in allrows:
                if r['split'] == part:
                    cohorts[(r['source'], r['episode_id'])].append(r)
            for (source, episode), rs in cohorts.items():
                for s in sequences(rs, length):
                    chosen = {r['row_index']: r for r in rs if r['video_id'] == s['video_id']}
                    # row indices are video-local, and sequences are track/video-local.
                    local.append(dict(s, source=source, episode_id=episode,
                                      appearance_mask=[chosen[i]['appearance_valid'] for i in s['row_indices']],
                                      kinematics_mask=[chosen[i]['kinematics_valid'] for i in s['row_indices']],
                                      m32_mask=[chosen[i]['m32_available'] for i in s['row_indices']],
                                      semantic_supervision=False))
            assert local, (length, part, 'No supported sequences')
            save(f'indices/{part}.length{length}.parquet', local)
            counts.append(dict(split=part, length=length, sequences=len(local),
                               all_appearance=sum(all(s['appearance_mask']) for s in local),
                               all_m32=sum(all(s['m32_mask']) for s in local)))
    return counts


class TrainingDataset:
    """Use immutable fitted tensors; RGB exact-PTS fallback is bounded and lossless."""
    def __init__(self, root=OUT, *, _building=False):
        from track1_remediation.store import TieredDataset
        self.root = Path(root)
        status = json.loads((self.root / 'status.json').read_text())
        if not status['training_ingestion_ready'] and not _building:
            raise ValueError('Release has not passed ingestion checks')
        self.readers = {s: TieredDataset(s) for s in BASES}
        rows = pq.read_table(self.root / 'frames.parquet').to_pylist()
        self.frames = {(r['video_id'], r['timestamp_us']): r for r in rows}

    def get_sequence(self, index):
        if split(index['video_id']) != index['split'] or index['source'] not in self.readers:
            raise ValueError('Invalid source/split')
        times = index['end_times_us']
        if len(times) != index['sequence_length'] or any(b-a != 250000 for a,b in zip(times, times[1:])):
            raise ValueError('Sequence has missing hops')
        states = [self.readers[index['source']].get_state(index['video_id'], index['entity_track_id'], t) for t in times]
        meta = [s['core16']['metadata'] for s in states]
        if len({r['epoch'] for r in meta}) != 1 or len({r['episode_id'] for r in meta}) != 1:
            raise ValueError('Sequence crosses an epoch/episode')
        return dict(coefficients=np.stack([s['core16']['coefficients'] for s in states]),
                    xi=np.stack([s['core16']['xi'] for s in states]),
                    appearance=np.stack([s['core16']['appearance_coefficients'] for s in states]),
                    appearance_mask=np.array([r['appearance_valid'] for r in meta], bool),
                    kinematics=[s['kinematics'] for s in states],
                    geometry32=[s['geometry32'] for s in states],
                    metadata=meta, semantic_ground_truth=None)

    def get_frame(self, video_id, timestamp_us):
        r = self.frames[(video_id, int(timestamp_us))]
        result = subprocess.run([sys.executable, '-m', 'track1_reference.native_frame',
                                 video_id, str(r['pts_raw'])], cwd=ROOT, check=True,
                                capture_output=True, timeout=120)
        with Image.open(io.BytesIO(result.stdout)) as image:
            rgb = np.array(image.convert('RGB'))
        if rgb.shape != (1080, 1920, 3):
            raise ValueError('Native RGB shape mismatch')
        return rgb

    def close(self):
        for reader in self.readers.values():
            reader.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def verify_indices():
    states=pq.read_table(OUT/'states.parquet').to_pylist()
    lookup={(r['source'],r['video_id'],r['entity_track_id'],r['window_end_us']):r for r in states}
    assert len(lookup)==len(states)
    participants=collections.defaultdict(set)
    checked=0
    for path in sorted((OUT/'indices').glob('*.parquet')):
        for seq in pq.read_table(path).to_pylist():
            v=seq['video_id']; assert split(v)==seq['split']
            participants[seq['split']].add(v.split('_')[0])
            chosen=[lookup[(seq['source'],v,seq['entity_track_id'],t)] for t in seq['end_times_us']]
            assert len(chosen)==seq['sequence_length']
            assert all(r['valid'] for r in chosen)
            assert all(b['window_end_us']-a['window_end_us']==250000 for a,b in zip(chosen,chosen[1:]))
            assert len({r['epoch'] for r in chosen})==1
            assert {r['episode_id'] for r in chosen}=={seq['episode_id']}
            assert seq['row_indices']==[r['row_index'] for r in chosen]
            for out,key in [('appearance_mask','appearance_valid'),('kinematics_mask','kinematics_valid'),('m32_mask','m32_available')]:
                assert seq[out]==[r[key] for r in chosen]
            checked+=1
    assert participants['train'].isdisjoint(participants['validation'])
    return checked


def verify(monitor):
    from track1_lab.variants import collocation
    from track1_lab.topology_certificate import input_hashes
    files, tensors, certificates = [], 0, 0
    index_checks=verify_indices()
    latest_certificates=0
    paths = []
    for base in BASES.values(): paths += list((base / 'functional').glob('*'))
    for source in BASES:
        paths += [p for p in (ROOT / f'derived/track1_remediation/{source}').glob('*') if p.suffix in ['.npy', '.parquet']]
    for path in sorted(p for p in paths if p.is_file()):
        if path.suffix == '.npy':
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            for start in range(0, len(a), 128):
                if not np.isfinite(a[start:start+128]).all(): raise ValueError('Nonfinite '+str(path))
                monitor.check()
            tensors += 1
            a._mmap.close()
        files.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size))
    for source, base in BASES.items():
        for v in VIDEOS:
            for model in [16, 32]:
                stem = base / 'functional' / f'{v}.m{model}'
                rows = pq.read_table(str(stem)+'.parquet').to_pylist()
                with_arrays = {n: np.load(str(stem)+f'.{n}.npy', mmap_mode='r') for n in ['coefficients','geometry_coefficients','xi','xi_dot','coefficient_cov']}
                for r in rows:
                    if not r['valid']: continue
                    i = r['row_index']; a = with_arrays
                    assert r['latest_input_us'] <= r['window_end_us']
                    c = a['coefficient_cov'][i].astype(float)
                    eig = np.linalg.eigvalsh(c)
                    assert (eig[...,0] >= -1e-5*np.maximum(np.abs(eig).max(-1),1e-20)).all()
                    decoded = collocation(a['coefficients'][i].astype(float), 1.)
                    np.testing.assert_allclose(decoded['xi'], a['xi'][i], rtol=1e-4, atol=2e-5)
                    np.testing.assert_allclose(decoded['xi_dot'], a['xi_dot'][i], rtol=1e-4, atol=2e-4)
                    if r['certified_geometry_valid']:
                        cert = json.loads(r['certificate_json'])
                        assert cert['certified'] and all(cert[k]==value for k,value in input_hashes(a['geometry_coefficients'][i], r['allocation'],r['roles'],r['parents'],1.).items())
                        certificates += 1
                for a in with_arrays.values(): a._mmap.close()
            monitor.check()
    for source in BASES:
        for v in VIDEOS:
            base=ROOT/f'derived/track1_remediation/{source}'
            rows=pq.read_table(base/f'{v}.geometry.parquet').to_pylist()
            a=np.load(base/f'{v}.coefficients.npy',mmap_mode='r')
            for r in rows:
                if not r['valid']: continue
                cert=json.loads(r['certificate_json'])
                expected=input_hashes(a[r['row_index']],r['allocation'],r['roles'],r['parents'],r['window_seconds'])
                assert cert['certified'] and all(cert[k]==value for k,value in expected.items())
                assert r['window_seconds']>0 and r['window_start_us']<r['window_end_us']
                latest_certificates+=1
            a._mmap.close()
            monitor.check()
    LOG.info('Verified %s tensor files and %s certificate bindings', tensors, certificates)
    source_results = []
    for v in VIDEOS:
        p = ROOT / f'dataset/videos/{v}.MP4'
        expected = pq.read_table(ROOT / f'canonical/v1/videos.{v}.parquet', columns=['sha256']).to_pylist()[0]['sha256']
        assert digest(p) == expected, ('Original source checksum', v)
        source_results.append(dict(video_id=v, sha256=expected, bytes=p.stat().st_size))
        LOG.info('Source verified %s', v); monitor.check()
    # Every indexed PTS must exist in the canonical native-frame inventory.
    frames = pq.read_table(OUT / 'frames.parquet').to_pylist()
    for v in VIDEOS:
        canonical = {r['timestamp_us']: (r['pts_raw'],r['native_frame_uid']) for r in pq.read_table(ROOT / f'canonical/v1/frames.{v}.parquet', columns=['timestamp_us','pts_raw','native_frame_uid']).to_pylist()}
        for r in frames:
            if r['video_id']==v: assert canonical[r['timestamp_us']] == (r['pts_raw'],r['native_frame_uid'])
    decoded_count, sequence_count = 0, 0
    try:
        with TrainingDataset(_building=True) as data:
            for v in VIDEOS:
                local = [r for r in frames if r['video_id']==v]
                for i in sorted({0,len(local)//2,len(local)-1}):
                    rgb = data.get_frame(v,local[i]['timestamp_us']); assert rgb.dtype==np.uint8
                    decoded_count += 1
            for index_path in sorted((OUT / 'indices').glob('*.parquet')):
                rows = pq.read_table(index_path).to_pylist()
                for source in BASES:
                    candidates = [r for r in rows if r['source']==source]
                    if candidates:
                        result = data.get_sequence(candidates[0]); assert result['coefficients'].shape == (candidates[0]['sequence_length'],4,35)
                        assert result['xi'].shape == (candidates[0]['sequence_length'],4,68)
                        sequence_count += 1
                monitor.check()
    except BaseException:
        atomic_json(OUT / 'status.json', dict(training_ingestion_ready=False, error='Ingestion smoke failed',track2_started=False))
        raise
    result = dict(tensor_files=tensors, certificate_bindings=certificates, latest_epoch_certificate_bindings=latest_certificates, sequence_indices_checked=index_checks, source_videos=source_results,
                  indexed_native_frames=len(frames), native_pts_inventory_checked=True,
                  exact_rgb_decodes_sampled=decoded_count, sequence_batches_sampled=sequence_count,
                  full_rgb_partition_materialized=False, tensor_files_manifest=files,
                  scientific_acceptance=False, track2_started=False)
    atomic_json(REPORT / 'verification.json', result)
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--verify', action='store_true'); p.add_argument('--resume', action='store_true'); args=p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    OUT.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    import fcntl
    lock=(ROOT/'work/track1_retrain/training_prepare.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with Monitor() as monitor:
        if args.verify:
            manifest=json.loads((OUT/'manifest.json').read_text())
            for path,expected in manifest['artifacts'].items():
                assert digest(ROOT/path)==expected, ('Release artifact changed',path)
            for entry in json.loads((REPORT/'verification.json').read_text())['tensor_files_manifest']:
                assert digest(ROOT/entry['path'])==entry['sha256'], ('Tensor dependency changed',entry['path'])
        if not args.verify:
            if not args.resume:
                snapshot_reviews()
            else:
                with contextlib.closing(sqlite3.connect(OUT/'reviews.sqlite')) as db:
                    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            atomic_json(OUT/'status.json',dict(training_ingestion_ready=False,phase='preprocessing',track2_started=False))
            review_summary = reviews(); LOG.info('Review export %s',review_summary)
            frames=frame_index(); counts=build_indices(frames,monitor)
            atomic_json(REPORT/'preprocessing.json',dict(reviews=review_summary,sequences=counts,source_backed_rgb=True,independent_references=False))
        atomic_json(OUT/'status.json',dict(training_ingestion_ready=False,phase='verification',track2_started=False))
        result=verify(monitor)
        with contextlib.closing(sqlite3.connect(OUT/'reviews.sqlite')) as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            db.execute('PRAGMA journal_mode=DELETE')
        artifacts={str(p.relative_to(ROOT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['status.json','manifest.json','reviews.sqlite-shm','reviews.sqlite-wal']}
        atomic_json(OUT/'manifest.json',dict(version=1,artifacts=artifacts,code_sha256=digest(Path(__file__)),
                    existing_tensors_immutable=True,model_training_started=False,track2_started=False))
        atomic_json(OUT/'status.json',dict(training_ingestion_ready=True,
                    ready_scope='retrospective Track 1 reconstruction with existing M16 tensors, masked appearance/kinematics and optional certified M32',
                    rgb_mode='exact-PTS original MP4 decode; incomplete NPY cache not required',
                    machine_annotations='separate optional projected-relation weak labels; no physical/contact/identity ground truth',
                    scientific_acceptance=False,independent_accuracy_validated=False,finalized=False,track2_started=False))
        LOG.info('READY: %s',result.keys())

if __name__=='__main__':main()
