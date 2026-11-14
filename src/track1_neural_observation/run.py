"""Stream existing native RGB through a frozen causal TAPNext++ checkpoint."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from pipeline.common import ROOT, digest, atomic_json
from track1_belief.common import arrays, save
from track1_observation_revision.observations import transport
from .tracker import Tracker, VENDOR, CHECKPOINT, resources

WORK = ROOT / 'work/track1_neural_observation'
DATA = ROOT / 'derived/track1_neural_observation'
REPORT = ROOT / 'reports/track1_neural_observation'


def initialize():
    for p in (WORK, DATA, REPORT):
        p.mkdir(parents=True, exist_ok=True)


def receipt(stage):
    inputs = [CHECKPOINT, ROOT / 'reports/track1_belief/anchor_data.json',
              ROOT / 'dataset/context_v3/frames.parquet', ROOT / 'src/track1_belief/data.py',
              ROOT / 'src/track1_encoder_revision/source_stream.py',
              ROOT / 'src/track1_observation_revision/observations.py']
    inputs += list((ROOT / 'src/track1_neural_observation').glob('*.py'))
    inputs += list(VENDOR.rglob('*.py'))
    if stage == 'davis':
        inputs.append(ROOT / 'work/track1_observation_revision/tapvid_davis/receipt.json')
    document = dict(source_commit='ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b',
                    source_url='https://github.com/google-deepmind/tapnet',
                    checkpoint_url='https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt',
                    hashes={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                    resolution=512, precision='float32', causal=True, recurrent_step='one native frame',
                    visibility='official visible logit >0, plus finite in-image coordinates',
                    stage=stage,
                    initialization=('official first-visible queries; future-query tokens masked by the frozen model'
                                    if stage == 'davis' else 'same retrospective manual seed and exact query IDs as frozen LK ledger'),
                    geometry=('point-level benchmark; no object geometry evaluated' if stage == 'davis'
                              else 'same adjacent/original-anchor material transport as local LK replay'),
                    independent_accuracy=stage == 'davis', no_validation_selection=True, promoted=False)
    path = REPORT / f'protocol.{stage}.json'
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError('Neural protocol or dependencies changed; preserve results before a new run')
    atomic_json(path, document)
    return digest(path)


def smoke(tracker):
    from PIL import Image
    image_path = ROOT / 'work/track1_belief_audit/observation_478cb915824f4d9b0d080594_0.png'
    rgb = np.asarray(Image.open(image_path).convert('RGB'))
    h, w = rgb.shape[:2]
    queries = np.array([[w * .25, h * .25], [w * .5, h * .5], [w * .75, h * .75]], np.float32)
    rows = []
    for i in range(3):
        xy, visible, seconds = tracker.step(rgb, i * 20000, queries if i == 0 else None)
        rows.append(dict(frame=i, seconds=seconds, visible=int(visible.sum()),
                         max_static_displacement=float(np.linalg.norm(xy - queries, axis=1).max()), **resources()))
        print('smoke', rows[-1], flush=True)
    tracker.reset()
    repeat, _, _ = tracker.step(rgb, 0, queries)
    tracker.reset()
    # No future frame has entered either initial call. Compare repeated state.
    again, _, _ = tracker.step(rgb, 0, queries)
    np.testing.assert_allclose(repeat, again, rtol=0, atol=1e-4)
    tracker.reset()
    atomic_json(REPORT / 'smoke.json', dict(rows=rows, repeated_initial_prediction=True,
                                           scope='integration check on a diagnostic image; not accuracy evaluation'))


def native(tracker, protocol_hash, limit):
    from track1_belief import data as existing_reader
    # Its existing isolated decoder already validates exact PTS and byte counts.
    # Relocate only its request-file directory; baseline source/data stay intact.
    original_work = existing_reader.W
    existing_reader.W = WORK
    try:
        metadata = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())['episodes']
        frames = pq.read_table(ROOT / 'dataset/context_v3/frames.parquet').to_pylist()
        results = []
        for meta in metadata[:limit]:
            uid, video = meta['episode_id'], meta['video_id']
            output, marker = DATA / (uid + '.npz'), DATA / (uid + '.json')
            if marker.exists():
                completed = json.loads(marker.read_text())
                if completed['protocol_sha256'] != protocol_hash or digest(output) != completed['sha256']:
                    raise ValueError('Neural checkpoint changed')
                results.append(completed)
                continue
            path = ROOT / meta['path']
            if digest(path) != meta['sha256']:
                raise ValueError('Original point ledger changed')
            with np.load(path, allow_pickle=False) as z:
                original = {k: z[k] for k in z.files}
            times = original['timestamp_us']
            selected = [f for f in frames if f['video_id'] == video and int(f['timestamp_us']) in set(times)]
            selected.sort(key=lambda f: f['timestamp_us'])
            if [f['timestamp_us'] for f in selected] != times.tolist():
                raise ValueError('Native source timestamps mismatch')
            indices = np.flatnonzero(times >= meta['manual_seed_us'])
            selected = [selected[i] for i in indices]
            positions = np.zeros((len(times), len(original['point_initial']), 2), np.float32)
            visible = np.zeros(positions.shape[:2], bool)
            seconds = np.zeros(len(times))
            tracker.reset()
            for j, rgb, decode_seconds in existing_reader.native_frames(video, selected, uid):
                i = indices[j]
                xy, vis, elapsed = tracker.step(rgb, int(times[i]), original['point_initial'] if j == 0 else None)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < rgb.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < rgb.shape[0])
                positions[i], visible[i], seconds[i] = xy, vis & inside, elapsed
                if j % 20 == 0:
                    row = dict(episode_id=uid, frame=j, total=len(indices), seconds=elapsed,
                               decode_seconds=decode_seconds, **resources())
                    with (WORK / 'resources.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print(row, flush=True)
            tracker.reset()
            roles = original['point_roles']
            fit_ids = roles == 0
            material = transport(positions[:, fit_ids], visible[:, fit_ids], original['point_initial'][fit_ids],
                                 original['seed_reference'], times, times >= meta['manual_seed_us'])
            arrays(output, timestamp_us=times, point_positions=positions, point_visible=visible,
                   point_initial=original['point_initial'], point_roles=roles, seconds=seconds, **material)
            completed = dict(episode_id=uid, video=video, split=meta['split'], proposed=len(times), decoded=len(indices),
                             material_supported=int(material['valid'].sum()), lk_global_supported=int(original['valid'].sum()),
                             neural_visible_point_frames=int(visible.sum()),
                             reacquired_point_frames=int((visible[1:] & ~visible[:-1] & np.maximum.accumulate(visible, axis=0)[:-1]).sum()),
                             median_frame_seconds=float(np.median(seconds[indices])),
                             source_sha256=meta['sha256'], protocol_sha256=protocol_hash, sha256=digest(output))
            atomic_json(marker, completed)
            results.append(completed)
            atomic_json(REPORT / 'progress.json', dict(completed=len(results), requested=min(limit, len(metadata)), episodes=results))
        save(REPORT / 'episodes.parquet', results)
        atomic_json(REPORT / 'status.json', dict(completed=len(results), requested=min(limit, len(metadata)),
                                                neural_tracking_completed=True, independent_accuracy=False,
                                                promoted=False, no_raw_rgb_copies=True))
    finally:
        existing_reader.W = original_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'native', 'davis', 'all'), default='smoke')
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        raise ValueError('Limit must be between one and thirty episodes')
    initialize()
    with (WORK / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = ('smoke', 'davis', 'native') if args.stage == 'all' else (args.stage,)
        receipts = {stage: receipt(stage) for stage in stages}
        tracker = Tracker()
        for stage in stages:
            if stage == 'smoke':
                smoke(tracker)
            elif stage == 'native':
                native(tracker, receipts[stage], args.limit)
            else:
                from .benchmark import evaluate
                evaluate(tracker, receipts[stage], args.limit)


if __name__ == '__main__':
    main()
