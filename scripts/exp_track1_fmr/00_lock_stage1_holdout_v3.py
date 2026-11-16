"""Freeze successor code, then select metadata-only episodes from newly acquired participants."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import av
from pipeline.common import ROOT, digest

BASE = ROOT / 'reports/track1_fmr'
PREVIOUS = {'P02', 'P03', 'P04', 'P06', 'P08', 'P11', 'P18', 'P21', 'P22', 'P27', 'P30'}
VIDEOS = ['P14_02', 'P26_15']


def code_hashes():
    paths = [ROOT / 'scripts/exp_track1_fmr/01_bench_grid_invariance.py',
             *sorted((ROOT / 'src/track1_fmr').glob('*.py'))]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)


def freeze():
    write_new(BASE / 'stage1_v3_code_freeze_v2.json', dict(
        code_hashes=code_hashes(), preparation_sha256=digest(Path(__file__).resolve()),
        parameters=dict(v_min=.0035, sigma_flow=.005, grid=160, comparison_grid=320),
        participants=VIDEOS, development_participants=sorted(PREVIOUS),
        selection='Four action-centered 30-frame intervals per video; >=5s centre separation; deterministic hash ranking among feasible combinations',
        selection_has_decoded_rgb=False))
    print('Successor code frozen before holdout episode selection.', flush=True)


def timestamp_us(value):
    h, m, s = value.split(':')
    return round((int(h) * 3600 + int(m) * 60 + float(s)) * 1e6)


def lock():
    freeze_path = BASE / 'stage1_v3_code_freeze_v2.json'
    frozen = json.loads(freeze_path.read_text())
    if frozen['code_hashes'] != code_hashes() or frozen['preparation_sha256'] != digest(Path(__file__).resolve()):
        raise ValueError('Successor changed after code freeze')
    official = list(csv.DictReader((ROOT / 'dataset/metadata/official_md5.csv').open()))
    action_path = ROOT / 'dataset/actions/EPIC_100_train.csv'
    action_rows = list(csv.DictReader(action_path.open()))
    hashes = {str(action_path.relative_to(ROOT)): digest(action_path),
              'reports/track1_fmr/stage1_v3_code_freeze_v2.json': digest(freeze_path)}
    episodes, acquisitions = [], []
    for video in VIDEOS:
        participant = video.split('_')[0]
        if participant in PREVIOUS:
            raise ValueError('Previously used participant cannot enter fresh holdout')
        path = ROOT / f'work/track1_fmr/{video}.mirror.MP4'
        expected = next(r for r in official if r['file_remote_path'].endswith(video + '.MP4'))
        if not path.exists() or digest(path, 'md5') != expected['md5']:
            raise ValueError('Complete publisher-MD5-verified video required: ' + video)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        # Demux compressed packets only; no frame pixels are decoded or inspected here.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (1920, 1080):
                raise ValueError('Existing exact-PTS RGB decoder requires native 1920x1080')
            pts = sorted({packet.pts for packet in container.demux(stream) if packet.pts is not None})
            frames = [dict(video_id=video, pts_raw=int(t), timestamp_us=round(t * stream.time_base * 1e6)) for t in pts]
        candidates = []
        for row in action_rows:
            if row['video_id'] != video:
                continue
            start, end = timestamp_us(row['start_timestamp']), timestamp_us(row['stop_timestamp'])
            centre = (start + end) // 2
            clip = [f for f in frames if f['timestamp_us'] >= max(start, centre - 250_000)][:30]
            if len(clip) != 30 or clip[-1]['timestamp_us'] >= end:
                continue
            identity = row['narration_id']
            candidates.append(dict(episode_id='v3_' + identity, video_id=video, participant=participant,
                                   frames=clip, source_path=str(path.relative_to(ROOT)), centre_us=centre, narration=row['narration'],
                                   activity_evidence='Publisher action interval; not independent frame-level motion truth'))
        candidates.sort(key=lambda e: hashlib.sha256(('stage1_holdout_v3:' + e['episode_id']).encode()).hexdigest())
        chosen = next((combo for combo in itertools.combinations(candidates, 4)
                       if all(abs(a['centre_us'] - b['centre_us']) >= 5_000_000
                              for a, b in itertools.combinations(combo, 2))), None)
        if chosen is None:
            raise ValueError('Insufficient four separated labeled action intervals: ' + video)
        episodes.extend(chosen)
        acquisitions.append(dict(video_id=video, publisher_md5=expected['md5'],
                                 url='https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/' + expected['file_remote_path'],
                                 sha256=hashes[str(path.relative_to(ROOT))], bytes=path.stat().st_size))
    write_new(BASE / 'stage1_holdout_v3_split.json', dict(
        name='stage1_holdout_v3', episodes=episodes, input_hashes=hashes,
        code_hashes=frozen['code_hashes'], acquisitions=acquisitions,
        freshness='Newly acquired P14/P26; absent from all recorded prior training/evaluation participants; no RGB decoding before code/episode lock. Pretrained external-model exposure is unknown.',
        participants=[v.split('_')[0] for v in VIDEOS], excluded_participants=sorted(PREVIOUS),
        selection_has_decoded_rgb=False, source_frame_index='Packet PTS metadata; benchmark enforces exact decoded PTS'))
    print('Locked eight episodes from two new participants, without RGB decoding.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'lock'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else lock()
