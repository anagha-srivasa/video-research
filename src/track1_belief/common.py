"""Scoped artifacts, bounded resources, and immutable-input receipts."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.common import ROOT, atomic_json, digest

C = json.loads((ROOT / 'configs/track1_belief.json').read_text())
D = ROOT / 'derived/track1_belief'
R = ROOT / 'reports/track1_belief'
W = ROOT / 'work/track1_belief'
LOG = logging.getLogger('track1_belief')


def initialize():
    for folder in (D, R, W):
        folder.mkdir(parents=True, exist_ok=True)


def save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression=None)
    temporary.replace(path)


def arrays(path, **values):
    """Uncompressed NumPy archive, with atomic publication and no pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **{k: np.asarray(v) for k, v in values.items()})
    temporary.replace(path)


def split(video):
    if video in C['train']:
        return 'train'
    if video in C['validation']:
        return 'validation'
    raise ValueError('Participant outside the declared development cohort')


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def guard():
    rss = 0
    for process in (psutil.Process(), *psutil.Process().children(recursive=True)):
        try:
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    available = psutil.virtual_memory().available
    if rss / 2**30 > C['memory_limit_gib'] or available / 2**30 < C['minimum_available_gib']:
        raise MemoryError(f'Track 1 guard: tree RSS {rss / 2**30:.2f} GiB; available {available / 2**30:.2f} GiB')
    return {'tree_rss_gib': rss / 2**30, 'available_gib': available / 2**30}


class Monitor:
    def __enter__(self):
        initialize()
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        while not self.stop.is_set():
            try:
                row = dict(time=time.time(), pid=os.getpid(), **guard())
                with (W / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            except Exception as exc:
                self.error = exc
                return
            self.stop.wait(3)

    def check(self):
        if self.error is not None:
            raise self.error
        return guard()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(4)


def baseline_receipt():
    """Snapshot old code and prior declared release inputs, never mutate them."""
    initialize()
    path = R / 'preserved_inputs.json'
    if path.exists():
        return json.loads(path.read_text())
    files = {}
    for folder in ('src', 'configs', 'tests'):
        for p in (ROOT / folder).rglob('*'):
            if p.suffix not in ('.py', '.json', '.yaml', '.toml') or 'track1_belief' in str(p):
                continue
            files[str(p.relative_to(ROOT))] = digest(p)
    frozen = ROOT / 'reports/track2_feasibility/frozen_inputs.json'
    if frozen.exists():
        files.update(json.loads(frozen.read_text())['source_hashes'])
    for p in [ROOT/'reports/track1_freeze_manifest.json', ROOT/'reports/track1_refinement/freeze_manifest.json']:
        if p.exists():
            files[str(p.relative_to(ROOT))] = digest(p)
            for name, info in json.loads(p.read_text()).get('files', {}).items():
                files[name] = info['sha256'] if isinstance(info, dict) else info
    atomic_json(path, files)
    return files


def check_preserved():
    files = baseline_receipt()
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Preserved input changed: ' + name)
    return len(files)


def source_hashes():
    paths = sorted((ROOT / 'src/track1_belief').glob('*.py'))
    paths += [ROOT / 'configs/track1_belief.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def output_bytes():
    return sum(p.stat().st_size for folder in (D, R, W) for p in folder.rglob('*') if p.is_file())
