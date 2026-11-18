"""Published TAP-Vid first-query metrics, frozen TAPNext++ versus causal LK."""
import ast
import importlib
import json
import pickle
from typing import Mapping
import numpy as np
import cv2
import torch
from pipeline.common import ROOT, digest, atomic_json
from track1_belief.common import arrays, save
from .tracker import VENDOR, resources


def official_metrics():
    """Load the exact NumPy metric without unrelated TensorFlow readers."""
    source = VENDOR / 'tapnet/tapvid/evaluation_datasets.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_tapvid_metrics')
    namespace = {'np': np, 'Mapping': Mapping}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['compute_tapvid_metrics']


class ArrayUnpickler(pickle.Unpickler):
    """Allow only array constructors in the publisher's numeric dataset."""
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'), ('numpy', 'dtype'),
                   ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
                   ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported dataset constructor: {module}.{name}')
        return getattr(importlib.import_module(module), name)


def lk_tracks(video, queries):
    """Same LK/FB parameters as the native baseline; no ground-truth reseeding."""
    cv2.setNumThreads(1)
    n, count = len(video), len(queries)
    positions = np.zeros((n, count, 2), np.float32)
    visible = np.zeros((n, count), bool)
    current = np.zeros((count, 2), np.float32)
    alive = np.zeros(count, bool)
    previous = None
    h, w = video.shape[1:3]
    for i, rgb in enumerate(video):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ids = np.flatnonzero(alive)
        if previous is not None and len(ids):
            p = current[ids].reshape(-1, 1, 2)
            forward, a, _ = cv2.calcOpticalFlowPyrLK(previous, gray, p, None, winSize=(25, 25), maxLevel=4)
            backward, b, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, winSize=(25, 25), maxLevel=4)
            q = forward[:, 0]
            keep = a.ravel().astype(bool) & b.ravel().astype(bool) & (np.linalg.norm(backward[:, 0] - p[:, 0], axis=1) <= 1.)
            keep &= np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            alive[ids] = keep
            current[ids[keep]] = q[keep]
        new = queries[:, 0] == i
        current[new] = queries[new, 1:][:, ::-1]
        alive[new] = True
        positions[i], visible[i], previous = current, alive, gray
    return positions, visible


def evaluate(tracker, protocol_hash, limit=30):
    from .run import WORK, REPORT, DATA
    from tapnet.tapnextpp.votsp2026 import utils
    dataset = ROOT / 'work/track1_observation_revision/tapvid_davis/tapvid_davis.pkl'
    publisher = json.loads((dataset.parent / 'receipt.json').read_text())
    if digest(dataset) != publisher['uncompressed_sha256']:
        raise ValueError('Benchmark dataset changed')
    metrics = official_metrics()
    with dataset.open('rb') as stream:
        cases = ArrayUnpickler(stream).load()
    rows = []
    for name in sorted(cases)[:limit]:
        destination = DATA / f'davis.{name}.npz'
        marker = DATA / f'davis.{name}.json'
        if marker.exists():
            done = json.loads(marker.read_text())
            if done['protocol_sha256'] != protocol_hash or digest(destination) != done['sha256']:
                raise ValueError('Benchmark checkpoint changed')
            rows.extend(done['rows'])
            continue
        case = cases[name]
        video = case['video']
        h, w = video.shape[1:3]
        target = case['points'] * np.array([w, h])
        hidden = np.asarray(case['occluded'], bool)
        keep = (~hidden).any(axis=1)
        target, hidden = target[keep], hidden[keep]
        first = np.argmax(~hidden, axis=1)
        queries = np.array([[i, target[k, i, 1], target[k, i, 0]] for k, i in enumerate(first)], np.float32)
        query_model = queries.copy()
        query_model[:, 1:] *= [256. / h, 256. / w]
        qt = torch.from_numpy(query_model).to(tracker.device)[None]
        outputs, visible, seconds = [], [], []
        state = None
        for i, rgb in enumerate(video):
            import time
            resources()
            start = time.perf_counter()
            frame = utils.preprocess_frame(rgb[..., ::-1], torch.device(tracker.device), 512)
            with torch.inference_mode():
                xy, _, vis, state = tracker.model._model(video=frame, query_points=qt if i == 0 else None, state=state)
            xy = xy[0, 0].cpu().numpy()[:, ::-1].copy() * [w / 256., h / 256.]
            visible.append((vis[0, 0, :, 0] > 0).cpu().numpy())
            if not np.isfinite(xy).all():
                raise FloatingPointError('Nonfinite benchmark prediction')
            outputs.append(xy)
            seconds.append(time.perf_counter() - start)
            if i % 20 == 0:
                print(dict(benchmark=name, frame=i, total=len(video), seconds=seconds[-1], **resources()), flush=True)
        state = None
        neural, neural_vis = np.array(outputs), np.array(visible)
        lk, lk_vis = lk_tracks(video, queries)
        scale = np.array([256. / w, 256. / h])
        case_rows = []
        for method, tracks, seen in [('TAPNext++512', neural, neural_vis), ('LK_forward_backward', lk, lk_vis)]:
            result = metrics(query_model[None], hidden[None], (target * scale)[None],
                             (~seen.T)[None], (tracks.transpose(1, 0, 2) * scale)[None], query_mode='first')
            case_rows.append(dict(video=name, method=method, frames=len(video), points=len(queries),
                                  **{k: float(v[0]) for k, v in result.items()}))
        arrays(destination, neural=neural, neural_visible=neural_vis, lk=lk, lk_visible=lk_vis,
               query_points=queries, seconds=seconds)
        atomic_json(marker, dict(protocol_sha256=protocol_hash, sha256=digest(destination), rows=case_rows))
        rows.extend(case_rows)
        print('completed benchmark', case_rows, flush=True)
    save(REPORT / 'davis.parquet', rows)
    atomic_json(REPORT / 'davis.json', dict(videos=len(rows) // 2, available=len(cases),
                                           query_mode='first', metric_resolution=256,
                                           input_resolution=512, input='published 480px-high RGB, bilinearly resized to 512 square',
                                           tuned_on_benchmark=False, rows=rows,
                                           statement='Frozen published tracker versus LK, not evidence of a novel tracker beating the published model.'))
