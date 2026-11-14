"""Reproducible, bounded audit repairs. python -m track1_observation_revision.run."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import time
import numpy as np
import psutil
import pyarrow.parquet as pq
from pipeline.common import ROOT, atomic_json, digest
from track1_belief.common import save, arrays
from .benchmark import episodes, estimate, METHODS, ARMS, event_head, score_events, continuation
from .geometry import fit_budget_path
from .observations import replay

REPORT = ROOT / 'reports/track1_observation_revision'
WORK = ROOT / 'work/track1_observation_revision'
DATA = ROOT / 'derived/track1_observation_revision'
Q = (1000., 10000., 100000.)
WINDOWS = (.1, .2, .4)
DECAYS = (.1, .25, .5, float('inf'))
EVENT_SPEEDS = (20., 50., 100.)


def guard():
    # Workers are threads in this process, so process RSS includes every worker.
    rss, available = psutil.Process().memory_info().rss, psutil.virtual_memory().available
    if rss > 10 * 2**30 or available < 6 * 2**30:
        raise MemoryError(f'Resource guard: RSS={rss}, available={available}')
    return dict(rss_gib=rss / 2**30, available_gib=available / 2**30)


def protocol():
    for directory in (REPORT, WORK, DATA):
        directory.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / 'src/track1_observation_revision').glob('*.py'))
    files += sorted((ROOT / 'src/track1_belief').glob('*.py'))
    document = dict(version=1, source_hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
                    seed=20261112, split='60 independent synthetic episodes per split; full 5x6x2 factorial',
                    q_grid=Q, cv_window_grid=WINDOWS, decay_grid=[.1, .25, .5, 'infinity'],
                    event_speeds=EVENT_SPEEDS, state_objective='mean episode (position RMSE + 0.1 seconds * velocity RMSE)',
                    selection='training only, same three trials for every filter and CV',
                    event_selection='training micro F1, then lower false alarms; common fixed head also reported',
                    comparison='common support across all candidates, coverage retained separately',
                    scope='synthetic mechanism tests and existing exploratory source geometry; no physical validation',
                    baseline_manifest_sha256=digest(ROOT / 'reports/track1_belief/manifest.json'),
                    promotion=False, neural_tracker='blocked: automatic approval review lacks workspace credits')
    path = REPORT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(document)):
        raise RuntimeError('Protocol/source changed. Preserve results and use a new experiment directory.')
    atomic_json(path, document)
    return digest(path)


def candidates():
    return [(m, p) for m in METHODS for p in ((.2,) if m == 'nearest' else WINDOWS if m == 'cv' else Q)]


def compute_episode(ep, receipt):
    path = DATA / (ep['id'] + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data['receipt']) != receipt:
                raise ValueError('Stale prediction cache')
            return {k: data[k] for k in ('means', 'valid', 'seconds')}
    guard()
    means, valid, seconds = [], [], []
    for method, parameter in candidates():
        start = time.perf_counter()
        mean, support = estimate(ep, method, parameter)
        means.append(mean)
        valid.append(support)
        seconds.append(time.perf_counter() - start)
    result = dict(means=np.array(means), valid=np.array(valid), seconds=np.array(seconds))
    arrays(path, receipt=receipt, **result)
    return result


def collect(split, receipt, workers):
    collection = list(episodes(split))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = pool.map(lambda e: compute_episode(e, receipt), collection)
        result = []
        for ep, out in zip(collection, outputs):
            result.append((ep, out))
            resource = dict(episode=ep['id'], time=time.time(), **guard())
            with (WORK / 'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(resource) + '\n')
            print(f"{split}: {len(result)}/{len(collection)} episodes", flush=True)
    return result


def common_support(ep, output):
    return output['valid'].all(axis=0) & (ep['time'] >= 1.)


def fit_selection(train):
    scores = np.zeros(len(candidates()))
    for ep, output in train:
        mask = common_support(ep, output)
        if not mask.any():
            raise ValueError('No common training support')
        errors = output['means'][:, mask] - np.c_[ep['truth'], ep['velocity']][None, mask]
        scores += np.sqrt(np.mean(np.sum(errors[..., :2]**2, axis=-1), axis=1))
        scores += .1 * np.sqrt(np.mean(np.sum(errors[..., 2:]**2, axis=-1), axis=1))
    selected = {}
    for method in METHODS:
        choices = [i for i, (m, _) in enumerate(candidates()) if m == method]
        index = min(choices, key=lambda i: scores[i])
        decay_losses = np.zeros(len(DECAYS))
        detector_counts = np.zeros((len(EVENT_SPEEDS), 3))
        for ep, out in train:
            mask = common_support(ep, out)
            for h in (.25, .5):
                for cutoff in np.arange(1., 5.51, .25):
                    i = np.searchsorted(ep['time'], cutoff, side='right') - 1
                    if not mask[i]:
                        continue
                    target = int(np.searchsorted(ep['time'], ep['time'][i] + h))
                    if target >= len(mask):
                        continue
                    actual = ep['time'][target] - ep['time'][i]
                    for d, decay in enumerate(DECAYS):
                        error = continuation(out['means'][index, i], actual, decay) - ep['truth'][target]
                        decay_losses[d] += error @ error
            for j, speed in enumerate(EVENT_SPEEDS):
                events = event_head(ep['time'], out['means'][index, :, 2:], out['valid'][index], speed)
                score = score_events(ep, events)
                detector_counts[j] += [score['matched_events'], score['reference_events'], score['false_alarms']]
        def event_objective(j):
            hit, reference, false = detector_counts[j]
            return (-2 * hit / max(reference + hit + false, 1), false, j)
        speed_index = min(range(len(EVENT_SPEEDS)), key=event_objective)
        decay = DECAYS[int(np.argmin(decay_losses))]
        selected[method] = dict(index=index, parameter=candidates()[index][1],
                                decay='infinity' if np.isinf(decay) else decay,
                                event_speed=EVENT_SPEEDS[speed_index],
                                training_objective=float(scores[index] / len(train)))
    return dict(methods=selected, trials=[dict(method=m, parameter=p, score=float(s / len(train)))
                                        for (m, p), s in zip(candidates(), scores)])


def evaluate(collection, selection):
    state_rows, event_rows, forecast_rows, ablation_rows = [], [], [], []
    for ep, out in collection:
        common = common_support(ep, out)
        base = {k: ep[k] for k in ('id', 'split', 'kind', 'corruption', 'duration')}
        for method, selected in selection['methods'].items():
            index = selected['index']
            mean, valid = out['means'][index], out['valid'][index]
            errors = mean - np.c_[ep['truth'], ep['velocity']]
            state_rows.append(dict(base, method=method, attempted=int((ep['time'] >= 1).sum()),
                                   supported=int((valid & (ep['time'] >= 1)).sum()), matched=int(common.sum()),
                                   position_mse=float(np.mean(np.sum(errors[common, :2]**2, axis=1))),
                                   velocity_mse=float(np.mean(np.sum(errors[common, 2:]**2, axis=1))),
                                   seconds=float(out['seconds'][index])))
            for detector, speed in (('common20', 20.), ('train_selected', selected['event_speed'])):
                events = event_head(ep['time'], mean[:, 2:], valid, speed)
                event_rows.append(dict(base, method=method, detector=detector,
                                       **score_events(ep, events)))
            decay = float('inf') if selected['decay'] == 'infinity' else selected['decay']
            # Fixed proposed cutoffs, never shifted to a conveniently valid frame.
            for t in np.arange(1., 5.51, .25):
                i = np.searchsorted(ep['time'], t, side='right') - 1
                for horizon in (.25, .5):
                    target = np.searchsorted(ep['time'], ep['time'][i] + horizon)
                    for name, d in (('cv', float('inf')), ('train_damped', decay)):
                        ok = target < len(valid) and bool(common[i])
                        error = continuation(mean[i], ep['time'][target] - ep['time'][i], d) - ep['truth'][target] if ok else None
                        forecast_rows.append(dict(base, method=method, continuation=name, horizon=horizon,
                                                  cutoff=float(ep['time'][i]), supported=ok,
                                                  error_squared=float(error @ error) if ok else None))
        # Fixed-q contrasts isolate mechanisms from tuning differences.
        for index, (method, parameter) in enumerate(candidates()):
            if method not in ARMS:
                continue
            error = out['means'][index] - np.c_[ep['truth'], ep['velocity']]
            ablation_rows.append(dict(base, method=method, q=parameter,
                                      position_mse=float(np.mean(np.sum(error[common, :2]**2, axis=1))),
                                      velocity_mse=float(np.mean(np.sum(error[common, 2:]**2, axis=1)))))
    return state_rows, event_rows, forecast_rows, ablation_rows


def synthetic(receipt, workers):
    train = collect('train', receipt, workers)
    selection = fit_selection(train)
    path = REPORT / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise RuntimeError('Locked selection differs on rerun')
    atomic_json(path, selection)  # Published before validation is opened.
    validation = collect('validation', receipt, workers)
    for name, rows in zip(('states', 'events', 'forecasts', 'factorial'), evaluate(train + validation, selection)):
        save(REPORT / (name + '.parquet'), rows)
    return selection


def geometry():
    old = pq.read_table(ROOT / 'reports/track1_belief/geometry.parquet').to_pylist()
    keys = {(r['video_id'], r['mask_uid']) for r in old}
    rows = []
    for video in sorted({v for v, _ in keys}):
        sources = pq.read_table(ROOT / f'canonical/v2/masks.{video}.parquet', columns=['mask_uid', 'contours_json']).to_pylist()
        for row in sources:
            if (video, row['mask_uid']) not in keys:
                continue
            guard()
            rings = json.loads(row['contours_json'])
            points = [np.array(r['points'], float) for r in rings]
            roles = [{'inner': 'hole'}.get(r['role'], r['role']) for r in rings]
            parents = [r.get('parent_ring', -1) for r in rings]
            if not points or min(map(len, points)) < 4:
                for budget in (16, 24, 32):
                    rows.append(dict(video=video, mask_uid=row['mask_uid'], budget=budget,
                                     supported=False, failure='insufficient_source_vertices'))
                continue
            for fit in fit_budget_path(points, roles, parents, max_certificate_attempts=2):
                if fit['supported']:
                    prefix = DATA / 'geometry' / f"{row['mask_uid']}.{fit['budget']}"
                    arrays(prefix.with_suffix(f'.{fit["budget"]}.npz'), coefficients=fit['coefficients'],
                           centre=fit['centre'], scale=fit['scale'], allocation=fit['allocation'],
                           roles=fit['roles'], parents=fit['parents'])
                    atomic_json(prefix.with_suffix(f'.{fit["budget"]}.json'), fit['certificate'])
                rows.append(dict(video=video, mask_uid=row['mask_uid'], budget=fit['budget'],
                                 supported=fit['supported'], failure=fit['failure_code'],
                                 carried=fit['carried_incumbent'], fresh_supported=fit['fresh_candidate_supported'],
                                 allocation=fit['allocation'], mse=fit.get('retained_normalized_mse'),
                                 certificate_hash=fit.get('certificate', {}).get('coefficient_sha256')))
            print(f"geometry: {len(rows) // 3}/{len(keys)} sources", flush=True)
    save(REPORT / 'geometry.parquet', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('synthetic', 'geometry', 'observations', 'all'), default='all')
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    receipt = protocol()
    if args.stage in ('synthetic', 'all'):
        synthetic(receipt, args.workers)
    if args.stage in ('geometry', 'all'):
        geometry()
    if args.stage in ('observations', 'all'):
        replay()


if __name__ == '__main__':
    main()
