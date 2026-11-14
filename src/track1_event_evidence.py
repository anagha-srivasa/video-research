"""Separate causal event evidence with explicit unknown phases after long gaps."""
from pathlib import Path
import json
import numpy as np
from pipeline.common import ROOT, atomic_json, digest
from track1_belief.common import save
from track1_observation_revision.benchmark import episodes, score_events


def detect(times, velocities, valid, speed=20., dwell=.04, maximum_gap=.15):
    """Missing data is not evidence for quiet or for a new movement onset.

    Retain established phase over short gaps but never accumulate dwell through
    a missing sample. Long gaps make phase unknown; the first subsequent known
    phase initializes silently. References during that gap stay in the recall
    denominator. This detects projected motion, not contact or physical intent.
    """
    times, velocities, valid = np.asarray(times), np.asarray(velocities), np.asarray(valid, bool)
    if (times.ndim != 1 or velocities.shape != (len(times), 2) or valid.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(velocities[valid]).all()
            or speed <= 0 or dwell < 0 or maximum_gap <= 0):
        raise ValueError('Finite ordered event evidence and valid thresholds required')
    events = []
    moving, direction, last, released = None, None, None, None
    pending, since = None, None
    for t, v, ok in zip(times, velocities, valid):
        if not ok:
            pending, since = None, None
            continue
        if last is not None and t - last > maximum_gap:
            moving, direction, released = None, None, None
            pending, since = None, None
        last = t
        magnitude = float(np.linalg.norm(v))
        if moving is None:
            if magnitude >= speed:
                moving, direction = True, v / magnitude
            elif magnitude < speed / 2:
                moving = False
            continue
        event = None
        if not moving and magnitude >= speed:
            event = ('reversal' if direction is not None and released is not None
                     and t - released <= .5 and v @ direction < -.5 * magnitude else 'onset')
        elif moving and magnitude < speed / 2:
            event = 'release'
        elif moving and magnitude >= speed and direction is not None and v @ direction < -.5 * magnitude:
            event = 'reversal'
        if event != pending:
            pending, since = event, t
        if event and t - since >= dwell - 1e-10:
            events.append((event, float(t)))
            moving = event != 'release'
            if moving:
                direction, released = v / magnitude, None
            else:
                released = t
            pending, since = None, None
    return events


def raw_velocity(ep, window=.08):
    """Short local slope from raw positions, independent of the position filter.

    No fit crosses a declared observation-reset group. The intercept eliminates
    a constant within-group displacement bias. No learned jump or contact label
    is inferred. Four robust reweighting steps bound isolated temporal outliers.
    """
    t, y = ep['time'], ep['y']
    velocities = np.zeros_like(y)
    valid = np.zeros(len(t), bool)
    for i in np.flatnonzero(ep['valid']):
        ids = np.flatnonzero(ep['valid'] & (t <= t[i]) & (t >= t[i] - window - 1e-10)
                            & (ep['groups'] == ep['groups'][i]))
        if len(ids) < 3 or np.ptp(t[ids]) < .04 - 1e-10:
            continue
        design = np.c_[np.ones(len(ids)), t[ids] - t[i]]
        weight = np.ones(len(ids))
        beta = None
        for _ in range(4):
            beta = np.linalg.lstsq(design * np.sqrt(weight[:, None]), y[ids] * np.sqrt(weight[:, None]), rcond=None)[0]
            residual = np.linalg.norm(y[ids] - design @ beta, axis=1)
            scale = max(np.sqrt(ep['noise_variance']), float(np.median(residual)) / 1.177, .1)
            weight = np.minimum(1., 2.5 * scale / np.maximum(residual, 1e-12))
        velocities[i], valid[i] = beta[1], True
    return velocities, valid


def main():
    report = ROOT / 'reports/track1_event_evidence'
    report.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / 'reports/track1_observation_revision/selection.json').read_text())
    primary = min((m for m in frozen['methods'] if m.startswith('d')), key=lambda m: frozen['methods'][m]['training_objective'])
    methods = ('nearest', 'cv', primary, 'raw_short_slope')
    protocol = dict(source_sha256=digest(Path(__file__)), primary=primary, speed_grid=[20., 50., 100.],
                    fixed_raw_window=.08, fixed_dwell=.04, maximum_gap=.15,
                    frozen_state_selection_sha256=digest(ROOT / 'reports/track1_observation_revision/selection.json'),
                    scope='Exploratory synthetic event-evidence correction; no independent physical event labels',
                    selection='training only, same three threshold trials for every evidence source', promoted=False,
                    input_hashes={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'derived/track1_observation_revision').glob('*.npz'))})
    path = report / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Event protocol changed')
    atomic_json(path, protocol)
    rows = []
    selected = None
    for split in ('train', 'validation'):
        part = []
        for ep in episodes(split):
            with np.load(ROOT / f'derived/track1_observation_revision/{ep["id"]}.npz', allow_pickle=False) as z:
                evidence = {m: (z['means'][frozen['methods'][m]['index'], :, 2:].copy(),
                                z['valid'][frozen['methods'][m]['index']].copy()) for m in methods[:-1]}
            evidence['raw_short_slope'] = raw_velocity(ep)
            for method, (velocity, valid) in evidence.items():
                for speed in protocol['speed_grid'] if split == 'train' else sorted({20., selected[method]}):
                    pred = detect(ep['time'], velocity, valid, speed)
                    part.append(dict(episode_id=ep['id'], split=split, kind=ep['kind'], corruption=ep['corruption'],
                                     method=method, speed=speed, valid_frames=int(valid.sum()),
                                     **score_events(ep, pred)))
        if split == 'train':
            selected = {}
            for method in methods:
                def objective(speed):
                    group = [r for r in part if r['method'] == method and r['speed'] == speed]
                    hits = sum(r['matched_events'] for r in group)
                    references = sum(r['reference_events'] for r in group)
                    false = sum(r['false_alarms'] for r in group)
                    return -2 * hits / max(references + hits + false, 1), false, speed
                selected[method] = min(protocol['speed_grid'], key=objective)
            atomic_json(report / 'selection.json', selected)
        rows.extend(part)
    save(report / 'episodes.parquet', rows)
    result = []
    for method in methods:
        for head, speed in [('common20', 20.), ('train_selected', selected[method])]:
            group = [r for r in rows if r['split'] == 'validation' and r['method'] == method and r['speed'] == speed]
            hits, references, false, quiet, seconds = (sum(r[k] for r in group) for k in
                ('matched_events', 'reference_events', 'false_alarms', 'quiet_false_alarms', 'quiet_seconds'))
            delays = [d for r in group for d in r['delay_seconds']]
            result.append(dict(method=method, head=head, speed=speed, matched=hits, references=references,
                               recall=hits / references, false_alarms=false, quiet_false_alarms=quiet,
                               quiet_seconds=seconds, quiet_far_per_minute=quiet * 60 / seconds,
                               median_delay=float(np.median(delays)) if delays else None))
    atomic_json(report / 'results.json', dict(validation=result, selected=selected, promoted=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
