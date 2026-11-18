"""Matched analytic-motion reconstruction versus fixed sliding windows."""
import json
import numpy as np
from pipeline.common import ROOT, atomic_json
from track1_belief.common import save
from track1_observation_revision.benchmark import episodes
from .lebesgue_tubes import TubeBuilder


def fixed_fit(t, position, valid, end):
    ids = np.flatnonzero(valid & (t >= end - 1.) & (t <= end))
    if len(ids) < 4 or t[ids[-1]] - t[ids[0]] < .5:
        return None
    u = 2 * (t[ids] - (end - 1.)) - 1
    return np.polynomial.legendre.legfit(u, position[ids], 3)


def run():
    selection = json.loads((ROOT / 'reports/track1_observation_revision/selection.json').read_text())
    method = min((m for m in selection['methods'] if m.startswith('d')),
                 key=lambda m: selection['methods'][m]['training_objective'])
    index = selection['methods'][method]['index']
    rows = []
    for ep in episodes('validation'):
        with np.load(ROOT / f'derived/track1_observation_revision/{ep["id"]}.npz') as z:
            state, valid = z['means'][index], z['valid'][index]
        t = ep['time']; builder = TubeBuilder(); segments = []
        prior = None
        for i in np.flatnonzero(valid):
            if prior is not None and t[i] - t[prior] > .12:
                segment = builder.close('gap')
                if segment: segments.append(segment)
                prior = None
            innovation = 0.
            if prior is not None:
                residual = ep['y'][i] - (state[prior, :2] + state[prior, 2:] * (t[i] - t[prior]))
                innovation = float(residual @ residual / max(ep['noise_variance'] * 2, .1))
            segment = builder.add(t[i], state[i, :2], innovation, 'OBSERVED')
            if segment: segments.append(segment)
            prior = i
        segment = builder.close()
        if segment: segments.append(segment)
        fixed = []
        for end in np.arange(1., 6.001, .25):
            coef = fixed_fit(t, state[:, :2], valid, end)
            if coef is not None:
                fixed.append((end, coef))
        dynamic_error, fixed_error = [], []
        for i in np.flatnonzero(valid & (t >= 1.)):
            matching = [s for s in segments if s.start <= t[i] <= s.end]
            windows = [(end, coef) for end, coef in fixed if end - 1 <= t[i] <= end]
            if not matching or not windows:
                continue
            a = matching[0].position(t[i])
            end, coef = min(windows, key=lambda pair: pair[0] - t[i])
            b = np.polynomial.legendre.legval(2 * (t[i] - (end - 1)) - 1, coef)
            dynamic_error.append(float(np.sum((a - ep['truth'][i])**2)))
            fixed_error.append(float(np.sum((b - ep['truth'][i])**2)))
        rows.append(dict(episode_id=ep['id'], kind=ep['kind'], corruption=ep['corruption'],
                         dynamic_records=len(segments), fixed_records=len(fixed), matched_frames=len(dynamic_error),
                         eligible_frames=int(np.sum(valid & (t >= 1.))),
                         dynamic_mse=float(np.mean(dynamic_error)) if dynamic_error else None,
                         fixed_mse=float(np.mean(fixed_error)) if fixed_error else None))
    report = ROOT / 'reports/track1_fmr'; report.mkdir(exist_ok=True, parents=True)
    save(report / 'tube_benchmark.parquet', rows)
    good = [r for r in rows if r['matched_frames']]
    summary = dict(scope='60 simulated validation episodes; retrospectively fitted historical segments, no real-video acceptance',
                   selected_state_method=method, episodes=len(rows), episodes_with_matched_frames=len(good),
                   matched_frames=sum(r['matched_frames'] for r in rows), eligible_frames=sum(r['eligible_frames'] for r in rows),
                   dynamic_records=sum(r['dynamic_records'] for r in rows), fixed_records=sum(r['fixed_records'] for r in rows),
                   dynamic_rmse=float(np.sqrt(np.mean([r['dynamic_mse'] for r in good]))) if good else None,
                   fixed_rmse=float(np.sqrt(np.mean([r['fixed_mse'] for r in good]))) if good else None,
                   fixed_windows='one-second window, quarter-second hop; closest closing window',
                   dynamic_parameters='innovation threshold 20, 4 second maximum; break at observation gaps >120ms',
                   target='analytic synthetic position; common matched timestamps',
                   scientific_acceptance=False)
    atomic_json(report / 'tube_benchmark.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
