"""Matched CV/constant-position comparison with one exploratory forward per model."""
import json
import time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import torch
from pipeline.common import ROOT, atomic_json, digest
from track1_retrain.common import save
from track2_feasibility.data import Data
from track2_feasibility.evaluate import prediction_rows
from track2_assessment import assess, ratio
from .model import CFG, baseline, predict
from .fit import DATA, REPORT, load


def evaluate():
    data = Data()
    ids = data.indices('validation', usable=False)
    paths = [DATA/'models'/f'{f}.{s}.pt' for f in CFG['models'] for s in CFG['seeds']]
    lock = dict(models={str(p.relative_to(ROOT)): digest(p) for p in paths},
                protocol_sha256=digest(REPORT/'protocol_lock.json'))
    lock_path = REPORT/'evaluation_lock.json'
    output = REPORT/'predictions.parquet'
    if lock_path.exists():
        stored = json.loads(lock_path.read_text())
        if any(stored.get(k) != v for k, v in lock.items()) or not output.exists() or stored.get('prediction_sha256') != digest(output):
            raise ValueError('Incomplete/changed locked evaluation; refusing automatic validation rerun')
        return
    atomic_json(lock_path, lock)
    history = data.x[ids, :, 140:144]
    rows, latencies = [], []
    speed = np.linalg.norm(data.x[data.indices(), -1, 142:144].astype(float), axis=1)
    boundaries = np.quantile(speed, [.25, .5, .75])
    for family, seed in [('cv', 0), ('constant_position', 0)] + [(f, s) for f in CFG['models'] for s in CFG['seeds']]:
        if family in CFG['models']:
            model, norm, _ = load(DATA/'models'/f'{family}.{seed}.pt')
            result = predict(model, norm, history)
            inference = lambda h: predict(model, norm, h)
        else:
            result = baseline(history, family == 'constant_position')
            inference = lambda h: baseline(h, family == 'constant_position')
        # Reuse the established scalar-error and unavailable-truth schema.
        carrier = np.zeros((len(ids), 3, 144), dtype=np.float64)
        carrier[:, :, :4] = result
        local = prediction_rows(data, ids, carrier, family, seed)
        for row in local:
            row['coarse_squared_error'] = None  # This model does not forecast shape/history.
        rows.extend(local)
        sample = data.x[data.indices()[:1], :, 140:144]
        for _ in range(10):
            inference(sample)
        times = []
        for _ in range(200):
            start = time.perf_counter()
            inference(sample)
            times.append((time.perf_counter()-start)*1000)
        latencies.append(dict(family=family, seed=seed, median_ms=float(np.median(times)),
                              p95_ms=float(np.quantile(times, .95)), repetitions=200))
    sample_by_id = {s['sequence_id']: i for i, s in enumerate(data.samples)}
    for row in rows:
        i = sample_by_id[row['sequence_id']]
        row['input_source_speed_quartile'] = int(np.searchsorted(boundaries, np.linalg.norm(data.x[i, -1, 142:144].astype(float)), side='right'))
    prior = [r for r in pq.read_table(ROOT/'reports/track2_feasibility/predictions.parquet').to_pylist() if r['family'] == 'cv']
    previous = {(r['sequence_id'], r['horizon_us']): r for r in prior}
    for row in [r for r in rows if r['family'] == 'cv']:
        old = previous[row['sequence_id'], row['horizon_us']]
        for field in ('predicted_position', 'predicted_velocity', 'target_position', 'target_velocity', 'target_supported'):
            if row[field] != old[field]:
                raise ValueError('CV baseline changed relative to the original experiment')
    save(output, rows)
    lock['prediction_sha256'] = digest(output)
    atomic_json(lock_path, lock)
    atomic_json(REPORT/'inference_latency.json', dict(records=latencies, device='cpu', batch_size=1,
        all_three_horizons=True, includes_normalization_and_cv_addition=True,
        upstream_rgb_and_track1_excluded=True, mobile_hardware_measured=False))
    atomic_json(REPORT/'strata.json', dict(input_speed_quartile_boundaries=boundaries.tolist(),
        learned_from='training input source velocities only', physical_motion_regimes=False,
        contact_occlusion_identity_labels='unavailable'))


def summarize():
    rows = pq.read_table(REPORT/'predictions.parquet').to_pylist()
    ledger = pq.read_table(ROOT/'reports/track2_feasibility/target_ledger.parquet').to_pylist()
    expected = {('cv', 0), ('constant_position', 0)} | {(f, s) for f in CFG['models'] for s in CFG['seeds']}
    audit, participants, episodes = assess(rows, ledger, expected)
    atomic_json(REPORT/'pairing_audit.json', audit)
    save(REPORT/'episode_metrics.parquet', episodes)
    summary, per = [], []
    for family, seed in sorted(expected):
        for h in CFG['horizons_us']:
            local = []
            for p in CFG['validation_participants']:
                values = [r for r in rows if r['family'] == family and r['seed'] == seed and r['horizon_us'] == h and r['participant'] == p and r['target_supported']]
                if not values:
                    raise ValueError('Empty participant/horizon')
                r = dict(family=family, seed=seed, horizon_us=h, participant=p, targets=len(values))
                for m in ('position', 'velocity'):
                    r[m+'_rmse'] = float(np.sqrt(np.mean([v[m+'_squared_error'] for v in values])))
                local.append(r)
                per.append(r)
            summary.append(dict(family=family, seed=seed, horizon_us=h,
                position_rmse=float(np.mean([r['position_rmse'] for r in local])),
                velocity_rmse=float(np.mean([r['velocity_rmse'] for r in local]))))
    save(REPORT/'participant_metrics.parquet', per)
    decisions = []
    for family in CFG['models']:
        comparisons = []
        for base in CFG['baselines']:
            for h in CFG['horizons_us']:
                b = next(r for r in summary if r['family'] == base and r['horizon_us'] == h)
                values = [r for r in summary if r['family'] == family and r['horizon_us'] == h]
                pos = float(np.median([r['position_rmse'] for r in values]))
                vel = float(np.median([r['velocity_rmse'] for r in values]))
                max_seed = max(r['position_rmse'] for r in values)
                limit = CFG['gates']['median_position_ratio_250_500_vs_each_baseline'] if h < 1000000 else CFG['gates']['median_position_1000_max_ratio_vs_each_baseline']
                comparisons.append(dict(baseline=base, horizon_us=h, median_position_ratio=ratio(pos, b['position_rmse']),
                    median_velocity_ratio=ratio(vel, b['velocity_rmse']),
                    worst_seed_position_ratio=ratio(max_seed, b['position_rmse']),
                    position_pass=bool(pos <= limit*b['position_rmse']),
                    velocity_pass=bool(vel <= CFG['gates']['median_velocity_max_ratio_vs_each_baseline']*b['velocity_rmse']),
                    seed_pass=bool(h == 1000000 or max_seed <= CFG['gates']['worst_seed_position_ratio_250_500_vs_each_baseline']*b['position_rmse'])))
        decisions.append(dict(family=family, comparisons=comparisons,
            proxy_gate_passed=all(c['position_pass'] and c['velocity_pass'] and c['seed_pass'] for c in comparisons)))
    strata = defaultdict(list)
    for r in rows:
        if r['target_supported']:
            strata[r['family'], r['seed'], r['horizon_us'], r['participant'], r['input_source_speed_quartile']].append(r)
    save(REPORT/'speed_strata.parquet', [dict(family=k[0], seed=k[1], horizon_us=k[2], participant=k[3], input_source_speed_quartile=k[4], targets=len(v),
        position_rmse=float(np.sqrt(np.mean([r['position_squared_error'] for r in v]))),
        velocity_rmse=float(np.sqrt(np.mean([r['velocity_squared_error'] for r in v])))) for k,v in sorted(strata.items())])
    atomic_json(REPORT/'metrics.json', dict(records=summary, units='intrinsic-normalized source positions and source velocity per second',
        aggregation='participant RMSE then participant mean, seed median', independent_physical_reference=False))
    atomic_json(REPORT/'decision.json', dict(models=decisions,
        any_proxy_gate_passed=any(d['proxy_gate_passed'] for d in decisions),
        validation_previously_exposed=True, independent_validation_participants=2,
        scientific_acceptance=False, full_track2_go=False, automatic_followup_training=False,
        event_contact_identity_calibration='unavailable independent references',
        interpretation='New kinematic parameterization/loss/input changes prevent causal attribution of differences to functional features alone'))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = json.loads((REPORT/'metrics.json').read_text())['records']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    cells = []
    for family in CFG['baselines']+CFG['models']:
        for h in CFG['horizons_us']:
            values = [r for r in metrics if r['family'] == family and r['horizon_us'] == h]
            cells.append(f"<tr><td>{family}</td><td>{h//1000}</td><td>{np.median([r['position_rmse'] for r in values]):.5f}</td><td>{np.median([r['velocity_rmse'] for r in values]):.5f}</td></tr>")
        for ax, metric in zip(axes, ['position', 'velocity']):
            ax.plot([h/1000 for h in CFG['horizons_us']], [np.median([r[metric+'_rmse'] for r in metrics if r['family']==family and r['horizon_us']==h]) for h in CFG['horizons_us']], 'o-', label=family)
            ax.set(xlabel='Horizon (ms)', ylabel=f'Source {metric} RMSE', xticks=[250, 500, 1000])
            ax.legend(fontsize=8)
    fig.savefig(REPORT/'quality.png', dpi=160)
    plt.close(fig)
    decision = json.loads((REPORT/'decision.json').read_text())
    status = 'Proxy gates passed; independent scientific gates remain unavailable.' if decision['any_proxy_gate_passed'] else 'Proxy gates failed. Full Track 2 remains on hold.'
    (REPORT/'dashboard.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Kinematics-only comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.5}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #aaa}</style><h1>Kinematics-only comparison</h1><p>'''+status+'''</p><p>Six fixed final runs, five training-participant stopping folds per run. Previously exposed P08/P18 validation; frozen six-second contexts. Four past position/velocity states; no functional coefficients, geometry, appearance or uncertainty inputs. Zero-initialized CV correction with translation-relative positions.</p><img src="quality.png" alt="Matched source-position and source-velocity errors versus CV and constant position"><table><tr><th>Family</th><th>Horizon ms</th><th>Position RMSE</th><th>Velocity RMSE</th></tr>'''+''.join(cells)+'''</table><p>Two validation participants and five source contexts; histories overlap. Targets are retrospective source kinematics. Constant position equals nearest-state continuation for position; its velocity is zero. It is not future interpolation or a nearest raw-observation estimator. Independent event/contact/identity and forecast uncertainty metrics remain unavailable. This study changes parameterization and loss as well as inputs; it cannot attribute gains solely to removing functional features.</p><p><a href="RESULTS.md">Protocol, results and limits</a> · <a href="decision.json">Gates</a> · <a href="pairing_audit.json">Coverage and matching</a> · <a href="episode_metrics.parquet">Episodes</a> · <a href="speed_strata.parquet">Source-speed diagnostics</a> · <a href="inference_latency.json">Cached-state latency</a> · <a href="verification.json">Verification</a></p></html>''')
