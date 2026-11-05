"""Post-hoc report assessment from locked forecasts; never trains or runs inference."""
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.common import ROOT, atomic_json, digest

REPORT = ROOT / 'reports/track2_report_assessment'
SOURCE = ROOT / 'reports/track2_feasibility'
HORIZONS = (250000, 500000, 1000000)
METRICS = ('position', 'velocity')


def key(row):
    return row['sequence_id'], row['horizon_us']


def rmse(rows, metric):
    return math.sqrt(statistics.mean(r[metric + '_squared_error'] for r in rows))


def ratio(value, baseline):
    # A zero baseline is not replaced by an arbitrary epsilon or a zero ratio.
    return value / baseline if baseline > 0 else (1.0 if value == 0 else None)


def within(value, baseline):
    return value <= 1.05 * baseline


def assess(rows, ledger, expected_groups):
    """Validate exact pairing, then summarize position/velocity noninferiority.

    Five-percent margins and the 80% episode diagnostic are post-hoc report
    criteria. They cannot replace the original experiment's locked decision.
    """
    opportunities = {}
    for row in ledger:
        if row['split'] != 'validation':
            continue
        if key(row) in opportunities:
            raise ValueError('Duplicate opportunity')
        opportunities[key(row)] = row
    if not opportunities or {k[1] for k in opportunities} != set(HORIZONS):
        raise ValueError('Missing validation horizon ledger')
    expected = {k for k, r in opportunities.items() if r['input_supported']}
    grouped = defaultdict(dict)
    for row in rows:
        group = row['family'], row['seed']
        k = key(row)
        if group not in expected_groups or k not in expected or k in grouped[group]:
            raise ValueError('Unexpected or duplicate forecast')
        source = opportunities[k]
        for field in ('participant', 'video_id', 'episode_id', 'input_cutoff_us',
                      'split', 'target_supported'):
            if row[field] != source[field]:
                raise ValueError('Forecast/ledger mismatch: ' + field)
        if row['physical_reference'] or row['coordinate_mode'] != 'intrinsic_normalized_2D':
            raise ValueError('Unexpected reference or coordinate contract')
        for metric in METRICS:
            prediction = row['predicted_' + metric]
            target = row['target_' + metric]
            error = row[metric + '_squared_error']
            if len(prediction) != 2 or not all(math.isfinite(v) for v in prediction):
                raise ValueError('Invalid prediction')
            if not row['target_supported']:
                if target is not None or error is not None:
                    raise ValueError('Unavailable truth must remain null')
            else:
                if target is None or len(target) != 2 or not all(math.isfinite(v) for v in target):
                    raise ValueError('Invalid target')
                calculated = sum((a-b)**2 for a, b in zip(prediction, target))
                if error is None or not math.isfinite(error) or not math.isclose(
                        error, calculated, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError('Stored error differs from forecast and target')
        grouped[group][k] = row
    if set(grouped) != set(expected_groups) or any(set(g) != expected for g in grouped.values()):
        raise ValueError('Missing model or silently excluded forecast')
    baseline = grouped['cv', 0]
    for group in grouped.values():
        for k, row in group.items():
            if any(row['target_' + m] != baseline[k]['target_' + m] for m in METRICS):
                raise ValueError('Models do not share identical targets')

    coverage = []
    participants = sorted({r['participant'] for r in opportunities.values()})
    for p in participants:
        for h in HORIZONS:
            local = [r for r in opportunities.values() if r['participant'] == p and r['horizon_us'] == h]
            reasons = defaultdict(int)
            for r in local:
                if not r['target_supported']:
                    reasons[r['reason'] or 'unspecified'] += 1
            coverage.append(dict(participant=p, horizon_us=h, proposed=len(local),
                input_supported=sum(r['input_supported'] for r in local),
                target_supported=sum(r['target_supported'] for r in local),
                missing_reasons=dict(reasons)))

    participant_rows, episode_rows, summaries = [], [], []
    all_episodes = sorted({(r['participant'], r['video_id'], r['episode_id']) for r in opportunities.values()})
    for (family, seed), group in sorted(grouped.items()):
        if family == 'cv':
            continue
        for p in participants:
            for h in HORIZONS:
                selected = [k for k, r in group.items() if r['participant'] == p and k[1] == h and r['target_supported']]
                if not selected:
                    raise ValueError('No matched participant/horizon support')
                result = dict(family=family, seed=seed, participant=p, horizon_us=h, targets=len(selected))
                for m in METRICS:
                    value = rmse([group[k] for k in selected], m)
                    base = rmse([baseline[k] for k in selected], m)
                    result.update({m+'_rmse': value, 'cv_'+m+'_rmse': base,
                                   m+'_ratio': ratio(value, base), m+'_within_5pct': within(value, base)})
                participant_rows.append(result)
        for p, video, episode in all_episodes:
            passed, supported = True, True
            result = dict(family=family, seed=seed, participant=p, video_id=video, episode_id=episode)
            for h in HORIZONS[:2]:
                selected = [k for k, r in group.items() if (r['participant'], r['video_id'], r['episode_id']) == (p, video, episode)
                            and k[1] == h and r['target_supported']]
                result[f'targets_{h}'] = len(selected)
                supported &= bool(selected)
                for m in METRICS:
                    value = rmse([group[k] for k in selected], m) if selected else None
                    base = rmse([baseline[k] for k in selected], m) if selected else None
                    result[f'{m}_rmse_{h}'] = value
                    result[f'cv_{m}_rmse_{h}'] = base
                    passed &= bool(selected) and within(value, base)
            result.update(scored=supported, both_horizons_position_velocity_within_5pct=passed if supported else None)
            episode_rows.append(result)
        local = [r for r in participant_rows if (r['family'], r['seed']) == (family, seed)]
        aggregate = []
        for h in HORIZONS:
            values = [r for r in local if r['horizon_us'] == h]
            out = dict(horizon_us=h)
            for m in METRICS:
                value = statistics.mean(r[m+'_rmse'] for r in values)
                base = statistics.mean(r['cv_'+m+'_rmse'] for r in values)
                out.update({m+'_ratio': ratio(value, base), m+'_within_5pct': within(value, base)})
            aggregate.append(out)
        episodes = [r for r in episode_rows if (r['family'], r['seed']) == (family, seed)]
        scored = sum(r['scored'] for r in episodes)
        passed = sum(r['both_horizons_position_velocity_within_5pct'] is True for r in episodes)
        # Every required metric must be present. Missing event/contact/reference
        # evidence cannot turn a successful proxy comparison into scientific GO.
        summaries.append(dict(family=family, seed=seed, horizons=aggregate,
            proxy_noninferiority_250_500=all(r[m+'_within_5pct'] for r in aggregate[:2] for m in METRICS),
            proposed_episodes=len(episodes), scored_episodes=scored, unscored_episodes=len(episodes)-scored,
            passing_episodes=passed, conditional_episode_pass_fraction=passed/scored if scored else None,
            demonstrated_pass_fraction_all_episodes=passed/len(episodes),
            episode_80pct_proxy_check=(passed/scored >= .8) if scored else False,
            event_gate=None, contact_gate=None, identity_gate=None, uncertainty_gate=None,
            full_pareto_assessment='unavailable_independent_references', scientific_go=False))
    return dict(post_hoc=True, used_for_selection=False, new_model_inference=False,
        coordinate_mode='intrinsic_normalized_2D', physical_reference=False,
        participants=participants, episode_definition='RGB source context grouped by participant/video/episode_id; overlapping histories aggregated',
        segment_definition_is_post_hoc=True, significance_tests_performed=False,
        significance_reason='Only two iteratively used validation participants; overlapping windows are not independent replicates',
        coverage=coverage, runs=summaries, full_track2_go=False), participant_rows, episode_rows


def main():
    from track2_feasibility.data import check_frozen
    from track1_encoder_revision.references import audit
    check_frozen()
    config = json.loads((ROOT/'configs/track2_feasibility.json').read_text())
    expected = {('cv', 0)} | {(f, s) for f in config['models'] for s in config['seeds']}
    paths = [SOURCE / name for name in ('predictions.parquet', 'target_ledger.parquet', 'decision.json', 'evaluation_lock.json')]
    manifest_path = ROOT / 'derived/track2_feasibility/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, value in manifest['source'].items():
        if digest(ROOT/name) != value:
            raise ValueError('Locked training source changed: ' + name)
    for name, value in manifest['outputs'].items():
        if digest(manifest_path.parent/name) != value:
            raise ValueError('Locked model/adapter changed: ' + name)
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths + [manifest_path]}
    result, participants, episodes = assess(pq.read_table(paths[0]).to_pylist(), pq.read_table(paths[1]).to_pylist(), expected)
    references = audit()
    result.update(original_locked_decision=json.loads(paths[2].read_text()),
                  accepted_independent_references=references['accepted_usable'],
                  input_hashes=before, assessment_code_sha256=digest(Path(__file__)))
    if any(digest(ROOT/name) != value for name, value in before.items()):
        raise ValueError('Input changed during audit')
    check_frozen()
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT/'assessment.json', result)
    atomic_json(REPORT/'independent_references.json', references)
    pq.write_table(pa.Table.from_pylist(participants), REPORT/'participant_metrics.parquet', compression='NONE')
    pq.write_table(pa.Table.from_pylist(episodes), REPORT/'episode_metrics.parquet', compression='NONE')
    print(json.dumps(dict(runs=len(result['runs']), participants=len(result['participants']),
        episodes=result['runs'][0]['proposed_episodes'],
        proxy_passes=sum(r['proxy_noninferiority_250_500'] for r in result['runs']),
        independent_references=references['accepted_usable'], full_track2_go=False)))


if __name__ == '__main__':
    main()
