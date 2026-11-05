import copy
import pytest
from track2_assessment import assess, ratio


def fixture():
    ledger, rows = [], []
    for episode in ('supported', 'missing'):
        for h in (250000, 500000, 1000000):
            valid = episode == 'supported'
            entry = dict(sequence_id=episode, horizon_us=h, split='validation',
                participant='P08', video_id='P08_09', episode_id=episode,
                input_cutoff_us=1000000, input_supported=True,
                target_supported=valid, reason=None if valid else 'missing_future_state')
            ledger.append(entry)
            for family, seed, error in [('cv', 0, 1.0), ('candidate', 17, .5)]:
                row = dict(entry, family=family, seed=seed, physical_reference=False,
                           coordinate_mode='intrinsic_normalized_2D')
                for metric in ('position', 'velocity'):
                    row['predicted_'+metric] = [error, 0.]
                    row['target_'+metric] = [0., 0.] if valid else None
                    row[metric+'_squared_error'] = error**2 if valid else None
                rows.append(row)
    return rows, ledger, {('cv', 0), ('candidate', 17)}


def test_episode_denominators_and_missing_references_cannot_pass():
    result, _, episodes = assess(*fixture())
    run = result['runs'][0]
    assert run['proxy_noninferiority_250_500']
    assert run['conditional_episode_pass_fraction'] == 1
    assert run['demonstrated_pass_fraction_all_episodes'] == .5
    assert run['unscored_episodes'] == 1
    assert run['event_gate'] is None and not run['scientific_go']
    assert next(r for r in episodes if not r['scored'])['both_horizons_position_velocity_within_5pct'] is None


@pytest.mark.parametrize('change', ['drop', 'duplicate', 'target', 'zero_imputation', 'error'])
def test_rejects_unmatched_or_corrupted_evidence(change):
    rows, ledger, groups = fixture()
    if change == 'drop':
        rows.pop()
    elif change == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif change == 'target':
        rows[1]['target_position'] = [1., 0.]
        # .5 has the same squared error relative to 0 and 1: only the
        # cross-model target equality guard detects this mismatch.
    elif change == 'zero_imputation':
        rows[-1]['target_position'] = [0., 0.]
    else:
        rows[1]['position_squared_error'] = .1
    with pytest.raises(ValueError):
        assess(rows, ledger, groups)


def test_no_epsilon_improvement_against_perfect_baseline():
    assert ratio(0., 0.) == 1.
    assert ratio(1., 0.) is None
    rows, ledger, groups = fixture()
    for row in rows:
        if row['family'] == 'cv' and row['target_supported']:
            for m in ('position', 'velocity'):
                row['predicted_'+m] = [0., 0.]
                row[m+'_squared_error'] = 0.
    result, _, _ = assess(rows, ledger, groups)
    assert not result['runs'][0]['proxy_noninferiority_250_500']
