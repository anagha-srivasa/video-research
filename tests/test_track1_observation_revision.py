import numpy as np
import pytest
from track1_observation_revision.benchmark import (
    episodes, estimate, event_head, score_events, continuation, ARMS)
from track1_observation_revision.geometry import fit_budget_path, split_boundary_vertices
from track1_belief.geometry import geometry_curves


def test_factorial_and_effective_corruptions():
    train, validation = list(episodes('train')), list(episodes('validation'))
    assert len(train) == len(validation) == 60
    assert len({(e['kind'], e['corruption'], e['duration']) for e in train}) == 60
    for a, b in zip(train, validation):
        assert a['onset'] != b['onset']
        assert not np.array_equal(a['y'], b['y'])
    hidden = next(e for e in train if e['corruption'] == 'occlusion')
    assert (~hidden['valid']).sum() > 15
    outlier = next(e for e in train if e['corruption'] == 'outlier')
    assert np.max(np.linalg.norm(outlier['y'] - outlier['truth'], axis=1)) > 50


@pytest.mark.parametrize('method', ('nearest', 'cv', *ARMS))
def test_future_and_truth_do_not_change_estimator(method):
    ep = next(episodes('train'))
    ep = {k: v[:60].copy() if isinstance(v, np.ndarray) else v for k, v in ep.items()}
    parameter = .2 if method in ('nearest', 'cv') else 10000.
    a, va = estimate(ep, method, parameter)
    altered = dict(ep, y=ep['y'].copy(), truth=ep['truth'] * 900, velocity=ep['velocity'] + 1e5,
                   corruption='not_disclosed', kind='not_disclosed')
    altered['y'][40:] += 10000
    b, vb = estimate(altered, method, parameter)
    np.testing.assert_array_equal(va[:40], vb[:40])
    np.testing.assert_allclose(a[:40], b[:40], rtol=0, atol=0)
    assert np.isfinite(a).all()


def test_common_events_detect_reversal_and_count_moving_false_alarms():
    ep = next(e for e in episodes('train') if e['kind'] == 'reversal' and e['corruption'] == 'clean')
    events = event_head(ep['time'], ep['velocity'], np.ones(len(ep['time']), bool), dwell=0.)
    assert any(k == 'reversal' for k, _ in events)
    score = score_events(ep, events + [('onset', 5.9)])
    assert score['matched_events'] == score['reference_events']
    assert score['false_alarms'] == 1
    assert score['quiet_false_alarms'] == 0
    # Future velocity changes cannot change previous event decisions.
    changed = ep['velocity'].copy()
    changed[150:] *= -100
    before = event_head(ep['time'], changed, np.ones(len(changed), bool))
    original = event_head(ep['time'], ep['velocity'], np.ones(len(changed), bool))
    assert [x for x in before if x[1] < 3.] == [x for x in original if x[1] < 3.]


def test_damped_continuation_limits():
    mean = np.array([1., 2., 3., 4.])
    np.testing.assert_allclose(continuation(mean, .5, np.inf), [2.5, 4.])
    np.testing.assert_allclose(continuation(mean, 0, .2), mean[:2])
    np.testing.assert_allclose(continuation(mean, .5, 1e9), continuation(mean, .5, np.inf))
    with pytest.raises(ValueError):
        continuation(mean, -.1, .2)


def test_geometry_retains_certified_incumbent(monkeypatch):
    import track1_observation_revision.geometry as module
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = np.c_[np.cos(angle), np.sin(angle)] * 30
    real = module.fit_adaptive_geometry
    def bounded(*args, max_total_controls, **kwargs):
        if max_total_controls > 16:
            return dict(supported=False, failure_code='search_exhausted')
        return real(*args, max_total_controls=max_total_controls, **kwargs)
    monkeypatch.setattr(module, 'fit_adaptive_geometry', bounded)
    fits = fit_budget_path([ring], ['outer'], [-1])
    assert all(f['supported'] for f in fits)
    assert [f['carried_incumbent'] for f in fits] == [False, True, True]
    np.testing.assert_array_equal(fits[0]['coefficients'], fits[-1]['coefficients'])
    geometry_curves(fits[-1])
    with pytest.raises(ValueError):
        split_boundary_vertices([ring[:12]], ['outer'], [-1])


def test_many_rings_record_budget_failure_without_search_exception():
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rings = [np.c_[np.cos(angle) + 4 * i, np.sin(angle)] for i in range(8)]
    fits = fit_budget_path(rings, ['outer'] * 8, [-1] * 8)
    assert not any(f['supported'] for f in fits)
    assert all(f['failure_code'] == 'all_source_rings_exceed_control_budget' for f in fits)


def test_transport_chain_translation_and_future_independence():
    from track1_observation_revision.observations import transport
    rng = np.random.default_rng(10)
    origin = rng.normal(size=(12, 2)) * 20
    points = origin[None] + np.arange(10)[:, None, None] * np.array([2., -1.])
    visible = np.ones((10, 12), bool)
    times = np.arange(10) * 20000
    result = transport(points, visible, origin, np.zeros(2), times, np.ones(10, bool))
    np.testing.assert_allclose(result['position'], np.arange(10)[:, None] * [2., -1.], atol=1e-10)
    assert result['valid'].all()
    points[6:] += rng.normal(size=points[6:].shape) * 100
    changed = transport(points, visible, origin, np.zeros(2), times, np.ones(10, bool))
    np.testing.assert_array_equal(changed['position'][:6], result['position'][:6])
