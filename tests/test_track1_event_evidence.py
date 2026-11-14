import numpy as np
from track1_event_evidence import detect, raw_velocity
from track1_observation_revision.benchmark import episodes


def test_missing_observation_does_not_invent_onset_or_release():
    t = np.arange(100) / 50
    v = np.tile([100., 0.], (len(t), 1))
    valid = np.ones(len(t), bool)
    valid[20] = False
    valid[50:65] = False
    assert detect(t, v, valid) == []


def test_actual_onset_and_release_are_detected_causally():
    t = np.arange(100) / 50
    v = np.zeros((len(t), 2))
    v[30:60, 0] = 100
    result = detect(t, v, np.ones(len(t), bool))
    assert result == [('onset', .64), ('release', 1.24)]
    v[70:] = -500
    assert [r for r in detect(t, v, np.ones(len(t), bool)) if r[1] < 1.4] == result


def test_raw_slope_rejects_future_and_cancels_constant_group_offset():
    ep = next(episodes('train'))
    a, valid = raw_velocity(ep)
    changed = dict(ep, y=ep['y'].copy())
    changed['y'] += np.array([500., -200.])
    b, vb = raw_velocity(changed)
    np.testing.assert_array_equal(valid, vb)
    np.testing.assert_allclose(a, b, atol=1e-8)
    changed['y'][200:] *= 100
    c, _ = raw_velocity(changed)
    np.testing.assert_allclose(a[:200], c[:200], atol=1e-8)
