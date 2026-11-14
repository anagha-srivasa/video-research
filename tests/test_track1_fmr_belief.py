import numpy as np
import pytest
from track1_fmr.belief_filter import Belief, advance


def test_occlusion_coasting_keeps_shape_and_increases_uncertainty():
    a = Belief(0., np.array([.4, .5, .1, 0]), np.eye(4), np.diag([.01, .02]), 'OBSERVED')
    b = advance(a, .2, .1)
    assert b.status == 'PREDICTED'
    np.testing.assert_array_equal(b.shape_scatter, a.shape_scatter)
    assert np.trace(b.covariance) > np.trace(a.covariance)
    assert not b.mean.flags.writeable
    with pytest.raises(ValueError):
        advance(b, .2, .1)


def test_directional_noise_reduces_update_on_occluded_axis():
    a = Belief(0., np.zeros(4), np.eye(4), np.eye(2), 'OBSERVED')
    b = advance(a, .1, .1, [1., 1.], np.diag([.01, 100.]))
    assert b.mean[0] > .9 and b.mean[1] < .02
    assert np.linalg.eigvalsh(b.covariance).min() >= 0
