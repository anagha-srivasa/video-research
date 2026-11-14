import numpy as np
import pytest
from track1_fmr.contracts import SpacetimeTubeSegment, EpistemicTag
from track1_fmr.foveal_lens import expected_value
from track1_fmr.junction_graph import clearance, candidates


def tube(track, x):
    c = np.zeros((4, 2)); c[0] = [x, .5]; c[1, 0] = .1
    h = np.zeros((2, 9)); h[0, 0] = .1
    return SpacetimeTubeSegment(track, 0., 2., c, h, np.eye(2), np.eye(4), np.zeros(24), None, None, EpistemicTag.OBSERVED)


def test_tube_derivative_and_proximity_do_not_invent_contact():
    a, b = tube(1, .2), tube(2, .35)
    np.testing.assert_allclose(a.velocity(1.), [.1, 0.])
    assert clearance(a, b, 1.) == pytest.approx(-.05)
    result = candidates(a, b)
    assert len(result) == 1 and result[0].r_phys_prob is None
    with pytest.raises(ValueError): a.evaluate(3.)


def test_information_gain_requires_calibrated_outcomes():
    assert expected_value(.5, [.1, .9], [.5, .5], .1) > 0
    assert expected_value(.5, [.5, .5], [.5, .5], .1) < 0
    with pytest.raises(ValueError): expected_value(.5, [.9], [1.], 0.)


def test_physical_time_collocation_and_jvp():
    import torch
    from track1_fmr.collocation import sample, pushforward
    data = sample(tube(1, .2))
    np.testing.assert_allclose(data['velocity'], np.tile([.1, 0.], (4, 1)))
    x = torch.tensor([2.], dtype=torch.float64)
    z, dz = pushforward(lambda y: y**2, x, torch.ones_like(x) * 3)
    assert z.item() == 4 and dz.item() == 12


def test_lazy_query_never_fabricates_contact_or_semantics():
    from track1_fmr.ventral_query import query
    calls = []
    def verifier(patch, label):
        calls.append(label)
        return .8
    result = query([tube(1, .2)], verifier=verifier, label='mug')
    assert not calls and result['candidates'][0]['semantic_score'] is None
    assert result['candidates'][0]['physical_contact'] is None
