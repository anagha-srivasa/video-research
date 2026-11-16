import cv2
import numpy as np
import pytest
from track1_fmr.dorsal_flux import (build_local_super_gaussian_prior, explained_density,
                                    TrackPriorMask, extract_proposal_lab_moments)


def test_local_super_gaussian_matches_dense_formula_and_stays_local():
    centre = np.array([.4, .6]); shape = np.array([[.0004, .00012], [.00012, .0009]])
    ys, xs, patch = build_local_super_gaussian_prior(centre, shape, 160, 320)
    assert patch.size < .1 * 160 * 320
    yy, xx = np.mgrid[:160, :320]
    delta = np.stack([(xx + .5) / 320, (yy + .5) / 160], axis=-1) - centre
    inverse = np.linalg.inv(shape.astype(np.float32) + np.eye(2, dtype=np.float32) * 1e-8)
    distance = np.einsum('...i,ij,...j->...', delta, inverse, delta)
    dense = np.exp(-distance**2 / 32)
    np.testing.assert_allclose(patch, dense[ys, xs], atol=3e-6)
    # Suppression stays local, including when local patch and global coordinates differ.
    flow = np.zeros((160, 320, 2), np.float32); flow[..., 0] = .02
    local = TrackPriorMask(0, patch, (.02, 0.), (ys.start, xs.start), (160, 320))
    explained = explained_density(flow, np.zeros_like(flow),
        [dict(density=local.density, displacement=local.vel_norm, slices=local.slices)])
    np.testing.assert_allclose(explained[ys, xs], patch)
    assert explained[:ys.start].sum() == 0
    assert np.exp(-1.4**4 / 32) > .886 and np.exp(-2**4 / 32) == pytest.approx(.60653066)
    assert np.exp(-1.4**2 / 2) < .65  # The old Gaussian's leaking skirt.
    empty_y, empty_x, empty = build_local_super_gaussian_prior((-10., 2.), shape, 160, 160)
    assert empty.size == 0
    with pytest.raises(ValueError):
        build_local_super_gaussian_prior(centre, np.diag([-1., 1.]), 160, 160)


def test_lab_sampling_is_bounded_and_does_not_invent_background(monkeypatch):
    image = np.zeros((1080, 1920, 3), np.uint8); image[:] = [15, 230, 20]
    mask = np.zeros((160, 160), bool); mask[20:140, 50:100] = True
    image[135:945, 600:1200] = [220, 20, 10]
    original = cv2.cvtColor; calls = []
    def inspect(tile, conversion):
        assert tile.shape[0] <= 32 and tile.shape[1] <= 32
        calls.append(tile.shape)
        return original(tile, conversion)
    monkeypatch.setattr(cv2, 'cvtColor', inspect)
    moments = extract_proposal_lab_moments(image, (50/160, 20/160, 100/160, 140/160), mask)
    expected = original(np.array([[[220, 20, 10]]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0]
    np.testing.assert_allclose(moments[:3], expected)
    np.testing.assert_allclose(moments[3:], 0)
    assert calls == [(32, 32, 3)]
    with pytest.raises(ValueError):
        extract_proposal_lab_moments(image, (0., 0., 1., 1.), np.zeros((160,160), bool))
