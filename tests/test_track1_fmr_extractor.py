import numpy as np
import pytest
from types import SimpleNamespace
from track1_fmr.dorsal_flux import DorsalFluxExtractor, TrackPriorMask
from track1_fmr.pipeline import Pipeline


def test_stateful_dis_cache_reset_and_prior_validation():
    rgb = np.random.default_rng(7).integers(0, 255, (320, 640, 3), dtype=np.uint8)
    extractor = DorsalFluxExtractor(160)
    initial, color = extractor.step(rgb)
    assert not initial.proposals and initial.background_mode == 'initialization'
    assert color is None
    extractor.step(rgb)
    cached = extractor._dis
    extractor.step(rgb)
    assert extractor._dis is cached and cached.getFinestScale() == 0
    extractor.reset()
    after, _ = extractor.step(rgb)
    assert not after.proposals and after.background_mode == 'initialization'
    with pytest.raises(ValueError):
        TrackPriorMask(0, np.array([[np.nan]]), (0., 0.))
    with pytest.raises(ValueError):
        DorsalFluxExtractor(sigma_flow=0)
    with pytest.raises(ValueError):
        DorsalFluxExtractor(max_proposals=0)


def test_flow_birth_velocity_uses_exact_dt_and_image_coordinate_frame(monkeypatch):
    pipeline = Pipeline(dorsal_grid=160, heartbeat=lambda rgb: [])
    rgb = np.random.default_rng(9).integers(0, 255, (160, 160, 3), dtype=np.uint8)
    pipeline.step(rgb, 10.)
    flow = np.zeros((160, 160, 2), np.float32)
    flow[..., 0] = .01  # camera motion
    flow[50:100, 40:90, 0] = .03  # .02 residual + .01 camera
    monkeypatch.setattr(pipeline.extractor, '_ensure_dis', lambda h, w: SimpleNamespace(
        calc=lambda previous, current, initial: flow * np.array([160, 160], np.float32)))
    background = np.zeros_like(flow); background[..., 0] = .01
    monkeypatch.setattr(pipeline.extractor, '_estimate_ego_residual',
                        lambda observed, known: (observed - background, 24, background, 'affine'))
    pipeline.step(rgb, 10.02)
    assert len(pipeline.tracker.tracks) == 1
    state = pipeline.tracker.tracks[0]
    np.testing.assert_allclose(state.mean[2:], [1.5, 0.], atol=1e-6)
    np.testing.assert_allclose(state.covariance[2:, 2:], np.eye(2) * (.005 / .02)**2)
    assert state.status == 'OBSERVED'
    # Reject bad timestamps before mutating extractor/tracker buffers.
    previous = pipeline.extractor._prev_gray_grid
    with pytest.raises(ValueError):
        pipeline.step(rgb, 10.02)
    assert pipeline.extractor._prev_gray_grid is previous


def test_soft_prior_onset_and_matching_motion_are_distinguished(monkeypatch):
    extractor = DorsalFluxExtractor(None)
    rgb = np.zeros((160, 160, 3), np.uint8)
    flow = np.zeros((160, 160, 2), np.float32)
    flow[50:100, 40:90, 0] = .02
    density = np.zeros((160, 160), np.float32); density[50:100, 40:90] = 1
    monkeypatch.setattr(extractor, '_ensure_dis', lambda h, w: SimpleNamespace(
        calc=lambda previous, current, initial: flow * np.array([160, 160], np.float32)))
    monkeypatch.setattr(extractor, '_estimate_ego_residual',
                        lambda observed, known: (observed, 24, np.zeros_like(flow), 'affine'))
    extractor.step(rgb)
    onset, _ = extractor.step(rgb, [TrackPriorMask(0, density, (0., 0.))])
    assert onset.proposals and onset.emitted_union_mask[75, 65]
    matched, _ = extractor.step(rgb, [TrackPriorMask(0, density, (.02, 0.))])
    assert matched.explained_density[75, 65] > .65
    assert not matched.emitted_union_mask[75, 65]
