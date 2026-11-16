import cv2
import numpy as np
import pytest
from track1_fmr.dorsal_flux import DorsalFluxExtractor, build_local_super_gaussian_prior
from track1_fmr.pipeline import Pipeline
from track1_fmr.belief_filter import Belief


def test_tight_background_core_keeps_wider_suppression_annulus_eligible(monkeypatch):
    extractor = DorsalFluxExtractor(160)
    ys, xs, patch = build_local_super_gaussian_prior((.5, .5), np.diag([.04, .04]), 160, 160)
    known = np.zeros((160,160), np.float32); known[ys,xs] = patch
    assert np.exp(-1.25**4 / 32) > .92
    assert .65 < np.exp(-1.8**4 / 32) < .92
    seen = []
    def fit(src, dst, **kwargs):
        points = src.reshape(-1,2)
        x = np.floor(points[:,0] * 160).astype(int)
        y = np.floor(points[:,1] * 160).astype(int)
        densities = known[y,x]
        assert np.all(densities <= .92)
        assert np.any((densities > .65) & (densities <= .92))
        seen.append(len(points))
        return np.array([[1.,0.,0.],[0.,1.,0.]]), np.ones((len(points),1), np.uint8)
    monkeypatch.setattr(cv2, 'estimateAffine2D', fit)
    residual, count, background, mode = extractor._estimate_ego_residual(np.zeros((160,160,2),np.float32), known)
    assert mode == 'bg_lattice' and count == seen[0] >= 16
    assert extractor._background_stats['lattice_eligible'] == seen[0]
    assert extractor.lattice_steps == 32
    np.testing.assert_allclose(residual, 0)


def test_sparse_noncore_sampling_never_uses_foreground_core(monkeypatch):
    extractor = DorsalFluxExtractor(160)
    known = np.ones((160,160),np.float32)
    known[0,:20] = 0  # Outside the lattice's .03 margin; enough genuine non-core samples.
    def fit(src, dst, **kwargs):
        points = src.reshape(-1,2)
        assert np.all(points[:,1] < 1/160)
        assert 16 <= len(points) <= 256
        return np.array([[1.,0.,0.],[0.,1.,0.]]), np.ones((len(points),1),np.uint8)
    monkeypatch.setattr(cv2,'estimateAffine2D',fit)
    _, _, _, mode = extractor._estimate_ego_residual(np.zeros((160,160,2),np.float32),known)
    assert mode == 'bg_sparse_pixels'
    assert extractor._background_stats['lattice_eligible'] == 0
    assert extractor._background_stats['fit_samples'] == 20


def test_prior_budget_prefers_real_recent_observations_and_does_not_refresh_coasting():
    pipeline = Pipeline(dorsal_grid=160,heartbeat=lambda rgb: [])
    rgb = np.zeros((160,160,3),np.uint8)
    pipeline.step(rgb,0.)
    for i in range(30):
        pipeline.tracker.tracks[i] = Belief(0.,np.array([.2+i*.01,.5,0.,0.]),np.eye(4)*.001,np.eye(2)*.0001,'OBSERVED')
        pipeline.tracker.signatures[i] = np.zeros(24,np.float32)
        pipeline.last_observed_ts[i] = i*.001
    pipeline.tracker.next_id = 30
    pipeline.tracker.last_time = 0.
    result = pipeline.step(rgb,.1)
    assert pipeline.last_prior_ids == list(range(29,14,-1))
    assert result['rendered_priors'] <= 15 and len(pipeline.tracker.tracks) == 30
    assert pipeline.last_observed_ts[29] == .029
    pipeline.step(rgb,.2)
    assert pipeline.last_observed_ts[29] == .029
    assert pipeline.last_prior_ids == list(range(29,14,-1))


def test_saliency_configuration_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        DorsalFluxExtractor(min_saliency=-1)
    with pytest.raises(ValueError):
        DorsalFluxExtractor(core_density_thresh=1)
