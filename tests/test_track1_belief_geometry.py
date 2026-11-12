import copy
import numpy as np
import pytest

from track1.numerics import wavelet_matrix
from track1_belief.compression import fit_distiller, fit_task_projection
from track1_belief.geometry import (evaluate_geometry, fit_adaptive_geometry,
    gauge_transform, geometry_curves, mixture_moments, transform_gaussian)


def ring(n=96, radius=1., centre=(0., 0.)):
    t = np.arange(n) * 2 * np.pi / n
    return radius * np.c_[np.cos(t), np.sin(t)] + centre


def test_gauge_mean_covariance_and_inverse_preserve_cross_terms():
    rng = np.random.default_rng(4)
    m = 8
    controls = rng.normal(size=(m, 2))
    wavelet = wavelet_matrix(m)
    mean = (wavelet @ controls).ravel()
    a = rng.normal(size=(2 * m, 2 * m))
    covariance = a @ a.T
    rotation = np.array([[0., -1.], [1., 0.]])
    transform = gauge_transform(m, 3, rotation)
    changed_mean, changed_cov = transform_gaussian(mean, covariance, transform)
    expected = (wavelet @ (np.roll(controls, 3, axis=0) @ rotation.T)).ravel()
    np.testing.assert_allclose(changed_mean, expected, atol=1e-12)
    restored_mean, restored_cov = transform_gaussian(changed_mean, changed_cov, transform.T)
    np.testing.assert_allclose(restored_mean, mean, atol=1e-12)
    np.testing.assert_allclose(restored_cov, covariance, atol=1e-12)
    v = rng.normal(size=2 * m)
    assert v @ changed_cov @ v == pytest.approx((transform.T @ v) @ covariance @ (transform.T @ v))
    assert np.linalg.eigvalsh(changed_cov).min() >= -1e-12


def test_mixture_includes_between_hypothesis_variance_and_rejects_bad_covariance():
    mean, cov = mixture_moments([[-2., 0.], [2., 0.]], [np.eye(2), np.eye(2)], [.5, .5])
    np.testing.assert_allclose(mean, 0.)
    np.testing.assert_allclose(cov, np.diag([5., 1.]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        transform_gaussian([0., 0.], [[1., 0.], [0., -1.]], np.eye(2))
    with pytest.raises(ValueError):
        mixture_moments([[0., 0.]], [np.eye(2)], [-1.])


def test_task_projection_matches_optimal_eigen_tail_and_weighted_mass_invariance():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 5))
    a = rng.normal(size=(5, 5))
    metric = a @ a.T + np.eye(5)
    weights = rng.uniform(.1, 2., size=30)
    model = fit_task_projection(x, metric, 2, ["train"] * 30, weights)
    actual = weights @ model.distortion(x) / weights.sum()
    assert actual == pytest.approx(model.training_distortion_lower_bound, rel=1e-12)
    duplicated = fit_task_projection(np.repeat(x, 2, axis=0), metric, 2,
                                    ["train"] * 60, np.repeat(weights / 2, 2))
    np.testing.assert_allclose(model.decode(model.encode(x)), duplicated.decode(duplicated.encode(x)), atol=1e-11)
    full = fit_task_projection(x, metric, 5, ["train"] * 30)
    np.testing.assert_allclose(full.decode(full.encode(x)), x, atol=1e-11)
    zero = fit_task_projection(x, metric, 0, ["train"] * 30)
    np.testing.assert_allclose(zero.decode(zero.encode(x)), np.broadcast_to(zero.mean, x.shape))


def test_projection_and_distiller_never_read_heldout_values():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(30, 4))
    y = x @ rng.normal(size=(4, 3)) + 3
    split = np.array(["train"] * 20 + ["validation"] * 10)
    p1 = fit_task_projection(x, np.eye(4), 2, split)
    student = fit_distiller(x, y, split, ridge=0.)
    np.testing.assert_allclose(student.predict(x), y, atol=1e-12)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[20:] = np.nan
    changed_y[20:] = np.inf
    p2 = fit_task_projection(changed_x, np.eye(4), 2, split)
    changed_student = fit_distiller(changed_x, changed_y, split, ridge=0.)
    np.testing.assert_array_equal(p1.encoder, p2.encoder)
    np.testing.assert_array_equal(student.coefficients, changed_student.coefficients)
    with pytest.raises(ValueError, match="positive definite"):
        fit_task_projection(x, np.zeros((4, 4)), 2, split)


def test_distiller_regularization_is_duplicate_mass_invariant():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(20, 4)), rng.normal(size=(20, 2))
    first = fit_distiller(x, y, ["train"] * 20, ridge=.3)
    second = fit_distiller(np.repeat(x, 3, axis=0), np.repeat(y, 3, axis=0),
                          ["train"] * 60, weights=np.full(60, 1/3), ridge=.3)
    np.testing.assert_allclose(first.predict(x), second.predict(x), atol=1e-12)


def test_adaptive_geometry_allocation_certificate_binding_and_holdout():
    outer, hole = ring(radius=4.), ring(radius=.7, centre=(.3, .2))
    fit = fit_adaptive_geometry([outer[::2], hole[::2]], ["outer", "hole"], [-1, 0],
                                control_options=(8, 16), max_total_controls=24,
                                fit_samples=48, curve_samples=64, fit_steps=2)
    assert fit["supported"] and fit["certified"]
    assert sum(fit["allocation"]) <= 24
    assert fit["represented_rings"] == fit["source_rings"] == 2
    assert fit["source_completeness"]
    assert fit["coefficient_bytes"] == fit["coefficients"].nbytes
    assert len(fit["candidates"]) == 3
    assert fit["certificate"]["decoder_version"] == "canonical_tensor_bezier1"
    original = fit["coefficients"].copy()
    score = evaluate_geometry(fit, [outer[1::2], hole[1::2]], samples=128)
    assert score["supported"] and score["forward_rmse"] < .1
    np.testing.assert_array_equal(original, fit["coefficients"])
    corrupted = copy.deepcopy(fit)
    corrupted["coefficients"][0, 2] += .01
    with pytest.raises(ValueError, match="stale"):
        geometry_curves(corrupted)
    with pytest.raises(ValueError, match="every source ring"):
        evaluate_geometry(fit, [outer])


def test_adaptive_geometry_never_drops_rings_to_claim_budget_support():
    fit = fit_adaptive_geometry([ring(), ring(centre=(4., 0.))], ["outer", "outer"], [-1, -1],
                               control_options=(8,), max_total_controls=8)
    assert not fit["supported"]
    assert fit["source_rings"] == 2 and fit["represented_rings"] == 0
    assert fit["failure_code"] == "all_source_rings_exceed_control_budget"
    crossing = fit_adaptive_geometry([ring(), ring(centre=(1., 0.))], ["outer", "outer"], [-1, -1],
                                    control_options=(8,), max_total_controls=16)
    assert crossing["failure_code"] == "source_ring_topology_invalid"
