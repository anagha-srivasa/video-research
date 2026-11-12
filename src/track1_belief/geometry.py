"""Full-covariance gauge changes and bounded, certificate-aware ring allocation.

This experiment fits source contours. It does not estimate physical topology,
identity or contact. The unchanged canonical decoder alone receives its proof.
"""
from itertools import product
import numpy as np
from track1.numerics import wavelet_matrix
from track1.spatial import resample
from track1_lab.architecture import general_fit
from track1_lab.spatial import closest_polyline
from track1_lab.topology_certificate import certify, evaluate_curves, input_hashes
from track1_lab.ring_topology import predicate


def _covariance(value, dimension):
    covariance = np.asarray(value, float)
    if covariance.shape != (dimension, dimension) or not np.isfinite(covariance).all():
        raise ValueError("Covariance has incompatible dimensions or nonfinite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance).min() < -1e-10 * max(1., np.linalg.norm(covariance, 2)):
        raise ValueError("Covariance is not positive semidefinite")
    return covariance


def transform_gaussian(mean, covariance, transform, offset=None):
    """Exact affine moments, retaining all cross-channel/cross-control terms."""
    mean, transform = np.asarray(mean, float), np.asarray(transform, float)
    if mean.ndim != 1 or transform.ndim != 2 or transform.shape[1] != len(mean):
        raise ValueError("Incompatible mean and transform dimensions")
    offset = np.zeros(len(transform)) if offset is None else np.asarray(offset, float)
    if offset.shape != (len(transform),) or not all(np.isfinite(x).all() for x in (mean, transform, offset)):
        raise ValueError("Affine inputs must be finite and compatible")
    covariance = _covariance(covariance, len(mean))
    transformed = transform @ covariance @ transform.T
    return transform @ mean + offset, (transformed + transformed.T) / 2


def gauge_transform(control_count, shift, spatial_transform=None):
    """Map flattened (control, xy) db2 coefficients after a cyclic seam shift.

    A positive shift equals np.roll(controls, shift, axis=0). A fixed 2D affine
    linear part is optional. Translation belongs in transform_gaussian.offset.
    This is a declared gauge change, not a learned correspondence estimate.
    """
    if not isinstance(control_count, (int, np.integer)) or control_count < 4 or control_count % 2 or not isinstance(shift, (int, np.integer)):
        raise ValueError("Require an even control count >=4 and integer shift")
    spatial = np.eye(2) if spatial_transform is None else np.asarray(spatial_transform, float)
    if spatial.shape != (2, 2) or not np.isfinite(spatial).all():
        raise ValueError("Spatial transform must be finite 2 by 2")
    wavelet = wavelet_matrix(control_count)
    permutation = np.roll(np.eye(control_count), shift, axis=0)
    return np.kron(wavelet @ permutation @ wavelet.T, spatial)


def mixture_moments(means, covariances, weights):
    """Law of total covariance, including between-hypothesis uncertainty.

    All components must already share a gauge. The mean of valid curves is not
    generally a valid curve; these moments carry no geometry certificate.
    """
    means, covariances, weights = map(lambda a: np.asarray(a, float), (means, covariances, weights))
    if means.ndim != 2 or len(means) == 0 or weights.shape != (len(means),):
        raise ValueError("Require component means and one weight per component")
    if covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("Incompatible component covariance dimensions")
    if not np.isfinite(means).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Require finite means and nonnegative nonzero weights")
    covariances = np.stack([_covariance(c, means.shape[1]) for c in covariances])
    weights = weights / weights.sum()
    mean = weights @ means
    delta = means - mean
    covariance = np.einsum("k,kij->ij", weights, covariances) + np.einsum("k,ki,kj->ij", weights, delta, delta)
    return mean, (covariance + covariance.T) / 2


def _source_rings(rings, roles, parents):
    if not len(rings) or len(rings) != len(roles) or len(rings) != len(parents):
        raise ValueError("Every ring requires a role and parent")
    result = []
    for i, (ring, role, parent) in enumerate(zip(rings, roles, parents)):
        p = np.asarray(ring, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
            raise ValueError("Each ring requires at least four finite xy points")
        if role not in ("outer", "hole") or not isinstance(parent, (int, np.integer)) or parent < -1 or parent >= len(rings) or parent == i:
            raise ValueError("Invalid ring role or parent")
        area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
        if (area > 0) != (role == "outer"):
            p = p[::-1]
        result.append(p)
    return result


def _distortion(source, curves):
    forward, reverse = [], []
    for target, curve in zip(source, curves):
        forward.append(float(np.mean(closest_polyline(target, curve)[1] ** 2)))
        reverse.append(float(np.mean(closest_polyline(curve, target)[1] ** 2)))
    return float(np.mean(forward)), float(np.mean(reverse))


def fit_adaptive_geometry(rings, roles, parents, *, control_options=(8, 16, 24, 32),
                          max_total_controls=32, rate_penalty=0., max_certificate_attempts=4,
                          fit_samples=128, curve_samples=256, fit_steps=8):
    """Allocate periodic cubic controls per ring under an explicit total budget.

    A bounded Cartesian search (<=4096 allocations) uses retained source points
    only. Objective is normalized symmetric squared boundary distance plus
    rate_penalty times coefficient bytes. No ring is silently omitted. This is
    adaptive *ring* allocation, not a local-knot or temporal-certificate model.
    Certified static curves may be transported only under a separate contract.
    """
    source = _source_rings(rings, roles, parents)
    options = sorted(set(control_options))
    if not options or any(not isinstance(m, (int, np.integer)) or m < 4 or m % 2 for m in options):
        raise ValueError("Control choices must be even integers >=4")
    if len(options) ** len(source) > 4096:
        raise ValueError("Allocation search exceeds the bounded experiment budget")
    if not isinstance(max_total_controls, (int, np.integer)) or max_total_controls < 1:
        raise ValueError("Total control budget must be a positive integer")
    if (not np.isfinite(rate_penalty) or rate_penalty < 0
            or not isinstance(max_certificate_attempts, (int, np.integer)) or max_certificate_attempts < 1):
        raise ValueError("Require a nonnegative rate penalty and positive certificate budget")
    if fit_samples < 16 or curve_samples < 16:
        raise ValueError("At least sixteen fitting and curve samples are required")
    base = dict(supported=False, certified=False, source_rings=len(source), represented_rings=0,
                source_completeness=False, allocation=[], candidates=[],
                scope="static source-contour reconstruction; no physical topology or identity claim")
    if min(options) * len(source) > max_total_controls:
        return dict(base, failure_code="all_source_rings_exceed_control_budget")
    if not predicate(source, roles, parents)["ring_topology_valid"]:
        return dict(base, failure_code="source_ring_topology_invalid")
    sampled = [resample(p, fit_samples) for p in source]
    points = np.concatenate(sampled)
    centre = points.mean(0)
    scale = float(np.sqrt(np.mean(np.sum((points - centre) ** 2, axis=1))))
    if scale <= np.finfo(float).eps:
        return dict(base, failure_code="degenerate_source_scale")
    normalized = [(p - centre) / scale for p in sampled]
    fits = {}
    for i, ring in enumerate(normalized):
        for m in options:
            if m + (len(source) - 1) * min(options) <= max_total_controls:
                fits[i, m] = general_fit(ring, m=m, steps=fit_steps)
    candidates = []
    for allocation in product(options, repeat=len(source)):
        if sum(allocation) > max_total_controls:
            continue
        if not all(fits[i, m]["valid"] for i, m in enumerate(allocation)):
            candidates.append(dict(allocation=list(allocation), fit_valid=False,
                                   certified=False, certificate_attempted=False))
            continue
        coefficients = np.zeros((4, 3 + 2 * sum(allocation)), np.float32)
        offset = 2
        for i, m in enumerate(allocation):
            coefficients[0, offset:offset + 2 * m] = (wavelet_matrix(m) @ fits[i, m]["controls"]).ravel()
            offset += 2 * m
        curves = evaluate_curves(coefficients, allocation, roles, .5, samples=curve_samples)
        forward, reverse = _distortion(normalized, curves)
        loss = (forward + reverse) / 2
        candidates.append(dict(allocation=list(allocation), fit_valid=True, certified=False,
                               certificate_attempted=False, retained_normalized_mse=loss,
                               coefficient_bytes=coefficients.nbytes,
                               objective=loss + rate_penalty * coefficients.nbytes,
                               coefficients=coefficients))
    ordered = sorted((c for c in candidates if c["fit_valid"]), key=lambda c: (c["objective"], c["coefficient_bytes"], c["allocation"]))
    selected = None
    for candidate in ordered[:max_certificate_attempts]:
        certificate = certify(candidate["coefficients"], candidate["allocation"], roles, parents,
                              window_seconds=1., max_depth=0, max_nodes=1, max_spatial_refinement=2)
        candidate.update(certificate_attempted=True, certified=certificate["certified"],
                         certificate_failure=certificate["failure_code"])
        if certificate["certified"]:
            selected = dict(candidate, certificate=certificate)
            break
    ledger = [{k: v for k, v in candidate.items() if k != "coefficients"} for candidate in candidates]
    if selected is None:
        return dict(base, candidates=ledger, failure_code="no_certified_allocation_within_search_budget")
    allocation = selected["allocation"]
    coefficients = selected["coefficients"]
    controls = []
    offset = 2
    for m in allocation:
        controls.append(wavelet_matrix(m).T @ coefficients[0, offset:offset + 2 * m].reshape(m, 2))
        offset += 2 * m
    return dict(base, supported=True, certified=True, source_completeness=True,
                represented_rings=len(source), allocation=allocation, roles=list(roles), parents=list(parents),
                coefficients=coefficients, controls=controls, centre=centre, scale=scale,
                coefficient_bytes=coefficients.nbytes, control_points=sum(allocation),
                numeric_payload_bytes=coefficients.nbytes + centre.nbytes + 8 + 9 * len(source),
                retained_normalized_mse=selected["retained_normalized_mse"],
                certificate=selected["certificate"], candidates=ledger, failure_code=None)


def geometry_curves(fitted, samples=512, *, require_certificate=True):
    """Decode only this fit's canonical curve; fail closed on changed bindings."""
    if not fitted.get("supported"):
        raise ValueError("Unsupported geometry fit")
    centre = np.asarray(fitted["centre"], float)
    scale = float(fitted["scale"])
    if centre.shape != (2,) or not np.isfinite(centre).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Geometry requires a finite centre and positive similarity scale")
    if require_certificate:
        certificate = fitted.get("certificate", {})
        actual = input_hashes(fitted["coefficients"], fitted["allocation"], fitted["roles"], fitted["parents"], 1.)
        if not certificate.get("certified") or any(certificate.get(k) != v for k, v in actual.items()):
            raise ValueError("Missing or stale canonical geometry certificate")
    normalized = evaluate_curves(fitted["coefficients"], fitted["allocation"], fitted["roles"], .5, samples=samples)
    return [p * scale + centre for p in normalized]


def evaluate_geometry(fitted, heldout_rings, *, samples=512):
    """Report source-point holdout error without changing or selecting the fit.

    Callers must split source points before fitting. Ring correspondence is
    supplied, not inferred here. Polygonal distances approximate sampled curves;
    they are not independent physical-boundary annotations.
    """
    if not fitted.get("supported"):
        return dict(supported=False, failure_code=fitted.get("failure_code"),
                    source_rings=fitted["source_rings"], represented_rings=0)
    if len(heldout_rings) != len(fitted["allocation"]):
        raise ValueError("Holdout must retain every source ring")
    targets = _source_rings(heldout_rings, fitted["roles"], fitted["parents"])
    forward, reverse = _distortion(targets, geometry_curves(fitted, samples))
    return dict(supported=True, certified=True, source_rings=len(targets), represented_rings=len(targets),
                forward_rmse=float(np.sqrt(forward)), reverse_rmse=float(np.sqrt(reverse)),
                symmetric_mse=(forward + reverse) / 2, control_points=fitted["control_points"],
                coefficient_bytes=fitted["coefficient_bytes"],
                reference_kind="withheld source polygon points; independence is caller-enforced",
                certificate_scope="normalized canonical static curves, not raster or physical geometry")
