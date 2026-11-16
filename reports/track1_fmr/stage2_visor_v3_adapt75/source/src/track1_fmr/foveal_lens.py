"""Evidence-gated geometry; entropy is not a substitute for an EVI model."""
import numpy as np


def entropy(probability):
    p = float(probability)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Probability must be in [0,1]')
    if p in (0., 1.):
        return 0.
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def expected_value(prior, posterior_probabilities, outcome_probabilities, cost):
    p = np.asarray(posterior_probabilities, float)
    weights = np.asarray(outcome_probabilities, float)
    if p.ndim != 1 or not len(p) or weights.shape != p.shape or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1) or not np.isfinite(cost) or cost < 0:
        raise ValueError('A normalized outcome model and nonnegative cost are required')
    if not np.isclose(np.dot(p, weights), prior, atol=1e-8):
        raise ValueError('Posterior outcome model must preserve the prior')
    return entropy(prior) - float(np.dot(weights, [entropy(x) for x in p])) - cost


def refine_if_useful(prior, posteriors, weights, cost, coefficients, metadata, timestamp_us):
    value = expected_value(prior, posteriors, weights, cost)
    if value <= 0:
        return {'triggered': False, 'evi': value, 'curves': None}
    from track1_encoder_revision.runtime import CompiledGeometry
    curves = CompiledGeometry(coefficients, metadata).evaluate(timestamp_us)
    return {'triggered': True, 'evi': value, 'curves': curves}
