"""Screen 2-D support proximity; no inferred physical contact or depth."""
import numpy as np
from .contracts import SpacetimeJunction


def support(coefficients, angle):
    c = np.asarray(coefficients, float)
    if c.shape != (9,) or not np.isfinite(c).all():
        raise ValueError('Nine finite support harmonics required')
    k = np.arange(1, 5)
    return float(c[0] + np.dot(c[1:5], np.cos(k * angle)) + np.dot(c[5:9], np.sin(k * angle)))


def clearance(first, second, time):
    a, b = first.evaluate(time), second.evaluate(time)
    d = b - a
    angle = float(np.arctan2(d[1], d[0]))
    def h(segment, phi):
        u = 2 * (time - segment.t_start) / (segment.t_end - segment.t_start) - 1
        return support(np.polynomial.legendre.legval(u, segment.legendre_h), phi)
    return float(np.linalg.norm(d) - h(first, angle) - h(second, angle + np.pi))


def candidates(first, second, threshold=0., samples=65):
    start, end = max(first.t_start, second.t_start), min(first.t_end, second.t_end)
    if end <= start:
        return []
    if samples < 2 or not np.isfinite(threshold):
        raise ValueError('Invalid screening parameters')
    times = np.linspace(start, end, samples)
    values = np.array([clearance(first, second, t) for t in times])
    # This sampled screen is not a proof of continuous-time minimum clearance.
    if values.min() > threshold:
        return []
    return [SpacetimeJunction((first.track_id, second.track_id), (start, end), 'PROXIMITY_PASS', float(values.min()))]
