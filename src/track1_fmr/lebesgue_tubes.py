"""Causal innovation-triggered trajectory segments; experimental."""
from dataclasses import dataclass
import numpy as np
from .contracts import SpacetimeTubeSegment, EpistemicTag


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    coefficients: np.ndarray
    count: int
    reason: str

    def position(self, time):
        if not self.start <= time <= self.end:
            raise ValueError('Time is outside the segment')
        u = 2 * (time - self.start) / (self.end - self.start) - 1
        return np.polynomial.legendre.legval(u, self.coefficients)


class TubeBuilder:
    def __init__(self, threshold=20., maximum_duration=4.):
        if threshold <= 0 or maximum_duration <= 0:
            raise ValueError('Positive threshold and duration required')
        self.threshold, self.maximum_duration = threshold, maximum_duration
        self.samples, self.energy = [], 0.

    def boundary_reason(self, time, innovation, status):
        if not self.samples:
            return None
        if status != self.samples[-1][2]:
            return 'visibility_transition'
        if time - self.samples[0][0] >= self.maximum_duration:
            return 'maximum_duration'
        if self.energy + max(0., innovation - 4.605170186) > self.threshold:
            return 'innovation'
        return None

    def add(self, time, position, innovation, status, weight=1.):
        point = np.asarray(position, float)
        if (point.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(time)
                or not np.isfinite(innovation) or innovation < 0 or status not in ('OBSERVED', 'PREDICTED')
                or not np.isfinite(weight) or weight <= 0):
            raise ValueError('Invalid tube observation')
        if self.samples and time <= self.samples[-1][0]:
            raise ValueError('Times must increase')
        energy = self.energy + max(0., innovation - 4.605170186)
        reason = self.boundary_reason(time, innovation, status)
        segment = self.close(reason) if reason else None
        self.samples.append((float(time), point.copy(), status, float(weight)))
        self.energy = 0. if reason else energy
        return segment

    def close(self, reason='end'):
        samples, self.samples, self.energy = self.samples, [], 0.
        if len(samples) < 2:
            return None
        times = np.array([s[0] for s in samples])
        u = 2 * (times - times[0]) / (times[-1] - times[0]) - 1
        degree = min(3, len(samples) - 1)
        coefficients = np.zeros((4, 2))
        weights = np.sqrt(np.array([s[3] for s in samples]))
        weights /= np.median(weights)
        coefficients[:degree + 1] = np.polynomial.legendre.legfit(u, np.array([s[1] for s in samples]), degree, w=weights)
        coefficients.setflags(write=False)
        return Segment(times[0], times[-1], coefficients, len(samples), reason)


class FunctionalTubeBuilder:
    """Bounded variable-epoch fit with a single selected RGB crop."""
    def __init__(self, track_id, threshold=20., maximum_duration=4., carry_prior=False):
        self.track_id = track_id
        self.clock = TubeBuilder(threshold, maximum_duration)
        self.data = []
        self.best = None
        self.carry_prior = bool(carry_prior)
        self.last_observed_cloud = None
        self.unsupported_short_epochs = 0
        self.unsupported_geometry_epochs = 0

    def wants_patch(self, state, innovation, quality):
        if not np.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('Finite crop quality in [0,1] required')
        return (self.best is None or quality > self.best[2]
                or self.clock.boundary_reason(state.time, innovation, state.status) is not None)

    def add(self, state, cloud, innovation, patch=None, quality=None):
        if state.status == 'OBSERVED' and cloud is None:
            raise ValueError('Observed state requires a cloud')
        if patch is not None and (cloud is None or quality is None or not np.isfinite(quality) or not 0 <= quality <= 1):
            raise ValueError('A patch requires an observed quality')
        before = len(self.clock.samples)
        weight = 1. / max(float(np.trace(state.covariance[:2, :2])), 1e-12)
        closed = self.clock.add(state.time, state.mean[:2], innovation, state.status, weight=weight)
        result = self._finish(closed) if closed else None
        if before and len(self.clock.samples) == 1:
            if closed is None:
                self.unsupported_short_epochs += 1
            self.data = []
            self.best = None
        if cloud is not None:
            self.last_observed_cloud = cloud
        self.data.append((state, cloud if cloud is not None else self.last_observed_cloud if self.carry_prior else None))
        if patch is not None and (self.best is None or quality > self.best[2]):
            image = np.asarray(patch)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('Best-view patch must be uint8 RGB')
            self.best = (state.time, image.copy(), float(quality))
        return result

    def close(self):
        segment = self.clock.close()
        result = self._finish(segment) if segment else None
        if self.data and segment is None:
            self.unsupported_short_epochs += 1
        self.data = []
        self.best = None
        self.last_observed_cloud = None
        return result

    def _finish(self, segment):
        samples = self.data
        times = np.array([s[0].time for s in samples])
        usable = [i for i, s in enumerate(samples) if s[1] is not None]
        if not usable:
            self.unsupported_geometry_epochs += 1
            return None
        harmonics = np.zeros((2, 9))
        signature = np.zeros(24, np.float32)
        if usable:
            u = 2 * (times[usable] - segment.start) / (segment.end - segment.start) - 1
            degree = min(1, len(usable) - 1)
            weight = np.sqrt([1. / max(float(np.trace(samples[i][1].obs_cov_c)), 1e-12) for i in usable])
            weight /= np.median(weight)
            harmonics[:degree + 1] = np.polynomial.legendre.legfit(u, np.array([samples[i][1].fourier_support for i in usable]), degree, w=weight)
            signature = np.mean([samples[i][1].algebraic_sig for i in usable], axis=0).astype(np.float32)
        best = self.best
        last = samples[-1][0]
        return SpacetimeTubeSegment(self.track_id, segment.start, segment.end, segment.coefficients, harmonics,
                                    last.shape_scatter, last.covariance, signature,
                                    None if best is None else best[1], None if best is None else best[0],
                                    EpistemicTag(last.status))
