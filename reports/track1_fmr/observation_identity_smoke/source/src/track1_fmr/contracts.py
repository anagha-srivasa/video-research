"""Validated experimental data contracts; missing physical evidence stays unknown."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .belief_filter import _cov


class EpistemicTag(str, Enum):
    OBSERVED = 'OBSERVED'
    INFERRED = 'INFERRED'
    PREDICTED = 'PREDICTED'
    AMBIGUOUS = 'AMBIGUOUS'
    UNOBSERVED = 'UNOBSERVED'


def _array(value, shape, dtype):
    a = np.asarray(value, dtype=dtype).copy()
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite array of shape {shape}')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class CloudMeasure:
    mass: float
    centroid: np.ndarray
    shape_scatter: np.ndarray
    fourier_support: np.ndarray
    obs_cov_c: np.ndarray
    algebraic_sig: np.ndarray
    visibility_quality: float

    def __post_init__(self):
        if not np.isfinite(self.mass) or self.mass <= 0 or not np.isfinite(self.visibility_quality) or not 0 <= self.visibility_quality <= 1:
            raise ValueError('Positive mass and quality in [0,1] required')
        c = _array(self.centroid, (2,), np.float64)
        if np.any(c < 0) or np.any(c > 1):
            raise ValueError('Centroid must be normalized')
        object.__setattr__(self, 'centroid', c)
        for name in ('shape_scatter', 'obs_cov_c'):
            a = _cov(getattr(self, name), 2, positive=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        object.__setattr__(self, 'fourier_support', _array(self.fourier_support, (9,), np.float64))
        object.__setattr__(self, 'algebraic_sig', _array(self.algebraic_sig, (24,), np.float32))


@dataclass(frozen=True)
class SpacetimeTubeSegment:
    track_id: int
    t_start: float
    t_end: float
    legendre_c: np.ndarray
    legendre_h: np.ndarray
    shape_scatter: np.ndarray
    belief_cov_end: np.ndarray
    algebraic_sig: np.ndarray
    best_view_patch: np.ndarray | None
    best_view_time: float | None
    epistemic_tag: EpistemicTag

    def __post_init__(self):
        if not np.isfinite([self.t_start, self.t_end]).all() or self.t_end <= self.t_start:
            raise ValueError('Positive finite segment interval required')
        for name, shape, dtype in [('legendre_c', (4, 2), np.float64), ('legendre_h', (2, 9), np.float64), ('algebraic_sig', (24,), np.float32)]:
            object.__setattr__(self, name, _array(getattr(self, name), shape, dtype))
        for name, size, positive in [('shape_scatter', 2, True), ('belief_cov_end', 4, False)]:
            a = _cov(getattr(self, name), size, positive=positive)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        object.__setattr__(self, 'epistemic_tag', EpistemicTag(self.epistemic_tag))
        if (self.best_view_patch is None) != (self.best_view_time is None):
            raise ValueError('Patch and its timestamp must be supplied together')
        if self.best_view_patch is not None:
            patch = np.asarray(self.best_view_patch)
            if patch.ndim != 3 or patch.shape[2] != 3 or patch.dtype != np.uint8 or not self.t_start <= self.best_view_time <= self.t_end:
                raise ValueError('Invalid RGB patch or timestamp')
            patch = patch.copy()
            patch.setflags(write=False)
            object.__setattr__(self, 'best_view_patch', patch)

    def evaluate(self, time):
        if not self.t_start <= time <= self.t_end:
            raise ValueError('Cannot extrapolate a closed segment')
        u = 2 * (time - self.t_start) / (self.t_end - self.t_start) - 1
        return np.polynomial.legendre.legval(u, self.legendre_c)

    def velocity(self, time):
        if not self.t_start <= time <= self.t_end:
            raise ValueError('Cannot extrapolate a closed segment')
        u = 2 * (time - self.t_start) / (self.t_end - self.t_start) - 1
        return np.polynomial.legendre.legval(u, np.polynomial.legendre.legder(self.legendre_c, axis=0)) * 2 / (self.t_end - self.t_start)


@dataclass(frozen=True)
class SpacetimeJunction:
    tube_ids: tuple[int, int]
    t_interval: tuple[float, float]
    junction_type: str
    r_2d_min: float
    r_3d_est: float | None = None
    r_phys_prob: float | None = None

    def __post_init__(self):
        if len(self.tube_ids) != 2 or self.tube_ids[0] == self.tube_ids[1]:
            raise ValueError('Two distinct tube IDs required')
        if len(self.t_interval) != 2 or not np.isfinite(self.t_interval).all() or self.t_interval[1] < self.t_interval[0]:
            raise ValueError('Invalid interval')
        if self.junction_type not in ('PROXIMITY_PASS', 'MERGE_GRASP', 'SPLIT_RELEASE', 'OCCLUSION_CROSS'):
            raise ValueError('Invalid junction type')
        if not np.isfinite(self.r_2d_min) or (self.r_3d_est is not None and not np.isfinite(self.r_3d_est)):
            raise ValueError('Finite distances required')
        if self.r_phys_prob is not None and (not np.isfinite(self.r_phys_prob) or not 0 <= self.r_phys_prob <= 1):
            raise ValueError('Invalid contact probability')
