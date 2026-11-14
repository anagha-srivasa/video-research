"""Causal 2-D constant-velocity belief; shape is separate from uncertainty."""
from dataclasses import dataclass
import numpy as np


def _cov(value, n, positive=False):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n, n) or not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-10):
        raise ValueError('Expected a finite symmetric covariance')
    a = (a + a.T) / 2
    if np.linalg.eigvalsh(a).min() < (1e-12 if positive else -1e-10):
        raise ValueError('Covariance must be positive definite' if positive else 'Covariance must be positive semidefinite')
    return a.copy()


@dataclass(frozen=True)
class Belief:
    time: float
    mean: np.ndarray
    covariance: np.ndarray
    shape_scatter: np.ndarray
    status: str

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        if mean.shape != (4,) or not np.isfinite(mean).all() or not np.isfinite(self.time):
            raise ValueError('Finite time and [x, y, vx, vy] required')
        if self.status not in ('OBSERVED', 'PREDICTED'):
            raise ValueError('Unsupported belief status')
        covariance = _cov(self.covariance, 4)
        shape = _cov(self.shape_scatter, 2, positive=True)
        for a in (mean, covariance, shape):
            a.setflags(write=False)
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'covariance', covariance)
        object.__setattr__(self, 'shape_scatter', shape)


def advance(state, time, acceleration_noise, position=None, observation_covariance=None, shape_scatter=None):
    """Predict, optionally update with a real observation; never invent one.

    Noise is a continuous white-acceleration spectral density. Covariance is
    conditional, not independently calibrated physical uncertainty.
    """
    dt = float(time) - state.time
    q = float(acceleration_noise)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(q) or q < 0:
        raise ValueError('Strictly increasing time and nonnegative noise required')
    eye = np.eye(2)
    f = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
    process = q * np.block([[dt**3 / 3 * eye, dt**2 / 2 * eye], [dt**2 / 2 * eye, dt * eye]])
    mean = f @ state.mean
    cov = f @ state.covariance @ f.T + process
    shape = state.shape_scatter
    status = 'PREDICTED'
    if position is not None:
        z = np.asarray(position, dtype=np.float64)
        if z.shape != (2,) or not np.isfinite(z).all() or observation_covariance is None:
            raise ValueError('Observed position requires a finite positive covariance')
        r = _cov(observation_covariance, 2, positive=True)
        h = np.c_[eye, np.zeros((2, 2))]
        gain = np.linalg.solve(h @ cov @ h.T + r, h @ cov).T
        mean += gain @ (z - h @ mean)
        a = np.eye(4) - gain @ h
        cov = a @ cov @ a.T + gain @ r @ gain.T
        if shape_scatter is not None:
            shape = _cov(shape_scatter, 2, positive=True)
        status = 'OBSERVED'
    elif observation_covariance is not None or shape_scatter is not None:
        raise ValueError('Coasting cannot consume observation-only inputs')
    return Belief(float(time), mean, (cov + cov.T) / 2, shape, status)


def association_distance(predicted, cloud):
    """Position gating distance; appearance identity is not certified."""
    delta = cloud.centroid - predicted.mean[:2]
    covariance = predicted.covariance[:2, :2] + cloud.obs_cov_c
    return float(delta @ np.linalg.solve(covariance, delta))


def directional_noise(covariance, angles, visibility, inflation):
    """Inflate declared ray directions, without inferring visibility from RGB."""
    cov = _cov(covariance, 2, positive=True)
    angles, visibility = np.asarray(angles, float), np.asarray(visibility, float)
    if angles.ndim != 1 or angles.shape != visibility.shape or not np.isfinite(angles).all() or not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)) or not np.isfinite(inflation) or inflation < 0:
        raise ValueError('Valid visibility rays and nonnegative inflation required')
    rays = np.c_[np.cos(angles), np.sin(angles)]
    return cov + inflation * np.einsum('n,ni,nj->ij', 1 - visibility, rays, rays) / max(len(rays), 1)


class BeliefTracker:
    """Bounded proposal association and coasting; IDs are provisional."""
    def __init__(self, acceleration_noise=.001, gate=9., max_tracks=100):
        if acceleration_noise < 0 or gate <= 0 or max_tracks < 1:
            raise ValueError('Invalid tracker configuration')
        self.noise, self.gate, self.max_tracks = acceleration_noise, gate, max_tracks
        self.tracks = {}
        self.signatures = {}
        self.next_id = 0
        self.last_time = None

    def update(self, time, clouds):
        from scipy.optimize import linear_sum_assignment
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError('Times must increase')
        ids = list(self.tracks)
        predicted = [advance(self.tracks[i], time, self.noise) for i in ids]
        cost = np.full((len(ids), len(clouds)), 1e6)
        for i, state in enumerate(predicted):
            for j, cloud in enumerate(clouds):
                d = association_distance(state, cloud)
                if d <= self.gate:
                    # Color and shape are heuristic features, not a calibrated identity likelihood.
                    difference = self.signatures[ids[i]][:10] - cloud.algebraic_sig[:10]
                    cost[i, j] = d + float(np.linalg.norm(difference) / 100)
        pairs = []
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(rows, columns) if cost[i, j] < 1e6]
        assigned = {i: j for i, j in pairs}
        used = {j for _, j in pairs}
        associated = {}
        for i, track_id in enumerate(ids):
            if i in assigned:
                cloud = clouds[assigned[i]]
                associated[track_id] = cloud
                # Current visible scatter may change with occlusion, so preserve prior shape.
                self.tracks[track_id] = advance(self.tracks[track_id], time, self.noise, cloud.centroid, cloud.obs_cov_c)
                self.signatures[track_id] = cloud.algebraic_sig.copy()
            else:
                self.tracks[track_id] = predicted[i]
        rejected = 0
        for j, cloud in enumerate(clouds):
            if j in used:
                continue
            if len(self.tracks) >= self.max_tracks:
                rejected += 1
                continue
            mean = np.r_[cloud.centroid, [0., 0.]]
            covariance = np.zeros((4, 4))
            # A region's centroid is not an accurately localized physical point.
            # This fixed floor is uncalibrated and must be evaluated on held-out references.
            covariance[:2, :2] = cloud.obs_cov_c + np.eye(2) * 1e-4
            covariance[2:, 2:] = np.eye(2) * .01
            self.tracks[self.next_id] = Belief(float(time), mean, covariance, cloud.shape_scatter, 'OBSERVED')
            associated[self.next_id] = cloud
            self.signatures[self.next_id] = cloud.algebraic_sig.copy()
            self.next_id += 1
        self.last_time = float(time)
        return dict(tracks=self.tracks.copy(), clouds=associated, matched=len(pairs), rejected_capacity=rejected,
                    identity_verified=False, covariance_calibrated=False)
