"""Train-only, task-metric optimal linear compression and a compact student.

The metric is supplied by a training-only task definition, not learned from
evaluation labels here. These are linear experimental models, not guarantees
about downstream task quality or calibrated uncertainty.
"""
from dataclasses import dataclass
import numpy as np


def _training(values, split, weights):
    x = np.asarray(values, dtype=float)
    labels = np.asarray(split)
    if x.ndim != 2 or labels.shape != (len(x),):
        raise ValueError("Expected a row matrix and one explicit split per row")
    selected = labels == "train"
    if not selected.any():
        raise ValueError("At least one training row is required")
    x = x[selected]
    if not np.isfinite(x).all():
        raise ValueError("Training values must be finite")
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    if w.shape != labels.shape:
        raise ValueError("Weights must have one value per input row")
    w = w[selected]
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Training weights must be nonnegative with positive sum")
    return x, w / w.sum(), selected


@dataclass(frozen=True)
class TaskProjection:
    mean: np.ndarray
    encoder: np.ndarray
    decoder: np.ndarray
    metric: np.ndarray
    eigenvalues: np.ndarray
    training_rows: int

    def encode(self, values):
        return (np.asarray(values, float) - self.mean) @ self.encoder

    def decode(self, codes):
        return np.asarray(codes, float) @ self.decoder + self.mean

    def distortion(self, values):
        error = np.asarray(values, float) - self.decode(self.encode(values))
        return np.einsum("...i,ij,...j->...", error, self.metric, error)

    @property
    def training_distortion_lower_bound(self):
        return float(self.eigenvalues[self.encoder.shape[1]:].sum())

    @property
    def parameter_bytes(self):
        return sum(a.nbytes for a in (self.mean, self.encoder, self.decoder))


def fit_task_projection(values, metric, rank, split, weights=None):
    """Solve min E[(x-xhat)' G (x-xhat)] over affine rank-r reconstructions.

    G must be strictly positive definite. Its square root maps to Euclidean
    PCA; mapping back retains the task metric. No ridge silently changes G.
    Held-out rows are not read by the fit, including normalization.
    """
    x, w, _ = _training(values, split, weights)
    d = x.shape[1]
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank <= d:
        raise ValueError("Rank must be an integer between zero and feature width")
    g = np.asarray(metric, float)
    if g.shape != (d, d) or not np.isfinite(g).all() or not np.allclose(g, g.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Task metric must be finite and symmetric")
    eig, vec = np.linalg.eigh((g + g.T) / 2)
    if eig.min() <= 0:
        raise ValueError("Task metric must be positive definite")
    root = (vec * np.sqrt(eig)) @ vec.T
    inverse = (vec / np.sqrt(eig)) @ vec.T
    mean = w @ x
    centered = (x - mean) @ root
    covariance = centered.T @ (w[:, None] * centered)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    vectors = vectors[:, order[:rank]]
    return TaskProjection(mean, root @ vectors, vectors.T @ inverse,
                          g.copy(), eigenvalues, len(x))


@dataclass(frozen=True)
class LinearDistiller:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    ridge: float
    training_rows: int

    def predict(self, inputs):
        x = np.asarray(inputs, float)
        return ((x - self.input_mean) / self.input_scale) @ self.coefficients + self.target_mean

    @property
    def parameter_bytes(self):
        return sum(a.nbytes for a in (self.input_mean, self.input_scale,
                                     self.target_mean, self.coefficients))


def fit_distiller(inputs, targets, split, weights=None, ridge=1e-3):
    """Fit a weighted ridge student with unpenalized intercept, train only.

    The teacher targets must be causal if the student is deployed causally.
    That provenance is the caller's contract; no validation rows enter fitting.
    Weight normalization makes duplication with conserved mass invariant.
    """
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge must be finite and nonnegative")
    x, w, selected = _training(inputs, split, weights)
    y = np.asarray(targets, float)
    if y.ndim != 2 or len(y) != len(selected):
        raise ValueError("Targets must be a matrix with one row per input")
    y = y[selected]
    if not np.isfinite(y).all():
        raise ValueError("Training targets must be finite")
    xm, ym = w @ x, w @ y
    scale = np.sqrt(w @ ((x - xm) ** 2))
    scale[scale <= np.finfo(float).eps] = 1
    design = (x - xm) / scale
    # An augmented least-squares solve also handles rank-deficient unregularized
    # fits without an unstable inversion of a squared-condition Gram matrix.
    a = np.vstack((np.sqrt(w[:, None]) * design, np.sqrt(ridge) * np.eye(x.shape[1])))
    b = np.vstack((np.sqrt(w[:, None]) * (y - ym), np.zeros((x.shape[1], y.shape[1]))))
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    return LinearDistiller(xm, scale, ym, coefficients, float(ridge), len(x))
