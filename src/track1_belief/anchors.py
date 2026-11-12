"""Native-pixel causal point-tracking baseline with disjoint diagnostic anchors.

The supplied initial mask is privileged initialization. No later mask is read by
the tracker. Covariances are conditional engineering estimates, not calibrated
physical uncertainty. LK cannot certify re-identification after an occlusion.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class AnchorConfig:
    foreground_points: int = 96
    background_points: int = 160
    minimum_fit_points: int = 6
    minimum_proxy_points: int = 4
    fb_threshold_px: float = 1.0
    ransac_threshold_px: float = 2.0
    measurement_floor_px: float = 0.5
    seed_uncertainty_px: float = 1.0
    maximum_gap_seconds: float = 0.1


def similarity_fit(origin, current, reference, *, minimum=4, threshold=2., floor=.5):
    """Fit only the supplied point set; return reference-point covariance.

    A four-parameter linear similarity is refit after deterministic RANSAC.
    Residual deformation remains separate from finite-point estimator variance.
    """
    origin = np.asarray(origin, float)
    current = np.asarray(current, float)
    reference = np.asarray(reference, float)
    if origin.shape != current.shape or origin.ndim != 2 or origin.shape[1] != 2:
        raise ValueError('Expected paired 2D points')
    if not np.isfinite(origin).all() or not np.isfinite(current).all():
        raise ValueError('Non-finite points')
    if len(origin) < minimum:
        return None
    cv2.setRNGSeed(112)
    affine, support = cv2.estimateAffinePartial2D(
        origin, current, method=cv2.RANSAC, ransacReprojThreshold=threshold,
        maxIters=1000, confidence=.995, refineIters=10)
    if affine is None or support is None:
        return None
    keep = support.ravel().astype(bool)
    if keep.sum() < minimum or keep.mean() < .5:
        return None
    p = origin[keep] - reference
    q = current[keep]
    design = np.zeros((2 * len(p), 4))
    design[0::2] = np.c_[p[:, 0], -p[:, 1], np.ones(len(p)), np.zeros(len(p))]
    design[1::2] = np.c_[p[:, 1], p[:, 0], np.zeros(len(p)), np.ones(len(p))]
    if np.linalg.matrix_rank(design) < 4:
        return None
    theta = np.linalg.lstsq(design, q.ravel(), rcond=None)[0]
    residual = q.ravel() - design @ theta
    sigma2 = max(float(residual @ residual / max(2 * len(p) - 4, 1)), floor ** 2)
    parameter_cov = np.linalg.inv(design.T @ design) * sigma2
    linear = np.array([[theta[0], -theta[1]], [theta[1], theta[0]]])
    # Positive scale does not establish a rigid physical object or valid camera.
    scale = float(np.hypot(theta[0], theta[1]))
    if not .2 < scale < 5:
        return None
    affine = np.c_[linear, theta[2:] - linear @ reference]
    return dict(position=theta[2:], covariance=parameter_cov[2:, 2:],
                affine=affine, parameter_covariance=parameter_cov,
                coherence=float(keep.mean()), inliers=int(keep.sum()),
                deformation_rmse=float(np.sqrt(np.mean(residual ** 2))),
                support=keep)


class PersistentAnchors:
    """Bounded past-only LK tracker. Lost anchors stay lost; no silent reseeding."""

    def __init__(self, gray, mask, timestamp_us, config=None, background_exclusion=None):
        self.config = config or AnchorConfig()
        cv2.setNumThreads(1)
        gray = self._gray(gray)
        mask = np.asarray(mask, bool)
        if mask.shape != gray.shape:
            raise ValueError('Initial mask must use native image dimensions')
        if not mask.any():
            raise ValueError('Empty initial mask')
        self.shape = gray.shape
        self.seed_us = self.last_us = int(timestamp_us)
        yy, xx = np.nonzero(mask)
        self.reference = np.array([xx.mean(), yy.mean()])
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        if eroded.sum() < 20:
            eroded = mask.astype(np.uint8)
        foreground = self._corners(gray, eroded, self.config.foreground_points)
        excluded = mask if background_exclusion is None else np.asarray(background_exclusion, bool) | mask
        if excluded.shape != gray.shape:
            raise ValueError('Background exclusion dimensions')
        background_mask = 1 - cv2.dilate(excluded.astype(np.uint8), np.ones((31, 31), np.uint8))
        background = self._corners(gray, background_mask, self.config.background_points)
        self.origin = np.concatenate([foreground, background], axis=0)
        self.points = self.origin.copy()
        self.roles = np.r_[np.arange(len(foreground)) % 3 == 0,
                           np.full(len(background), 2)].astype(np.int8)
        # Foreground role 0 fits the state; role 1 is never consumed by that fit.
        self.alive = np.ones(len(self.points), bool)
        self.fb_error = np.zeros(len(self.points))
        self.previous = gray.copy()
        self.last = self._summarize()

    @staticmethod
    def _gray(image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError('Images must be uint8')
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim == 2:
            return image
        raise ValueError('Expected grayscale or RGB')

    @staticmethod
    def _corners(gray, mask, count):
        points = cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=.01,
                                        minDistance=6, mask=mask * 255, blockSize=5)
        return np.empty((0, 2), np.float32) if points is None else points[:, 0]

    def update(self, image, timestamp_us):
        gray = self._gray(image)
        timestamp_us = int(timestamp_us)
        if gray.shape != self.shape or timestamp_us <= self.last_us:
            raise ValueError('Image shape or strict timestamp order violated')
        if timestamp_us - self.last_us > self.config.maximum_gap_seconds * 1e6:
            self.alive[:] = False
        ids = np.flatnonzero(self.alive)
        if len(ids):
            p = self.points[ids].reshape(-1, 1, 2).astype(np.float32)
            kwargs = dict(winSize=(25, 25), maxLevel=4,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            q, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, p, None, **kwargs)
            if q is None or ok is None:
                self.alive[ids] = False
            else:
                back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, q, None, **kwargs)
                if back is None or back_ok is None:
                    self.alive[ids] = False
                else:
                    err = np.linalg.norm(back[:, 0] - p[:, 0], axis=1)
                    height, width = self.shape
                    inside = ((q[:, 0, 0] >= 0) & (q[:, 0, 0] < width) &
                              (q[:, 0, 1] >= 0) & (q[:, 0, 1] < height))
                    good = ((ok.ravel() > 0) & (back_ok.ravel() > 0) & inside &
                            np.isfinite(q[:, 0]).all(1) & (err <= self.config.fb_threshold_px))
                    self.alive[ids] = good
                    self.points[ids[good]] = q[good, 0]
                    self.fb_error[ids[good]] = err[good]
        self.previous = gray.copy()
        self.last_us = timestamp_us
        self.last = self._summarize()
        return self.last

    def _fit(self, role, minimum):
        ids = self.alive & (self.roles == role)
        return similarity_fit(self.origin[ids], self.points[ids], self.reference,
                              minimum=minimum, threshold=self.config.ransac_threshold_px,
                              floor=self.config.measurement_floor_px)

    def _summarize(self):
        cfg = self.config
        fitted = self._fit(0, cfg.minimum_fit_points)
        proxy = self._fit(1, cfg.minimum_proxy_points)
        background = self._fit(2, max(10, cfg.minimum_fit_points))
        zeros = np.zeros(2)
        cov = np.eye(2) * cfg.measurement_floor_px ** 2
        result = dict(position=zeros.copy(), covariance_local=cov.copy(),
                      covariance_shared=np.eye(2) * cfg.seed_uncertainty_px ** 2,
                      valid=fitted is not None, reason='' if fitted is not None else 'insufficient_consistent_persistent_anchors',
                      proxy_position=zeros.copy(), proxy_valid=proxy is not None,
                      camera_relative_position=zeros.copy(), camera_relative_covariance=cov.copy(),
                      camera_valid=False, coherence=0., deformation_rmse=0.,
                      latest_evidence_us=self.last_us, surviving_fit=int(np.sum(self.alive & (self.roles == 0))),
                      surviving_proxy=int(np.sum(self.alive & (self.roles == 1))),
                      surviving_background=int(np.sum(self.alive & (self.roles == 2))),
                      point_positions=self.points.copy(), point_valid=self.alive.copy())
        if proxy is not None:
            result['proxy_position'] = proxy['position']
        if fitted is None:
            return result
        result.update(position=fitted['position'], covariance_local=fitted['covariance'],
                      coherence=fitted['coherence'], deformation_rmse=fitted['deformation_rmse'])
        # Shared deformation allowance is NOT divided by point count. Temporal
        # correlation is represented downstream by the persistent seed group.
        result['covariance_shared'] += np.eye(2) * fitted['deformation_rmse'] ** 2
        if background is not None:
            ids = self.alive & (self.roles == 2)
            p = self.origin[ids][background['support']]
            height, width = self.shape
            tiles = {(min(2, int(x * 3 / width)), min(2, int(y * 3 / height))) for x, y in p}
            # A spatially concentrated foreground patch cannot authorize a camera correction.
            if len(tiles) >= 4 and background['coherence'] >= .7:
                matrix = background['affine'][:, :2]
                inverse = np.linalg.inv(matrix)
                result['camera_relative_position'] = inverse @ (fitted['position'] - background['affine'][:, 2])
                result['camera_relative_covariance'] = inverse @ (
                    fitted['covariance'] + result['covariance_shared'] + background['covariance'] +
                    np.eye(2) * background['deformation_rmse'] ** 2) @ inverse.T
                result['camera_valid'] = True
        return result
