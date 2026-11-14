"""Background-aligned image change; clouds are proposals, not objects."""
import cv2
import numpy as np
from .contracts import CloudMeasure


def measure(rgb, mask):
    """Summarize a proposed region; depth, visibility and semantics are unknown."""
    mask = np.asarray(mask, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2] or mask.sum() < 3:
        raise ValueError('RGB and nonempty region required')
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    xy = np.c_[xx / width, yy / height]
    centre = xy.mean(0)
    scatter = np.cov(xy, rowvar=False, bias=True) + np.eye(2) * 1e-9
    directions = np.arange(36) * (2 * np.pi / 36)
    projections = (xy - centre) @ np.c_[np.cos(directions), np.sin(directions)].T
    radius = projections.max(0)
    harmonic = np.r_[radius.mean(), [2 * np.mean(radius * np.cos(k * directions)) for k in range(1, 5)],
                     [2 * np.mean(radius * np.sin(k * directions)) for k in range(1, 5)]]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(float)
    signature = np.zeros(24, np.float32)
    signature[:4] = np.hypot(harmonic[1:5], harmonic[5:9])
    signature[4:10] = np.r_[lab.mean(0), lab.std(0)]
    signature[10:12] = np.linalg.eigvalsh(scatter)
    # Geometric moment dispersion is only an engineering proxy for observation noise.
    covariance = scatter / len(xy) + np.eye(2) * 1e-8
    return CloudMeasure(float(mask.sum() / (height * width)), centre, scatter, harmonic,
                        covariance, signature, 0.)


def static_heartbeat(rgb, min_area=100, max_area_fraction=.25):
    """Class-agnostic edge proposals; stationary-object recall is unverified."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not 0 < max_area_fraction <= 1:
        raise ValueError('RGB and a valid maximum area fraction required')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    clouds = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > height * width * max_area_fraction:
            continue
        mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        if mask.sum() >= 3:
            clouds.append(measure(rgb, mask))
    return clouds


def flux(previous, current, excluded=None, threshold=25, min_area=40):
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[2] != 3:
        raise ValueError('Matching RGB images required')
    h, w = previous.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    if excluded is not None:
        if excluded.shape != (h, w):
            raise ValueError('Exclusion mask dimensions')
        mask[np.asarray(excluded, bool)] = 0
    before = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    p = cv2.goodFeaturesToTrack(before, 500, .01, 8, mask=mask)
    if p is None or len(p) < 6:
        return {'supported': False, 'reason': 'insufficient_background_points', 'clouds': []}
    q, status, _ = cv2.calcOpticalFlowPyrLK(before, after, p, None)
    if q is None or status is None or status.sum() < 6:
        return {'supported': False, 'reason': 'insufficient_flow', 'clouds': []}
    keep = status.ravel().astype(bool)
    warp, inliers = cv2.estimateAffinePartial2D(p[keep], q[keep], method=cv2.RANSAC, ransacReprojThreshold=2.)
    if warp is None or inliers is None or inliers.sum() < 6:
        return {'supported': False, 'reason': 'background_fit_failed', 'clouds': []}
    aligned = cv2.warpAffine(previous, warp, (w, h))
    support = cv2.warpAffine(np.ones((h, w), np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    energy = np.mean(np.abs(current.astype(np.float32) - aligned.astype(np.float32)), axis=2)
    binary = ((energy > threshold) & support).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    clouds = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        yy, xx = np.nonzero(labels == i)
        xy = np.column_stack((xx / w, yy / h))
        clouds.append(dict(area=area, centroid=centers[i] / [w, h], scatter=np.cov(xy, rowvar=False, bias=True),
                           measure=measure(current, labels == i)))
    return {'supported': True, 'reason': None, 'warp': warp, 'inliers': int(inliers.sum()), 'clouds': clouds}
