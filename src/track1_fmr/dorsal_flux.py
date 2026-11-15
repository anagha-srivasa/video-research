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


def proposal_nms(proposals, limit=15, iou_threshold=.5):
    """Stable saliency ranking; boxes use normalized xyxy coordinates."""
    if limit < 1 or not 0 <= iou_threshold <= 1:
        raise ValueError('Invalid proposal selection bounds')
    kept = []
    for proposal in sorted(proposals, key=lambda p: -p['saliency']):
        box = np.asarray(proposal['box'])
        duplicate = False
        for other in kept:
            b = np.asarray(other['box'])
            overlap = np.maximum(0, np.minimum(box[2:], b[2:]) - np.maximum(box[:2], b[:2])).prod()
            union = np.prod(box[2:] - box[:2]) + np.prod(b[2:] - b[:2]) - overlap
            if union > 0 and overlap / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
        if len(kept) == limit:
            break
    return kept


class PromptFreeHeartbeat:
    """Native-frame semantic anchors; lazy local weights, no text prompting.

    Model labels are discarded: these remain provisional region observations.
    """
    def __init__(self, checkpoint=None, device=None):
        from pathlib import Path
        from pipeline.common import ROOT
        self.checkpoint = Path(checkpoint or ROOT / 'work/yoloe-26s-seg-pf.pt').resolve()
        if not self.checkpoint.is_file() or not self.checkpoint.is_relative_to(ROOT):
            raise FileNotFoundError('Project-local prompt-free YOLOE checkpoint required')
        self.device, self.model = device, None
        self.calls = 0

    def __call__(self, rgb):
        if self.model is None:
            import os
            from pipeline.common import ROOT
            os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/track1_fmr_config'))
            os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
            from ultralytics import YOLOE
            import torch
            self.device = self.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
            self.model = YOLOE(str(self.checkpoint))
        result = self.model.predict(source=np.ascontiguousarray(rgb[..., ::-1]),
                                    device=self.device, conf=.25, imgsz=640,
                                    max_det=100, verbose=False)[0]
        self.calls += 1
        if result.masks is None:
            return []
        # A common normalized measurement grid keeps anchors independent of flux resolution.
        image = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)
        proposals = []
        for polygon, confidence in zip(result.masks.xyn, result.boxes.conf.cpu().numpy()):
            mask = np.zeros((160, 160), np.uint8)
            points = np.clip(np.round(polygon * 160), 0, 159).astype(np.int32)
            if len(points) < 3:
                continue
            cv2.fillPoly(mask, [points], 1)
            area = float(mask.mean())
            if not .0005 <= area <= .40 or mask.sum() < 3:
                continue
            ys, xs = np.nonzero(mask)
            proposals.append(dict(measure=measure(image, mask), saliency=float(confidence) * np.sqrt(area),
                                  box=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / 160))
        return proposal_nms(proposals)


def static_heartbeat(rgb):
    """Compatibility entry point; no edge fallback or hidden model downloads."""
    return [p['measure'] for p in PromptFreeHeartbeat()(rgb)]


def flux(previous, current, excluded=None, threshold=25, min_area_frac=.0005,
         max_area_frac=.40, radius_frac=.012, max_proposals=15):
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[2] != 3:
        raise ValueError('Matching RGB images required')
    if (not 0 < min_area_frac <= max_area_frac <= 1 or not np.isfinite(radius_frac)
            or radius_frac < 0 or max_proposals < 1):
        raise ValueError('Invalid normalized proposal bounds')
    h, w = previous.shape[:2]
    span = min(h, w)
    mask = np.full((h, w), 255, np.uint8)
    if excluded is not None:
        if excluded.shape != (h, w):
            raise ValueError('Exclusion mask dimensions')
        mask[np.asarray(excluded, bool)] = 0
    before = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    p = cv2.goodFeaturesToTrack(before, 500, .01, max(1., .005 * span), mask=mask)
    if p is None or len(p) < 6:
        return {'supported': False, 'reason': 'insufficient_background_points', 'clouds': []}
    q, status, _ = cv2.calcOpticalFlowPyrLK(before, after, p, None,
                                             winSize=(max(5, round(.02 * span)) | 1,) * 2)
    if q is None or status is None or status.sum() < 6:
        return {'supported': False, 'reason': 'insufficient_flow', 'clouds': []}
    keep = status.ravel().astype(bool)
    warp, inliers = cv2.estimateAffinePartial2D(p[keep], q[keep], method=cv2.RANSAC, ransacReprojThreshold=max(.3, .002 * span))
    if warp is None or inliers is None or inliers.sum() < 6:
        return {'supported': False, 'reason': 'background_fit_failed', 'clouds': []}
    aligned = cv2.warpAffine(previous, warp, (w, h))
    support = cv2.warpAffine(np.ones((h, w), np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    energy = np.mean(np.abs(current.astype(np.float32) - aligned.astype(np.float32)), axis=2)
    binary = ((energy > threshold) & support).astype(np.uint8)
    radius = max(1, round(radius_frac * span))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    proposals = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        fraction = area / (h * w)
        if area < 3 or not min_area_frac <= fraction <= max_area_frac:
            continue
        region = labels == i
        x, y, width, height, _ = stats[i]
        # Grid perimeter is a proxy for compactness, not a physical-object certificate.
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        compactness = min(1., 4 * np.pi * area / max(perimeter**2, 1.))
        saliency = float(energy[region].mean() * np.sqrt(fraction) * compactness)
        proposals.append(dict(area=area, centroid=centers[i] / [w, h],
                              box=np.array([x, y, x + width, y + height]) / [w, h, w, h],
                              saliency=saliency, component_id=i))
    clouds = proposal_nms(proposals, limit=max_proposals)
    for proposal in clouds:
        cloud = measure(current, labels == proposal.pop('component_id'))
        proposal.update(measure=cloud, scatter=cloud.shape_scatter)
    return {'supported': True, 'reason': None, 'warp': warp, 'inliers': int(inliers.sum()),
            'raw_components': count - 1, 'eligible_components': len(proposals), 'clouds': clouds}
