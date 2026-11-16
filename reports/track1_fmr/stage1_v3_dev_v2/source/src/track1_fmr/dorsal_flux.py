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
            proposals.append(dict(measure=measure(image, mask), mask=mask.astype(bool), saliency=float(confidence) * np.sqrt(area),
                                  box=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / 160))
        return proposal_nms(proposals)


def static_heartbeat(rgb):
    """Compatibility entry point; no edge fallback or hidden model downloads."""
    return [p['measure'] for p in PromptFreeHeartbeat()(rgb)]


def anti_alias(rgb, size=None, *, grayscale=False):
    """Area-to-2x, 5x5 Gaussian, then linear decimation (a new filter response)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
        raise ValueError('uint8 RGB image with dimensions >=16 required')
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if grayscale else rgb
    if size is None:
        return image
    if isinstance(size, bool) or not isinstance(size, int) or size < 16:
        raise ValueError('Grid must be an integer >=16')
    h, w = rgb.shape[:2]
    if size > min(h, w):
        raise ValueError('Dorsal grid must not upsample either image axis')
    intermediate = cv2.resize(image, (2 * size, 2 * size), interpolation=cv2.INTER_AREA)
    filtered = cv2.GaussianBlur(intermediate, (5, 5), sigmaX=1., sigmaY=1.)
    return cv2.resize(filtered, (size, size), interpolation=cv2.INTER_LINEAR)


def explained_density(residual, background, regions, sigma_flow=.005):
    """Density explained by matching motion; all displacements are normalized/frame.

    Region displacement is in image coordinates (state velocity * exact dt).
    Convert it to the same affine-residual coordinates as the measured flow.
    """
    if not np.isfinite(sigma_flow) or sigma_flow <= 0:
        raise ValueError('Positive flow displacement scale required')
    explained = np.zeros(residual.shape[:2], np.float32)
    image_flow = residual + background
    for region in regions:
        density = np.asarray(region['density'], np.float32)
        displacement = np.asarray(region['displacement'], np.float32)
        if (density.shape != explained.shape or displacement.shape != (2,)
                or not np.isfinite(density).all() or not np.isfinite(displacement).all()
                or np.any((density < 0) | (density > 1))):
            raise ValueError('Finite density in [0,1] and image displacement required')
        ys, xs = np.nonzero(density)
        delta = image_flow[ys, xs] - displacement
        weight = np.exp(-np.sum(delta**2, axis=-1) / (2 * sigma_flow**2))
        explained[ys, xs] = np.maximum(explained[ys, xs], density[ys, xs] * weight)
    return explained


def flux(previous, current, excluded=None, v_min=.0035, min_area_frac=.00025,
         max_area_frac=.40, radius_frac=.012, max_proposals=10,
         previous_gray=None, current_gray=None, known_regions=(), sigma_flow=.005):
    """DIS motion residuals in normalized image units/frame, not physical velocity.

    Affine image-motion subtraction does not resolve parallax or metric motion.
    Spatial exclusion selects background samples; velocity-consistent density gates
    discovery. Coasting is never converted to an observation.
    """
    if (previous.dtype != np.uint8 or current.dtype != np.uint8 or previous.shape != current.shape
            or previous.ndim != 3 or previous.shape[2] != 3 or min(previous.shape[:2]) < 16):
        raise ValueError('Matching uint8 RGB images with dimensions >=16 required')
    if (not 0 < min_area_frac <= max_area_frac <= 1 or not np.isfinite(radius_frac)
            or radius_frac < 0 or not np.isfinite(v_min) or v_min <= 0 or max_proposals < 1):
        raise ValueError('Invalid normalized proposal bounds')
    h, w = previous.shape[:2]
    known = np.zeros((h, w), np.float32) if excluded is None else np.asarray(excluded, np.float32)
    if known.shape != (h, w) or not np.isfinite(known).all() or np.any((known < 0) | (known > 1)):
        raise ValueError('Known density must be finite in [0,1] and match the image')
    before = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY) if previous_gray is None else previous_gray
    after = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY) if current_gray is None else current_gray
    if before.shape != (h, w) or after.shape != (h, w) or before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError('Precomputed grayscale flow images must match RGB dimensions')
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    # 160@scale0 and 320@scale1 both use an 8px patch at an effective 160 grid.
    dis.setFinestScale(0 if max(h, w) <= 160 else 1 if max(h, w) <= 320 else 2)
    dis.setPatchSize(8)
    dis.setPatchStride(3)
    raw = dis.calc(before, after, None)
    yy, xx = np.mgrid[:h, :w]
    # Bound background fitting to a regular <=32x32 lattice, independently of resolution.
    ys = np.linspace(0, h - 1, min(h, 32)).astype(int)
    xs = np.linspace(0, w - 1, min(w, 32)).astype(int)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    valid = known[gy, gx] < .3
    points = np.c_[gx[valid] / w, gy[valid] / h].astype(np.float32)
    displacement = raw[gy[valid], gx[valid]] / [w, h]
    if len(points) < 6:
        return dict(supported=False, reason='insufficient_background_support', clouds=[],
                    proposal_union=np.zeros((h, w), bool), motion_mask=None)
    warp, inliers = cv2.estimateAffine2D(points, (points + displacement).astype(np.float32),
                                       method=cv2.RANSAC, ransacReprojThreshold=.003,
                                       maxIters=500, confidence=.99, refineIters=5)
    if warp is None or inliers is None or inliers.sum() < 6:
        return dict(supported=False, reason='background_fit_failed', clouds=[],
                    proposal_union=np.zeros((h, w), bool), motion_mask=None)
    x, y = xx / w, yy / h
    bg_x = (warp[0, 0] - 1) * x + warp[0, 1] * y + warp[0, 2]
    bg_y = warp[1, 0] * x + (warp[1, 1] - 1) * y + warp[1, 2]
    background = np.stack([bg_x, bg_y], axis=-1).astype(np.float32)
    residual = raw / np.array([w, h], np.float32) - background
    magnitude = np.linalg.norm(residual, axis=-1)
    # Flow outside image support is not an observation.
    support = ((xx + raw[..., 0] >= 0) & (xx + raw[..., 0] <= w - 1)
               & (yy + raw[..., 1] >= 0) & (yy + raw[..., 1] <= h - 1))
    motion = (magnitude > v_min) & support
    explained = explained_density(residual, background, known_regions, sigma_flow)
    energy = magnitude * motion * (1 - explained)
    binary = (energy > v_min).astype(np.uint8)
    radius = max(1, round(radius_frac * min(h, w)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    proposals = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        fraction = area / (h * w)
        if area < 3 or not min_area_frac <= fraction <= max_area_frac:
            continue
        region = labels[y:y + height, x:x + width] == i
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        compactness = min(1., 4 * np.pi * area / max(perimeter**2, 1.))
        saliency = float(energy[y:y + height, x:x + width][region].mean() * np.sqrt(fraction) * compactness)
        proposals.append(dict(area=int(area), centroid=centers[i] / [w, h],
                              box=np.array([x, y, x + width, y + height]) / [w, h, w, h],
                              saliency=saliency, component_id=i))
    clouds = proposal_nms(proposals, limit=max_proposals)
    union = np.zeros((h, w), bool)
    for proposal in clouds:
        region = labels == proposal.pop('component_id')
        union |= region
        cloud = measure(current, region)
        proposal.update(measure=cloud, scatter=cloud.shape_scatter, mask=region)
    return dict(supported=True, reason=None, warp=warp, inliers=int(inliers.sum()),
                raw_components=count - 1, eligible_components=len(proposals), clouds=clouds,
                proposal_union=union, motion_mask=motion, explained_density=explained,
                residual_flow=residual.astype(np.float32), finest_scale=dis.getFinestScale())
