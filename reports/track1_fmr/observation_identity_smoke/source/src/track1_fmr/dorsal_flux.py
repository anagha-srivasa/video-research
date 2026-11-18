from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np
from .contracts import CloudMeasure


def measure(rgb, mask, *, lab_moments=None):
    """Summarize a proposed region; depth, visibility and semantics are unknown."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.sum() < 3:
        raise ValueError('A nonempty 2D region is required')
    if lab_moments is None and (rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]):
        raise ValueError('Matching RGB or explicit Lab moments required')
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
    if lab_moments is None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(float)
        lab_moments = np.r_[lab.mean(0), lab.std(0)]
    lab_moments = np.asarray(lab_moments, float)
    if lab_moments.shape != (6,) or not np.isfinite(lab_moments).all():
        raise ValueError('Six finite Lab moments required')
    signature = np.zeros(24, np.float32)
    signature[:4] = np.hypot(harmonic[1:5], harmonic[5:9])
    signature[4:10] = lab_moments
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
            proposals.append(dict(mask=mask.astype(bool), saliency=float(confidence) * np.sqrt(area),
                                  box=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / 160))
        kept = proposal_nms(proposals)
        for proposal in kept:
            color = extract_proposal_lab_moments(rgb, proposal['box'], proposal['mask'])
            proposal['measure'] = measure(None, proposal['mask'], lab_moments=color)
        return kept


def static_heartbeat(rgb):
    """Compatibility entry point; no edge fallback or hidden model downloads."""
    return [p['measure'] for p in PromptFreeHeartbeat()(rgb)]


def preprocess_gray_pyramid(rgb_native, grid_h, grid_w):
    """Binomial 2x decimation followed by linear sampling; no RGB grid allocation."""
    if (rgb_native.dtype != np.uint8 or rgb_native.ndim != 3 or rgb_native.shape[2] != 3
            or grid_h < 32 or grid_w < 32 or grid_h > rgb_native.shape[0] or grid_w > rgb_native.shape[1]):
        raise ValueError('uint8 RGB and non-upsampling target dimensions >=32 required')
    gray = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2GRAY)
    while gray.shape[1] // 2 >= grid_w and gray.shape[0] // 2 >= grid_h:
        gray = cv2.pyrDown(gray)
    if gray.shape != (grid_h, grid_w):
        gray = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    return gray


def build_local_super_gaussian_prior(centroid_norm, shape_scatter_norm, grid_h, grid_w, sigma_cutoff=1.8, inv_shape_scatter=None):
    """Finite 1.8-sigma bbox, exp(-(Mahalanobis distance squared)^2/32).

    The footprint is a shape-moment heuristic, not a physical-boundary certificate.
    """
    from .belief_filter import _cov
    centre = np.asarray(centroid_norm, float)
    scatter = _cov(shape_scatter_norm, 2, positive=True)
    if centre.shape != (2,) or not np.isfinite(centre).all() or not np.isfinite(sigma_cutoff) or sigma_cutoff <= 0 or min(grid_h, grid_w) < 1:
        raise ValueError('Finite centre, SPD shape, positive cutoff and grid required')
    extent = sigma_cutoff * np.sqrt(np.maximum(np.diag(scatter), 1e-8))
    x0, y0 = np.clip(np.floor((centre - extent) * [grid_w, grid_h]), 0, [grid_w, grid_h]).astype(int)
    x1, y1 = np.clip(np.ceil((centre + extent) * [grid_w, grid_h]) + 1, 0, [grid_w, grid_h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return slice(0, 0), slice(0, 0), np.empty((0, 0), np.float32)
    xs = (np.arange(x0, x1, dtype=np.float32) + .5) / grid_w - centre[0]
    ys = (np.arange(y0, y1, dtype=np.float32) + .5) / grid_h - centre[1]
    gx, gy = np.meshgrid(xs, ys)
    if inv_shape_scatter is None:
        inverse = np.linalg.inv(scatter.astype(np.float32) + np.eye(2, dtype=np.float32) * 1e-8)
    else:
        inverse = np.asarray(inv_shape_scatter, np.float32)
        if inverse.shape != (2, 2) or not np.isfinite(inverse).all():
            raise ValueError('Finite cached 2x2 inverse required; caller must bind it to the current shape')
    distance_squared = inverse[0, 0] * gx**2 + (inverse[0, 1] + inverse[1, 0]) * gx * gy + inverse[1, 1] * gy**2
    return slice(y0, y1), slice(x0, x1), np.exp(-.03125 * distance_squared**2).astype(np.float32)


def extract_proposal_lab_moments(rgb_native, box_norm, mask_grid):
    """At most 32^2 actual foreground sample positions, cropped native RGB only.

    Sampling mask pixels directly avoids inventing background color when thin
    foreground support disappears in a resized mask. OpenCV uint8 Lab encoding.
    """
    box = np.asarray(box_norm, float)
    mask = np.asarray(mask_grid, bool)
    if (rgb_native.dtype != np.uint8 or rgb_native.ndim != 3 or rgb_native.shape[2] != 3
            or mask.ndim != 2 or box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box < 0) or np.any(box > 1) or np.any(box[2:] <= box[:2])):
        raise ValueError('Native uint8 RGB, normalized box and 2D region mask required')
    gh, gw = mask.shape
    nh, nw = rgb_native.shape[:2]
    gx0, gy0 = np.floor(box[:2] * [gw, gh]).astype(int)
    gx1, gy1 = np.minimum([gw, gh], np.ceil(box[2:] * [gw, gh]).astype(int))
    ys, xs = np.nonzero(mask[gy0:gy1, gx0:gx1])
    if not len(xs):
        raise ValueError('No foreground inside proposal crop')
    if len(xs) > 1024:
        selected = np.linspace(0, len(xs) - 1, 1024).astype(int)
        ys, xs = ys[selected], xs[selected]
    nx0, ny0 = np.floor(box[:2] * [nw, nh]).astype(int)
    nx1, ny1 = np.minimum([nw, nh], np.ceil(box[2:] * [nw, nh]).astype(int))
    crop = rgb_native[ny0:ny1, nx0:nx1]
    nx = np.clip(((xs + gx0 + .5) * nw / gw).astype(int) - nx0, 0, crop.shape[1] - 1)
    ny = np.clip(((ys + gy0 + .5) * nh / gh).astype(int) - ny0, 0, crop.shape[0] - 1)
    colors = crop[ny, nx]
    side = int(np.ceil(np.sqrt(len(colors))))
    tile = np.zeros((side, side, 3), np.uint8)
    tile.reshape(-1, 3)[:len(colors)] = colors
    lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB).reshape(-1, 3)[:len(colors)].astype(float)
    return np.r_[lab.mean(0), lab.std(0)].astype(np.float32)


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
        slices = region.get('slices', (slice(None), slice(None)))
        target = explained[slices]
        if (density.shape != target.shape or displacement.shape != (2,)
                or not np.isfinite(density).all() or not np.isfinite(displacement).all()
                or np.any((density < 0) | (density > 1))):
            raise ValueError('Finite density in [0,1] and image displacement required')
        ys, xs = np.nonzero(density)
        delta = image_flow[slices][ys, xs] - displacement
        weight = np.exp(-np.sum(delta**2, axis=-1) / (2 * sigma_flow**2))
        target[ys, xs] = np.maximum(target[ys, xs], density[ys, xs] * weight)
    return explained


@dataclass(frozen=True)
class TrackPriorMask:
    """Active track state projected onto the dorsal grid for velocity-consistent exclusion."""
    track_id: int
    density: np.ndarray          # (grid_h, grid_w) float32 in [0, 1]
    vel_norm: tuple[float, float]  # IMAGE-coordinate displacement / frame (velocity/sec * exact dt)
    offset_yx: tuple[int, int] = (0, 0)
    grid_hw: tuple[int, int] | None = None

    @property
    def slices(self):
        y, x = self.offset_yx
        return slice(y, y + self.density.shape[0]), slice(x, x + self.density.shape[1])

    def __post_init__(self):
        density = np.array(self.density, dtype=np.float32, copy=True)
        velocity = np.asarray(self.vel_norm, dtype=float)
        if (density.ndim != 2 or not np.isfinite(density).all() or np.any((density < 0) | (density > 1))
                or velocity.shape != (2,) or not np.isfinite(velocity).all()):
            raise ValueError('Finite 2D density in [0,1] and image displacement/frame required')
        grid = self.grid_hw or density.shape
        y, x = self.offset_yx
        if (len(grid) != 2 or min(grid) < 1 or any(not isinstance(v, (int, np.integer)) for v in (*grid, y, x))
                or min(y, x) < 0 or y + density.shape[0] > grid[0] or x + density.shape[1] > grid[1]):
            raise ValueError('Prior patch must be bounded inside its full grid')
        object.__setattr__(self, 'grid_hw', tuple(grid))
        density.setflags(write=False)
        object.__setattr__(self, 'density', density)
        object.__setattr__(self, 'vel_norm', tuple(float(v) for v in velocity))


@dataclass(frozen=True)
class FluxProposal:
    """Single blind motion proposal extracted from ego-compensated residual flow."""
    box_norm: tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0, 1]
    mask_grid: np.ndarray                        # (grid_h, grid_w) bool
    area_frac: float
    mean_novel_energy: float
    mean_vel_norm: tuple[float, float]
    compactness: float
    saliency_score: float
    mean_image_vel_norm: tuple[float, float]  # Residual plus fitted camera displacement, for image-state birth


@dataclass(frozen=True)
class FluxStepResult:
    """Source-separated telemetry and emitted proposals for one frame step."""
    grid_hw: tuple[int, int]
    preprocess_ms: float
    discovery_ms: float
    residual_flow_norm: np.ndarray   # (grid_h, grid_w, 2) float32
    raw_motion_mask: np.ndarray      # (grid_h, grid_w) bool (||u_res|| > v_min)
    explained_density: np.ndarray    # (grid_h, grid_w) float32 in [0, 1]
    novel_energy: np.ndarray         # (grid_h, grid_w) float32
    emitted_union_mask: np.ndarray   # (grid_h, grid_w) bool (union of kept proposals)
    proposals: list[FluxProposal]
    raw_active_frac: float
    novel_active_frac: float
    ransac_inliers: int
    proposals_before_nms: int
    background_flow_norm: np.ndarray
    background_mode: str
    background_lattice_eligible: int = 0
    background_fit_samples: int = 0
    effective_v_min: float = 0.
    bg_speed_median: float = 0.
    border_rejected: int = 0
    proposals_after_merge: int = 0


class DorsalFluxExtractor:
    """
    Stage 1 v3 Blind Dorsal Front-End:
    - Two-stage SIMD anti-aliased pyramid decimation
    - Scale-matched OpenCV DIS Optical Flow
    - Normalized RANSAC affine background ego-motion subtraction
    - Velocity-consistent track exclusion (prevents static heartbeat self-suppression)
    - Saliency-ranked, NMS-filtered Top-K proposal extraction
    """

    def __init__(
        self,
        dorsal_grid: int | None = 160,
        v_min: float = 0.0035,
        sigma_flow: float = 0.005,
        min_area_frac: float = 0.00045,
        max_area_frac: float = 0.40,
        radius_frac: float = 0.012,
        nms_iou_thresh: float = 0.50,
        max_proposals: int = 10,
        ego_speed_scale: float = .12,
        merge_pad_norm: float = .02,
        border_margin_norm: float = .025,
        lattice_steps: int = 32,
        min_saliency: float = 2.5e-5,
        core_density_thresh: float = .92,
        ransac_reproj_norm: float = 0.004,
    ) -> None:
        if dorsal_grid is not None and (isinstance(dorsal_grid, bool) or not isinstance(dorsal_grid, int) or dorsal_grid < 32):
            raise ValueError('dorsal_grid must be an integer >=32 or None')
        if (not np.isfinite([v_min, sigma_flow, min_area_frac, max_area_frac, radius_frac,
                             nms_iou_thresh, ransac_reproj_norm, min_saliency, core_density_thresh, ego_speed_scale, merge_pad_norm, border_margin_norm]).all()
                or v_min <= 0 or sigma_flow <= 0 or not 0 < min_area_frac <= max_area_frac <= 1
                or radius_frac < 0 or not 0 <= nms_iou_thresh <= 1 or ransac_reproj_norm <= 0
                or min_saliency < 0 or not 0 < core_density_thresh < 1
                or ego_speed_scale < 0 or not 0 <= merge_pad_norm <= 1 or not 0 <= border_margin_norm < .5
                or not isinstance(max_proposals, int) or isinstance(max_proposals, bool) or max_proposals < 1
                or not isinstance(lattice_steps, int) or isinstance(lattice_steps, bool) or lattice_steps < 4):
            raise ValueError('Invalid extractor configuration')
        self.dorsal_grid = dorsal_grid
        self.v_min = float(v_min)
        self.sigma_flow = float(sigma_flow)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.radius_frac = float(radius_frac)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.max_proposals = int(max_proposals)
        self.lattice_steps = int(lattice_steps)
        self.min_saliency = float(min_saliency)
        self.core_density_thresh = float(core_density_thresh)
        self.ego_speed_scale = float(ego_speed_scale)
        self.merge_pad_norm = float(merge_pad_norm)
        self.border_margin_norm = float(border_margin_norm)
        self._background_stats = {}
        self.ransac_reproj_norm = float(ransac_reproj_norm)

        self._prev_gray_grid: np.ndarray | None = None
        self._prev_grid_hw: tuple[int, int] | None = None
        self._dis: cv2.DISOpticalFlow | None = None
        self._dis_grid_hw: tuple[int, int] | None = None

    def reset(self) -> None:
        self._prev_gray_grid = None
        self._prev_grid_hw = None

    def _target_hw(self, native_h: int, native_w: int) -> tuple[int, int]:
        if self.dorsal_grid is None:
            return (native_h, native_w)
        return (self.dorsal_grid, self.dorsal_grid)

    def _ensure_dis(self, grid_h: int, grid_w: int) -> cv2.DISOpticalFlow:
        if self._dis is not None and self._dis_grid_hw == (grid_h, grid_w):
            return self._dis
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        # Scale-match physical receptive field relative to 160x160 baseline
        scale_ratio = max(grid_w, grid_h) / 160.0
        if scale_ratio <= 1.25:
            finest_scale = 0
        elif scale_ratio <= 2.5:
            finest_scale = 1
        else:
            finest_scale = max(2, int(round(np.log2(scale_ratio))))
        dis.setFinestScale(finest_scale)
        dis.setPatchSize(8)
        dis.setPatchStride(3)
        self._dis = dis
        self._dis_grid_hw = (grid_h, grid_w)
        return dis

    def preprocess_frame(self, rgb_native: np.ndarray) -> tuple[np.ndarray, None]:
        """Grayscale pyrDown/linear pyramid; color is sampled only for kept proposals."""
        if (rgb_native.dtype != np.uint8 or rgb_native.ndim != 3 or rgb_native.shape[2] != 3
                or min(rgb_native.shape[:2]) < 32):
            raise ValueError('Native uint8 RGB image with dimensions >=32 required')
        nat_h, nat_w = rgb_native.shape[:2]
        grid_h, grid_w = self._target_hw(nat_h, nat_w)
        return preprocess_gray_pyramid(rgb_native, grid_h, grid_w), None

    def _estimate_ego_residual(
        self,
        flow_norm: np.ndarray,
        known_spatial_mask: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray, str]:
        """Fit outside the tight density core; foreground-inclusive fallbacks stay diagnostic."""
        grid_h, grid_w = flow_norm.shape[:2]
        ys = np.linspace(.03, .97, self.lattice_steps, dtype=np.float32)
        xs = np.linspace(.03, .97, self.lattice_steps, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        ix = np.clip((gx * grid_w).astype(int), 0, grid_w - 1)
        iy = np.clip((gy * grid_h).astype(int), 0, grid_h - 1)
        # Use actual sampled pixel centers for both fitting and affine evaluation.
        gx = (ix.astype(np.float32) + .5) / grid_w
        gy = (iy.astype(np.float32) + .5) / grid_h
        valid = known_spatial_mask[iy, ix] <= self.core_density_thresh
        valid_count = int(valid.sum())
        if valid_count >= 16:
            points = np.stack([gx[valid], gy[valid]], axis=-1)
            displacement = flow_norm[iy[valid], ix[valid]]
            mode = 'bg_lattice'
        else:
            non_y, non_x = np.where(known_spatial_mask <= self.core_density_thresh)
            if len(non_x) >= 16:
                indices = np.linspace(0, len(non_x) - 1, min(256, len(non_x))).astype(int)
                x, y = non_x[indices], non_y[indices]
                points = np.c_[(x + .5) / grid_w, (y + .5) / grid_h].astype(np.float32)
                displacement = flow_norm[y, x]
                mode = 'bg_sparse_pixels'
            else:
                points = np.stack([gx.ravel(), gy.ravel()], axis=-1)
                displacement = flow_norm[iy.ravel(), ix.ravel()]
                mode = 'all_pixels_fallback'
        self._background_stats = dict(lattice_eligible=valid_count, fit_samples=len(points))
        affine, inliers = cv2.estimateAffine2D(points.reshape(-1, 1, 2),
            (points + displacement).reshape(-1, 1, 2), method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_norm, maxIters=500, confidence=.99)
        if affine is None:
            background = np.broadcast_to(np.median(displacement, axis=0).astype(np.float32), flow_norm.shape)
            return flow_norm - background, 0, background, 'median_fallback'
        full_x = (np.arange(grid_w, dtype=np.float32) + .5) / grid_w
        full_y = (np.arange(grid_h, dtype=np.float32) + .5) / grid_h
        x, y = np.meshgrid(full_x, full_y)
        background = np.stack([(affine[0, 0] - 1) * x + affine[0, 1] * y + affine[0, 2],
                               affine[1, 0] * x + (affine[1, 1] - 1) * y + affine[1, 2]], axis=-1).astype(np.float32)
        return flow_norm - background, int(inliers.sum()) if inliers is not None else 0, background, mode

    def _is_border_strip(self, box):
        return (box[2] <= self.border_margin_norm or box[0] >= 1 - self.border_margin_norm
                or box[3] <= self.border_margin_norm or box[1] >= 1 - self.border_margin_norm)

    @staticmethod
    def _padded_boxes_intersect(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
        pad: float,
    ) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (
            (ax2 + pad) < (bx1 - pad)
            or (bx2 + pad) < (ax1 - pad)
            or (ay2 + pad) < (by1 - pad)
            or (by2 + pad) < (ay1 - pad)
        )

    def _merge_comoving_proposals(
        self,
        proposals: list[FluxProposal],
        grid_h: int,
        grid_w: int,
    ) -> list[FluxProposal]:
        """
        Greedily merges spatially adjacent flux fragments that share consistent
        residual velocity vectors (e.g., articulated arm/utensil fragments)
        subject to max_area_frac before NMS and top-K capping.
        """
        if len(proposals) <= 1:
            return proposals

        ordered = sorted(proposals, key=lambda p: p.saliency_score, reverse=True)
        merged: list[FluxProposal] = []
        grid_area = float(grid_h * grid_w)

        for cand in ordered:
            did_merge = False
            cvx, cvy = cand.mean_vel_norm
            c_speed = float(np.hypot(cvx, cvy))

            for idx, base in enumerate(merged):
                if (base.area_frac + cand.area_frac) > self.max_area_frac:
                    continue
                if not self._padded_boxes_intersect(base.box_norm, cand.box_norm, self.merge_pad_norm):
                    continue

                bvx, bvy = base.mean_vel_norm
                b_speed = float(np.hypot(bvx, bvy))
                vel_diff = float(np.hypot(cvx - bvx, cvy - bvy))
                vel_tol = max(self.sigma_flow, 0.35 * max(c_speed, b_speed))
                if vel_diff > vel_tol:
                    continue

                union_mask = np.logical_or(base.mask_grid, cand.mask_grid)
                area_px = int(union_mask.sum())
                area_frac = area_px / grid_area
                if area_frac > self.max_area_frac:
                    continue

                w_base = base.area_frac
                w_cand = cand.area_frac
                w_tot = max(w_base + w_cand, 1e-12)

                mean_energy = float((base.mean_novel_energy * w_base + cand.mean_novel_energy * w_cand) / w_tot)
                mean_vx = float((bvx * w_base + cvx * w_cand) / w_tot)
                mean_vy = float((bvy * w_base + cvy * w_cand) / w_tot)
                img_vx = float((base.mean_image_vel_norm[0] * w_base + cand.mean_image_vel_norm[0] * w_cand) / w_tot)
                img_vy = float((base.mean_image_vel_norm[1] * w_base + cand.mean_image_vel_norm[1] * w_cand) / w_tot)

                ux1 = min(base.box_norm[0], cand.box_norm[0])
                uy1 = min(base.box_norm[1], cand.box_norm[1])
                ux2 = max(base.box_norm[2], cand.box_norm[2])
                uy2 = max(base.box_norm[3], cand.box_norm[3])

                gx0 = max(0, int(np.floor(ux1 * grid_w)))
                gy0 = max(0, int(np.floor(uy1 * grid_h)))
                gx1 = min(grid_w, max(gx0 + 1, int(np.ceil(ux2 * grid_w))))
                gy1 = min(grid_h, max(gy0 + 1, int(np.ceil(uy2 * grid_h))))
                sub_u8 = union_mask[gy0:gy1, gx0:gx1].astype(np.uint8)
                contours, _ = cv2.findContours(sub_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                perim = float(sum(cv2.arcLength(c, closed=True) for c in contours))
                compactness = float(np.clip((4.0 * np.pi * area_px) / max(perim * perim, 1.0), 0.05, 1.0))
                saliency = mean_energy * float(np.sqrt(area_frac)) * compactness

                merged[idx] = FluxProposal(
                    box_norm=(ux1, uy1, ux2, uy2),
                    mask_grid=union_mask,
                    area_frac=area_frac,
                    mean_novel_energy=mean_energy,
                    mean_vel_norm=(mean_vx, mean_vy),
                    mean_image_vel_norm=(img_vx, img_vy),
                    compactness=compactness,
                    saliency_score=saliency,
                )
                did_merge = True
                break

            if not did_merge:
                merged.append(cand)

        return merged


    @staticmethod
    def _nms_boxes(proposals: list[FluxProposal], iou_thresh: float, max_keep: int) -> list[FluxProposal]:
        if not proposals:
            return []
        ordered = sorted(proposals, key=lambda p: p.saliency_score, reverse=True)
        kept: list[FluxProposal] = []
        for cand in ordered:
            if len(kept) >= max_keep:
                break
            cx1, cy1, cx2, cy2 = cand.box_norm
            c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            suppress = False
            for prev in kept:
                px1, py1, px2, py2 = prev.box_norm
                ix1 = max(cx1, px1)
                iy1 = max(cy1, py1)
                ix2 = min(cx2, px2)
                iy2 = min(cy2, py2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
                union = c_area + p_area - inter
                if union > 1e-9 and (inter / union) > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append(cand)
        return kept

    def step(
        self,
        rgb_native: np.ndarray,
        active_tracks: Sequence[TrackPriorMask] = (),
    ) -> tuple[FluxStepResult, None]:
        """
        Processes one native RGB frame and returns (FluxStepResult, None); no RGB grid.
        On the first frame after reset(), initializes previous buffer and returns empty proposals.
        """
        t0 = cv2.getTickCount()
        gray_grid, rgb_grid = self.preprocess_frame(rgb_native)
        grid_h, grid_w = gray_grid.shape[:2]
        if any(t.grid_hw != (grid_h, grid_w) for t in active_tracks):
            raise ValueError('Track densities must match the current grid')
        t1 = cv2.getTickCount()
        preprocess_ms = (t1 - t0) * 1000.0 / cv2.getTickFrequency()

        if self._prev_gray_grid is None or self._prev_grid_hw != (grid_h, grid_w):
            self._prev_gray_grid = gray_grid
            self._prev_grid_hw = (grid_h, grid_w)
            empty_flow = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
            empty_bool = np.zeros((grid_h, grid_w), dtype=bool)
            empty_f32 = np.zeros((grid_h, grid_w), dtype=np.float32)
            res = FluxStepResult(
                grid_hw=(grid_h, grid_w),
                preprocess_ms=preprocess_ms,
                discovery_ms=0.0,
                residual_flow_norm=empty_flow,
                raw_motion_mask=empty_bool,
                explained_density=empty_f32,
                novel_energy=empty_f32,
                emitted_union_mask=empty_bool,
                proposals=[],
                raw_active_frac=0.0,
                novel_active_frac=0.0,
                ransac_inliers=0, proposals_before_nms=0,
                background_flow_norm=empty_flow, background_mode='initialization',
            )
            return res, rgb_grid

        result = self._analyze(self._prev_gray_grid, gray_grid, active_tracks, preprocess_ms)
        self._prev_gray_grid = gray_grid
        return result, rgb_grid

    def _analyze(self, previous_gray, gray_grid, active_tracks, preprocess_ms=0.):
        grid_h, grid_w = gray_grid.shape
        if any(t.grid_hw != (grid_h, grid_w) for t in active_tracks):
            raise ValueError('Track densities must match the current grid')
        t2 = cv2.getTickCount()
        dis = self._ensure_dis(grid_h, grid_w)
        flow_px = dis.calc(previous_gray, gray_grid, None)
        flow_norm = np.empty_like(flow_px, dtype=np.float32)
        flow_norm[..., 0] = flow_px[..., 0] / float(grid_w)
        flow_norm[..., 1] = flow_px[..., 1] / float(grid_h)

        # Build spatial known mask for background RANSAC exclusion
        known_spatial = np.zeros((grid_h, grid_w), dtype=np.float32)
        for trk in active_tracks:
            patch = known_spatial[trk.slices]
            np.maximum(patch, trk.density, out=patch)

        self._background_stats = {}
        residual_flow, inliers, background_flow, background_mode = self._estimate_ego_residual(flow_norm, known_spatial)
        speed_norm = np.linalg.norm(residual_flow, axis=-1)
        yy, xx = np.mgrid[:grid_h, :grid_w]
        support = ((xx + flow_px[..., 0] >= 0) & (xx + flow_px[..., 0] <= grid_w - 1)
                   & (yy + flow_px[..., 1] >= 0) & (yy + flow_px[..., 1] <= grid_h - 1))
        bg_speed_median = float(np.median(np.linalg.norm(background_flow, axis=-1)))
        v_eff = self.v_min + self.ego_speed_scale * bg_speed_median
        raw_motion_mask = (speed_norm > v_eff) & support

        # Velocity-consistent exclusion: suppress only regions where track velocity matches flow
        # Track displacements are image-coordinate; remove the same camera field
        # before residual comparison (equivalently compare raw image flow to them).
        explained = explained_density(residual_flow, background_flow,
            [dict(density=t.density, displacement=t.vel_norm, slices=t.slices) for t in active_tracks], self.sigma_flow)
        novel_energy = np.where(
            raw_motion_mask,
            speed_norm * np.clip(1.0 - explained, 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        novel_binary = (novel_energy > (0.85 * v_eff)).astype(np.uint8)
        morph_r = max(1, int(round(self.radius_frac * min(grid_h, grid_w))))
        k_size = 2 * morph_r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        novel_binary = cv2.morphologyEx(novel_binary, cv2.MORPH_CLOSE, kernel)
        # Re-apply explained-track gate after morphological closing
        novel_binary[explained > 0.65] = 0

        novel_binary[~support] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(novel_binary, connectivity=8)
        grid_area = float(grid_h * grid_w)
        min_pixels = max(4, int(np.ceil(self.min_area_frac * grid_area)))
        max_pixels = int(np.floor(self.max_area_frac * grid_area))

        raw_proposals: list[FluxProposal] = []
        border_rejected = 0
        for lbl in range(1, num_labels):
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < min_pixels or area_px > max_pixels:
                continue

            x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
            y0 = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            comp_mask = labels == lbl
            area_frac = area_px / grid_area
            mean_energy = float(novel_energy[comp_mask].mean())
            if mean_energy <= 0.0:
                continue

            # Perimeter-based compactness on the local crop
            sub_u8 = comp_mask[y0 : y0 + bh, x0 : x0 + bw].astype(np.uint8)
            contours, _ = cv2.findContours(sub_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perim = float(sum(cv2.arcLength(c, closed=True) for c in contours))
            compactness = float(np.clip((4.0 * np.pi * area_px) / max(perim * perim, 1.0), 0.05, 1.0))

            saliency = mean_energy * float(np.sqrt(area_frac)) * compactness
            if saliency < self.min_saliency:
                continue
            mean_vx = float(residual_flow[..., 0][comp_mask].mean(dtype=np.float64))
            mean_vy = float(residual_flow[..., 1][comp_mask].mean(dtype=np.float64))

            box_norm = (
                x0 / float(grid_w),
                y0 / float(grid_h),
                (x0 + bw) / float(grid_w),
                (y0 + bh) / float(grid_h),
            )
            if self._is_border_strip(box_norm):
                border_rejected += 1
                continue
            raw_proposals.append(
                FluxProposal(
                    box_norm=box_norm,
                    mask_grid=comp_mask,
                    area_frac=area_frac,
                    mean_novel_energy=mean_energy,
                    mean_vel_norm=(mean_vx, mean_vy),
                    compactness=compactness,
                    saliency_score=saliency,
                    mean_image_vel_norm=tuple(float(v) for v in flow_norm[comp_mask].mean(axis=0, dtype=np.float64)),
                )
            )

        merged_proposals = self._merge_comoving_proposals(raw_proposals, grid_h, grid_w)
        kept_proposals = self._nms_boxes(merged_proposals, self.nms_iou_thresh, self.max_proposals)
        if background_mode not in ('affine', 'bg_lattice', 'bg_sparse_pixels'):
            kept_proposals = []  # Weak fallbacks are diagnostic, not accepted ego-compensated observations.
        emitted_union = np.zeros((grid_h, grid_w), dtype=bool)
        for prop in kept_proposals:
            np.logical_or(emitted_union, prop.mask_grid, out=emitted_union)

        t3 = cv2.getTickCount()
        discovery_ms = (t3 - t2) * 1000.0 / cv2.getTickFrequency()

        res = FluxStepResult(
            grid_hw=(grid_h, grid_w),
            preprocess_ms=preprocess_ms,
            discovery_ms=discovery_ms,
            residual_flow_norm=residual_flow,
            raw_motion_mask=raw_motion_mask,
            explained_density=explained,
            novel_energy=novel_energy,
            emitted_union_mask=emitted_union,
            proposals=kept_proposals,
            raw_active_frac=float(raw_motion_mask.mean()),
            novel_active_frac=float(emitted_union.mean()),
            ransac_inliers=inliers, proposals_before_nms=len(raw_proposals),
            background_flow_norm=background_flow, background_mode=background_mode,
            background_lattice_eligible=self._background_stats.get('lattice_eligible', 0),
            background_fit_samples=self._background_stats.get('fit_samples', 0),
            effective_v_min=v_eff, bg_speed_median=bg_speed_median,
            border_rejected=border_rejected, proposals_after_merge=len(merged_proposals),
        )
        return res


def anti_alias(rgb, size=None, *, grayscale=False):
    """Compatibility helper for the stateful extractor's preprocessing."""
    gray, _ = DorsalFluxExtractor(dorsal_grid=size).preprocess_frame(rgb)
    if grayscale:
        return gray
    # Compatibility-only RGB helper; active ingestion never allocates an RGB grid.
    return rgb if size is None else cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def as_legacy_result(result, rgb_native):
    """Existing cloud consumers share the stateful kernel and moment implementation."""
    clouds = []
    for proposal in result.proposals:
        lab = extract_proposal_lab_moments(rgb_native, proposal.box_norm, proposal.mask_grid)
        cloud = measure(None, proposal.mask_grid, lab_moments=lab)
        clouds.append(dict(measure=cloud, scatter=cloud.shape_scatter, mask=proposal.mask_grid,
                           box=np.asarray(proposal.box_norm), saliency=proposal.saliency_score,
                           mean_vel_norm=proposal.mean_vel_norm, mean_image_vel_norm=proposal.mean_image_vel_norm))
    return dict(supported=result.background_mode in ('affine', 'bg_lattice', 'bg_sparse_pixels'), reason=result.background_mode,
                clouds=clouds, proposal_union=result.emitted_union_mask, motion_mask=result.raw_motion_mask,
                residual_flow=result.residual_flow_norm, explained_density=result.explained_density,
                novel_energy=result.novel_energy, inliers=result.ransac_inliers,
                background_mode=result.background_mode, preprocess_ms=result.preprocess_ms,
                discovery_ms=result.discovery_ms)


def flux(previous, current, excluded=None, v_min=.0035, min_area_frac=.00045,
         max_area_frac=.40, radius_frac=.012, max_proposals=10,
         previous_gray=None, current_gray=None, known_regions=(), sigma_flow=.005):
    """Legacy preprocessed interface; active Pipeline uses DorsalFluxExtractor directly."""
    if previous.shape != current.shape or previous.dtype != np.uint8 or current.dtype != np.uint8:
        raise ValueError('Matching uint8 RGB frames required')
    extractor = DorsalFluxExtractor(None, v_min=v_min, sigma_flow=sigma_flow,
                                   min_area_frac=min_area_frac, max_area_frac=max_area_frac,
                                   radius_frac=radius_frac, max_proposals=max_proposals)
    if excluded is not None:
        known = np.asarray(excluded)
        if known.shape != current.shape[:2] or not np.isfinite(known).all() or np.any((known < 0) | (known > 1)):
            raise ValueError('Valid spatial density required')
    before = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY) if previous_gray is None else previous_gray
    after = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY) if current_gray is None else current_gray
    if before.shape != previous.shape[:2] or after.shape != current.shape[:2] or before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError('Precomputed grayscale dimensions/dtype mismatch')
    tracks = [TrackPriorMask(i, r['density'], tuple(r['displacement'])) for i, r in enumerate(known_regions)]
    result = extractor._analyze(before, after, tracks)
    value = as_legacy_result(result, current)
    value['finest_scale'] = extractor._dis.getFinestScale()
    return value
