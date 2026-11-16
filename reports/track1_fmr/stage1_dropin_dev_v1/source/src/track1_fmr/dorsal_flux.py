from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
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


@dataclass(frozen=True)
class TrackPriorMask:
    """Active track state projected onto the dorsal grid for velocity-consistent exclusion."""
    track_id: int
    density: np.ndarray          # (grid_h, grid_w) float32 in [0, 1]
    vel_norm: tuple[float, float]  # IMAGE-coordinate displacement / frame (velocity/sec * exact dt)

    def __post_init__(self):
        density = np.array(self.density, dtype=np.float32, copy=True)
        velocity = np.asarray(self.vel_norm, dtype=float)
        if (density.ndim != 2 or not np.isfinite(density).all() or np.any((density < 0) | (density > 1))
                or velocity.shape != (2,) or not np.isfinite(velocity).all()):
            raise ValueError('Finite 2D density in [0,1] and image displacement/frame required')
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
        min_area_frac: float = 0.00025,
        max_area_frac: float = 0.40,
        radius_frac: float = 0.012,
        nms_iou_thresh: float = 0.50,
        max_proposals: int = 10,
        lattice_steps: int = 24,
        ransac_reproj_norm: float = 0.004,
    ) -> None:
        if dorsal_grid is not None and (isinstance(dorsal_grid, bool) or not isinstance(dorsal_grid, int) or dorsal_grid < 32):
            raise ValueError('dorsal_grid must be an integer >=32 or None')
        if (not np.isfinite([v_min, sigma_flow, min_area_frac, max_area_frac, radius_frac,
                             nms_iou_thresh, ransac_reproj_norm]).all()
                or v_min <= 0 or sigma_flow <= 0 or not 0 < min_area_frac <= max_area_frac <= 1
                or radius_frac < 0 or not 0 <= nms_iou_thresh <= 1 or ransac_reproj_norm <= 0
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

    def preprocess_frame(self, rgb_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Two-stage SIMD anti-aliased decimation:
        Stage A: INTER_AREA box decimation to 2x target grid (avoids 1080p big-kernel blur)
        Stage B: 5x5 Gaussian anti-alias at 2x grid -> INTER_LINEAR to target grid
        Returns (gray_grid uint8, rgb_grid uint8).
        """
        if (rgb_native.dtype != np.uint8 or rgb_native.ndim != 3 or rgb_native.shape[2] != 3
                or min(rgb_native.shape[:2]) < 32):
            raise ValueError('Native uint8 RGB image with dimensions >=32 required')
        nat_h, nat_w = rgb_native.shape[:2]
        grid_h, grid_w = self._target_hw(nat_h, nat_w)
        if grid_h > nat_h or grid_w > nat_w:
            raise ValueError('Dorsal grid must not upsample either native axis')
        gray_nat = cv2.cvtColor(rgb_native, cv2.COLOR_RGB2GRAY)

        if (grid_h, grid_w) == (nat_h, nat_w):
            return gray_nat, rgb_native

        mid_w = min(nat_w, 2 * grid_w)
        mid_h = min(nat_h, 2 * grid_h)

        if (mid_w, mid_h) != (nat_w, nat_h):
            gray_2x = cv2.resize(gray_nat, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
        else:
            gray_2x = gray_nat

        sigma_x = max(0.5, 0.5 * (mid_w / float(grid_w)))
        sigma_y = max(0.5, 0.5 * (mid_h / float(grid_h)))
        gray_blur = cv2.GaussianBlur(gray_2x, (5, 5), sigmaX=sigma_x, sigmaY=sigma_y)
        gray_grid = cv2.resize(gray_blur, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
        rgb_grid = cv2.resize(rgb_native, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        return gray_grid, rgb_grid

    def _estimate_ego_residual(
        self,
        flow_norm: np.ndarray,
        known_spatial_mask: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray, str]:
        """
        Fits a 2D affine background motion field in normalized [0, 1]^2 coordinates
        on lattice points outside known foreground support and subtracts it.
        """
        grid_h, grid_w = flow_norm.shape[:2]
        ys = np.linspace(0.05, 0.95, self.lattice_steps, dtype=np.float32)
        xs = np.linspace(0.05, 0.95, self.lattice_steps, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)  # (L, L) in normalized coords

        ix = np.clip((gx * grid_w).astype(np.int32), 0, grid_w - 1)
        iy = np.clip((gy * grid_h).astype(np.int32), 0, grid_h - 1)

        gx = (ix.astype(np.float32) + .5) / grid_w
        gy = (iy.astype(np.float32) + .5) / grid_h
        bg_valid = known_spatial_mask[iy, ix] < 0.25
        mode = 'affine' if int(bg_valid.sum()) >= 12 else 'all_pixels_fallback'
        if int(bg_valid.sum()) >= 12:
            src_pts = np.stack([gx[bg_valid], gy[bg_valid]], axis=-1)
            du = flow_norm[iy[bg_valid], ix[bg_valid]]
        else:
            src_pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
            du = flow_norm[iy.reshape(-1), ix.reshape(-1)]

        dst_pts = src_pts + du
        affine_mat, inliers = cv2.estimateAffine2D(
            src_pts.reshape(-1, 1, 2),
            dst_pts.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_norm,
            maxIters=500,
            confidence=0.99,
        )

        if affine_mat is None:
            med_flow = np.median(du, axis=0).astype(np.float32)
            bg_flow = np.broadcast_to(med_flow.reshape(1, 1, 2), flow_norm.shape)
            return flow_norm - bg_flow, 0, np.asarray(bg_flow), 'median_fallback'

        # Evaluate predicted affine background displacement across full normalized grid
        full_x = (np.arange(grid_w, dtype=np.float32) + 0.5) / float(grid_w)
        full_y = (np.arange(grid_h, dtype=np.float32) + 0.5) / float(grid_h)
        fx, fy = np.meshgrid(full_x, full_y)

        pred_x = affine_mat[0, 0] * fx + affine_mat[0, 1] * fy + affine_mat[0, 2]
        pred_y = affine_mat[1, 0] * fx + affine_mat[1, 1] * fy + affine_mat[1, 2]
        bg_flow = np.stack([pred_x - fx, pred_y - fy], axis=-1).astype(np.float32)

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        return flow_norm - bg_flow, inlier_count, bg_flow, mode

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
    ) -> tuple[FluxStepResult, np.ndarray]:
        """
        Processes one native RGB frame and returns (FluxStepResult, rgb_grid).
        On the first frame after reset(), initializes previous buffer and returns empty proposals.
        """
        t0 = cv2.getTickCount()
        gray_grid, rgb_grid = self.preprocess_frame(rgb_native)
        grid_h, grid_w = gray_grid.shape[:2]
        if any(t.density.shape != (grid_h, grid_w) for t in active_tracks):
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
        if any(t.density.shape != (grid_h, grid_w) for t in active_tracks):
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
            np.maximum(known_spatial, trk.density, out=known_spatial)

        residual_flow, inliers, background_flow, background_mode = self._estimate_ego_residual(flow_norm, known_spatial)
        speed_norm = np.linalg.norm(residual_flow, axis=-1)
        yy, xx = np.mgrid[:grid_h, :grid_w]
        support = ((xx + flow_px[..., 0] >= 0) & (xx + flow_px[..., 0] <= grid_w - 1)
                   & (yy + flow_px[..., 1] >= 0) & (yy + flow_px[..., 1] <= grid_h - 1))
        raw_motion_mask = (speed_norm > self.v_min) & support

        # Velocity-consistent exclusion: suppress only regions where track velocity matches flow
        # Track displacements are image-coordinate; remove the same camera field
        # before residual comparison (equivalently compare raw image flow to them).
        explained = explained_density(residual_flow, background_flow,
            [dict(density=t.density, displacement=t.vel_norm) for t in active_tracks], self.sigma_flow)
        novel_energy = np.where(
            raw_motion_mask,
            speed_norm * np.clip(1.0 - explained, 0.0, 1.0),
            0.0,
        ).astype(np.float32)

        novel_binary = (novel_energy > (0.5 * self.v_min)).astype(np.uint8)
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
            mean_vx = float(residual_flow[..., 0][comp_mask].mean(dtype=np.float64))
            mean_vy = float(residual_flow[..., 1][comp_mask].mean(dtype=np.float64))

            box_norm = (
                x0 / float(grid_w),
                y0 / float(grid_h),
                (x0 + bw) / float(grid_w),
                (y0 + bh) / float(grid_h),
            )
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

        kept_proposals = self._nms_boxes(raw_proposals, self.nms_iou_thresh, self.max_proposals)
        if background_mode != 'affine':
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
        )
        return res


def anti_alias(rgb, size=None, *, grayscale=False):
    """Compatibility helper for the stateful extractor's preprocessing."""
    gray, color = DorsalFluxExtractor(dorsal_grid=size).preprocess_frame(rgb)
    return gray if grayscale else color


def as_legacy_result(result, rgb_grid):
    """Existing cloud consumers share the stateful kernel and moment implementation."""
    clouds = []
    for proposal in result.proposals:
        cloud = measure(rgb_grid, proposal.mask_grid)
        clouds.append(dict(measure=cloud, scatter=cloud.shape_scatter, mask=proposal.mask_grid,
                           box=np.asarray(proposal.box_norm), saliency=proposal.saliency_score,
                           mean_vel_norm=proposal.mean_vel_norm, mean_image_vel_norm=proposal.mean_image_vel_norm))
    return dict(supported=result.background_mode == 'affine', reason=result.background_mode,
                clouds=clouds, proposal_union=result.emitted_union_mask, motion_mask=result.raw_motion_mask,
                residual_flow=result.residual_flow_norm, explained_density=result.explained_density,
                novel_energy=result.novel_energy, inliers=result.ransac_inliers,
                background_mode=result.background_mode, preprocess_ms=result.preprocess_ms,
                discovery_ms=result.discovery_ms)


def flux(previous, current, excluded=None, v_min=.0035, min_area_frac=.00025,
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
