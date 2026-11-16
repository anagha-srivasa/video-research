"""Experimental image-to-tube pipeline; proposals do not certify objects."""
import time as clock
import cv2
import numpy as np
from .belief_filter import BeliefTracker, bind_anchor
from .dorsal_flux import DorsalFluxExtractor, TrackPriorMask, PromptFreeHeartbeat, proposal_nms, as_legacy_result, build_local_super_gaussian_prior
from .lebesgue_tubes import FunctionalTubeBuilder
from .junction_graph import candidates
from track1_encoder_revision.events import InnovationEvidence


class Pipeline:
    def __init__(self, heartbeat_seconds=2., max_tracks=40, carry_prior=False, dorsal_grid=None, heartbeat=None, v_min=.0035, sigma_flow=.005, acceleration_noise=.001, measurement_floor=0.):
        if heartbeat_seconds <= 0:
            raise ValueError('Heartbeat period must be positive')
        self.heartbeat = heartbeat if heartbeat is not None else PromptFreeHeartbeat()
        self.period = heartbeat_seconds
        if dorsal_grid is not None and (isinstance(dorsal_grid, bool) or not isinstance(dorsal_grid, int) or dorsal_grid < 32):
            raise ValueError('Dorsal grid must be an integer >= 32, or None for native resolution')
        if not np.isfinite(v_min) or v_min <= 0:
            raise ValueError('Positive normalized flow threshold required')
        if not np.isfinite(sigma_flow) or sigma_flow <= 0:
            raise ValueError('Positive flow displacement scale required')
        self.extractor = DorsalFluxExtractor(dorsal_grid=dorsal_grid, v_min=v_min, sigma_flow=sigma_flow)
        self.last_observed_ts = {}
        self.last_prior_ids = []
        self.last_time = None
        self.sigma_flow = float(sigma_flow)
        self.v_min = float(v_min)
        self.dorsal_grid = dorsal_grid
        self.carry_prior = bool(carry_prior)
        self.tracker = BeliefTracker(max_tracks=max_tracks, acceleration_noise=acceleration_noise, measurement_floor=measurement_floor)
        self.last_proposal_union = None
        self.last_observations = {}
        self.last_flux = None
        self.previous_gray = None
        self.previous = None
        self.last_heartbeat = -np.inf
        self.builders = {}
        self.tubes = []
        self.failures = []
        self.events = []
        self.evidence = {}

    def step(self, rgb, time):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Native uint8 RGB frame required')
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError('Native PTS must increase')
        dt = None if self.last_time is None else time - self.last_time
        height, width = self.extractor._target_hw(*rgb.shape[:2])
        prior_full_grid_pixels = len(self.tracker.tracks) * height * width
        prior_start = clock.perf_counter()
        active = []
        ranked = sorted(self.tracker.tracks.items(), key=lambda item:
            (self.last_observed_ts.get(item[0], -np.inf), item[1].status == 'OBSERVED', -item[0]), reverse=True)[:15]
        self.last_prior_ids = [track_id for track_id, _ in ranked]
        prior_predictions = self.tracker.predict_at(time, self.last_prior_ids)
        for track_id, prior in ranked:
            state = prior_predictions[track_id]
            ys, xs, density = build_local_super_gaussian_prior(state.mean[:2], state.shape_scatter, height, width,
                inv_shape_scatter=self.tracker.inverse_shape_scatter(track_id))
            if density.size:
                active.append(TrackPriorMask(track_id, density, tuple(state.mean[2:] * dt),
                                             offset_yx=(ys.start, xs.start), grid_hw=(height, width)))
        prior_render_ms = (clock.perf_counter() - prior_start) * 1000
        result, dorsal = self.extractor.step(rgb, active)
        self.last_flux = as_legacy_result(result, rgb)
        preprocessing_ms = result.preprocess_ms
        flux_ms = result.discovery_ms
        heartbeat_ms = 0.
        proposals = []
        flux_proposals = 0
        heartbeat_proposals = 0
        if self.last_flux['supported']:
            proposals.extend(dict(p, source='flux') for p in self.last_flux['clouds'])
            flux_proposals = len(self.last_flux['clouds'])
        elif result.background_mode != 'initialization':
            self.failures.append((float(time), result.background_mode))
        if time - self.last_heartbeat >= self.period:
            heartbeat_start = clock.perf_counter()
            anchors = self.heartbeat(rgb)
            heartbeat_ms = (clock.perf_counter() - heartbeat_start) * 1000
            proposals.extend(dict(p, source='heartbeat') for p in anchors)
            heartbeat_proposals = len(anchors)
            self.last_heartbeat = float(time)
        selected = proposal_nms(proposals)
        self.last_proposal_union = np.zeros((160, 160), bool)
        for proposal in selected:
            if 'mask' in proposal:
                mask = cv2.resize(proposal['mask'].astype(np.float32), (160, 160), interpolation=cv2.INTER_AREA)
                self.last_proposal_union |= mask >= .5
        clouds = [p['measure'] for p in selected]
        before = self.tracker.tracks.copy()
        flow_advection = {}
        if dt is not None and self.last_flux['supported']:
            image_flow = result.residual_flow_norm + result.background_flow_norm
            for prior in active:
                core = prior.density > .65
                vectors = image_flow[prior.slices][core]
                if len(vectors) >= 4 and np.isfinite(vectors).all():
                    flow_advection[prior.track_id] = (vectors.mean(axis=0, dtype=np.float64), np.eye(2) * self.sigma_flow**2)
        birth_kinematics = {}
        if dt is not None:
            for index, proposal in enumerate(selected):
                if proposal['source'] == 'flux':
                    # State centroids live in image coordinates, not camera-residual coordinates.
                    velocity = np.asarray(proposal['mean_image_vel_norm'], np.float64) / dt
                    covariance = np.eye(2) * (self.sigma_flow / dt)**2
                    birth_kinematics[index] = (velocity, covariance)
        visual_context = None
        if self.last_flux['supported']:
            visual_context = (self.extractor._prev_gray_grid, result.residual_flow_norm,
                              result.effective_v_min, max(self.tracker.measurement_floor, self.sigma_flow))
        updated = self.tracker.update(time, clouds, birth_kinematics=birth_kinematics,
                                      flow_advection=flow_advection, visual_context=visual_context)
        self.last_measurement_sources = updated['measurement_sources']
        self.last_observations = updated['clouds'].copy()
        for track_id in set(updated['evicted_working_set'] + updated['archived']):
            builder = self.builders.pop(track_id, None)
            if builder is not None:
                segment = builder.close()
                if segment is not None:
                    self.tubes.append(segment)
            self.evidence.pop(track_id, None)
            self.last_observed_ts.pop(track_id, None)
        patch_crops = 0
        for track_id in updated['clouds']:
            self.last_observed_ts[track_id] = float(time)
        for track_id, state in updated['tracks'].items():
            cloud = updated['clouds'].get(track_id)
            innovation = 0.
            if cloud is not None and track_id in before:
                pred = updated['predictions'][track_id]
                delta = cloud.centroid - pred.mean[:2]
                innovation = float(delta @ np.linalg.solve(pred.covariance[:2, :2] + cloud.obs_cov_c, delta))
            if cloud is not None:
                h, w = rgb.shape[:2]
                scale = np.array([w, h])
                detector = self.evidence.setdefault(track_id, InnovationEvidence())
                detector.update(time, cloud.centroid * scale, np.diag(cloud.obs_cov_c) * scale**2)
                self.events.extend(dict(track_id=track_id, **event) for event in detector.drain(time))
            builder = self.builders.setdefault(track_id, FunctionalTubeBuilder(track_id, carry_prior=self.carry_prior))
            patch = None
            quality = None if cloud is None else float(cloud.visibility_quality)
            if cloud is not None and builder.wants_patch(state, innovation, quality):
                h, w = rgb.shape[:2]
                centre = cloud.centroid * [w, h]
                extent = np.maximum(4, 3 * np.sqrt(np.diag(cloud.shape_scatter)) * [w, h])
                left, top = np.maximum(0, (centre - extent).astype(int))
                right, bottom = np.minimum([w, h], np.ceil(centre + extent).astype(int))
                if right > left and bottom > top:
                    patch = cv2.resize(rgb[top:bottom, left:right], (96, 96))
                    patch_crops += 1
            # No independent visibility/depth estimate: zero is an unknown-quality marker.
            segment = builder.add(state, cloud, innovation, patch, quality if patch is not None else None)
            if segment is not None:
                self.tubes.append(segment)
        self.previous_gray = self.extractor._prev_gray_grid
        self.last_time = float(time)
        self.previous = self.previous_gray
        return dict(observed=sum(s.status == 'OBSERVED' for s in updated['tracks'].values()),
                    coasting=sum(s.status == 'PREDICTED' for s in updated['tracks'].values()),
                    flow_advected=updated['flow_advected'], assignment_cost=updated['assignment_cost'],
                    covariance_calibrated=False,
                    tracks=len(updated['tracks']), rejected_capacity=updated['rejected_capacity'],
                    preprocessing_ms=preprocessing_ms, flux_ms=flux_ms, heartbeat_ms=heartbeat_ms,
                    prior_render_ms=prior_render_ms, prior_patch_pixels=sum(t.density.size for t in active),
                    rendered_priors=len(active), prior_budget=15,
                    max_tracks=self.tracker.max_tracks, evicted_working_set=len(updated['evicted_working_set']),
                    patch_crops=patch_crops, flux_proposal_limit=self.extractor.max_proposals,
                    prior_full_grid_pixels=prior_full_grid_pixels,
                    raw_active_frac=result.raw_active_frac, novel_active_frac=result.novel_active_frac,
                    background_mode=result.background_mode, ransac_inliers=result.ransac_inliers,
                    background_lattice_eligible=result.background_lattice_eligible,
                    background_fit_samples=result.background_fit_samples,
                    eligible_flux_proposals=result.proposals_before_nms,
                    merged_flux_proposals=result.proposals_after_merge, border_rejected=result.border_rejected,
                    bg_speed_median=result.bg_speed_median, effective_v_min=result.effective_v_min,
                    flux_cap_reached=len(result.proposals) == self.extractor.max_proposals,
                    proposals=len(clouds), flux_proposals=flux_proposals,
                    heartbeat_proposals=heartbeat_proposals,
                    selected_flux_proposals=sum(p['source'] == 'flux' for p in selected),
                    selected_heartbeat_proposals=sum(p['source'] == 'heartbeat' for p in selected),
                    flux_supported=None if self.last_flux is None else self.last_flux['supported'],
                    tentative=sum(v=='TENTATIVE' for v in self.tracker.lifecycle.values()),
                    confirmed=sum(v=='CONFIRMED' for v in self.tracker.lifecycle.values()),
                    dormant=len(self.tracker.dormant_entities), resurrected=len(updated['resurrected']),
                    template_verified=sum(v=='template_verified' for v in updated['measurement_sources'].values()),
                    lifecycle_implemented=True,
                    identity_verified=False)

    def bind_seed(self, point_norm, image_hw=(1080,1920)):
        return bind_anchor(self.tracker.tracks, point_norm, image_hw)

    def close(self):
        for builder in self.builders.values():
            segment = builder.close()
            if segment is not None:
                self.tubes.append(segment)
        return self.tubes

    def junctions(self):
        return [j for i, first in enumerate(self.tubes) for second in self.tubes[i + 1:]
                if first.track_id != second.track_id for j in candidates(first, second)]
