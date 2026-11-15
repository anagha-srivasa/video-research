"""Experimental image-to-tube pipeline; proposals do not certify objects."""
import cv2
import numpy as np
from .belief_filter import BeliefTracker, advance
from .dorsal_flux import flux, PromptFreeHeartbeat, proposal_nms
from .lebesgue_tubes import FunctionalTubeBuilder
from .junction_graph import candidates
from track1_encoder_revision.events import InnovationEvidence


class Pipeline:
    def __init__(self, heartbeat_seconds=2., max_tracks=100, carry_prior=False, dorsal_grid=None, heartbeat=None):
        if heartbeat_seconds <= 0:
            raise ValueError('Heartbeat period must be positive')
        self.heartbeat = heartbeat if heartbeat is not None else PromptFreeHeartbeat()
        self.period = heartbeat_seconds
        if dorsal_grid is not None and (isinstance(dorsal_grid, bool) or not isinstance(dorsal_grid, int) or dorsal_grid < 16):
            raise ValueError('Dorsal grid must be an integer >= 16, or None for native resolution')
        self.dorsal_grid = dorsal_grid
        self.carry_prior = bool(carry_prior)
        self.tracker = BeliefTracker(max_tracks=max_tracks)
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
        # Clouds use normalized coordinates; native pixels are retained only for crops.
        dorsal = rgb if self.dorsal_grid is None else cv2.resize(
            rgb, (self.dorsal_grid, self.dorsal_grid), interpolation=cv2.INTER_AREA)
        proposals = []
        flux_proposals = 0
        heartbeat_proposals = 0
        if self.previous is not None:
            height, width = dorsal.shape[:2]
            excluded = np.zeros((height, width), np.uint8)
            for state in self.tracker.tracks.values():
                centre = tuple(np.round(state.mean[:2] * [width, height]).astype(int))
                scale = np.diag([width, height])
                covariance = scale @ state.shape_scatter @ scale
                values, vectors = np.linalg.eigh(covariance)
                axes = tuple(np.maximum(2, np.minimum([width, height], np.ceil(2 * np.sqrt(np.maximum(values[::-1], 0))))).astype(int))
                angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))
                cv2.ellipse(excluded, centre, axes, angle, 0, 360, 1, -1)
            result = flux(self.previous, dorsal, excluded=excluded)
            if result['supported']:
                proposals.extend(result['clouds'])
                flux_proposals = len(result['clouds'])
            else:
                self.failures.append((float(time), result['reason']))
        if time - self.last_heartbeat >= self.period:
            anchors = self.heartbeat(rgb)
            proposals.extend(anchors)
            heartbeat_proposals = len(anchors)
            self.last_heartbeat = float(time)
        selected = proposal_nms(proposals)
        clouds = [p['measure'] for p in selected]
        before = self.tracker.tracks.copy()
        updated = self.tracker.update(time, clouds)
        for track_id, state in updated['tracks'].items():
            cloud = updated['clouds'].get(track_id)
            innovation = 0.
            if cloud is not None and track_id in before:
                pred = advance(before[track_id], time, self.tracker.noise)
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
            if cloud is not None:
                h, w = rgb.shape[:2]
                centre = cloud.centroid * [w, h]
                extent = np.maximum(4, 3 * np.sqrt(np.diag(cloud.shape_scatter)) * [w, h])
                left, top = np.maximum(0, (centre - extent).astype(int))
                right, bottom = np.minimum([w, h], np.ceil(centre + extent).astype(int))
                if right > left and bottom > top:
                    patch = cv2.resize(rgb[top:bottom, left:right], (96, 96))
            # No independent visibility/depth estimate: zero is an unknown-quality marker.
            segment = builder.add(state, cloud, innovation, patch, 0. if patch is not None else None)
            if segment is not None:
                self.tubes.append(segment)
        self.previous = dorsal.copy()
        return dict(observed=sum(s.status == 'OBSERVED' for s in updated['tracks'].values()),
                    coasting=sum(s.status == 'PREDICTED' for s in updated['tracks'].values()),
                    tracks=len(updated['tracks']), rejected_capacity=updated['rejected_capacity'],
                    proposals=len(clouds), flux_proposals=flux_proposals,
                    heartbeat_proposals=heartbeat_proposals,
                    tentative=None, confirmed=None, lifecycle_implemented=False,
                    identity_verified=False)

    def close(self):
        for builder in self.builders.values():
            segment = builder.close()
            if segment is not None:
                self.tubes.append(segment)
        return self.tubes

    def junctions(self):
        return [j for i, first in enumerate(self.tubes) for second in self.tubes[i + 1:]
                if first.track_id != second.track_id for j in candidates(first, second)]
