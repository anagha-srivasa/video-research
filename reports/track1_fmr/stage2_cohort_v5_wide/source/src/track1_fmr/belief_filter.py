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


def condition(predicted, position, observation_covariance, shape_scatter=None):
    """Joseph-form conditioning of an existing prediction; preserves exact source time."""
    z = np.asarray(position, np.float64)
    if z.shape != (2,) or not np.isfinite(z).all():
        raise ValueError('Finite observed position required')
    r = _cov(observation_covariance, 2, positive=True)
    h = np.c_[np.eye(2), np.zeros((2, 2))]
    cov = predicted.covariance
    gain = np.linalg.solve(h @ cov @ h.T + r, h @ cov).T
    mean = predicted.mean + gain @ (z - h @ predicted.mean)
    a = np.eye(4) - gain @ h
    updated = a @ cov @ a.T + gain @ r @ gain.T
    return Belief(predicted.time, mean, (updated + updated.T) / 2,
                  predicted.shape_scatter if shape_scatter is None else shape_scatter, 'OBSERVED')


_ANGLES = np.arange(36) * (2 * np.pi / 36)
_CIRCLE = np.c_[np.cos(_ANGLES), np.sin(_ANGLES)]


def _ellipse_polygon(centre, scatter):
    import cv2
    values, vectors = np.linalg.eigh(_cov(scatter, 2, positive=True))
    points = (np.asarray(centre) + 2 * (_CIRCLE * np.sqrt(values)) @ vectors.T).astype(np.float32)
    return points, float(cv2.contourArea(points))


def _polygon_iou(first, second):
    import cv2
    a, area_a = first; b, area_b = second
    intersection = min(max(float(cv2.intersectConvexConvex(a, b)[0]), 0.), area_a, area_b)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def ellipse_iou(centre_a, scatter_a, centre_b, scatter_b):
    """36-gon approximation to two 2-sigma shape ellipses, not a geometry certificate."""
    return _polygon_iou(_ellipse_polygon(centre_a, scatter_a), _ellipse_polygon(centre_b, scatter_b))


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


def association_distance(predicted, cloud, measurement_floor=0.):
    """Position gating distance; appearance identity is not certified."""
    delta = cloud.centroid - predicted.mean[:2]
    covariance = predicted.covariance[:2, :2] + cloud.obs_cov_c + np.eye(2) * measurement_floor**2
    return float(delta @ np.linalg.solve(covariance, delta))


def directional_noise(covariance, angles, visibility, inflation):
    """Inflate declared ray directions, without inferring visibility from RGB."""
    cov = _cov(covariance, 2, positive=True)
    angles, visibility = np.asarray(angles, float), np.asarray(visibility, float)
    if angles.ndim != 1 or angles.shape != visibility.shape or not np.isfinite(angles).all() or not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)) or not np.isfinite(inflation) or inflation < 0:
        raise ValueError('Valid visibility rays and nonnegative inflation required')
    rays = np.c_[np.cos(angles), np.sin(angles)]
    return cov + inflation * np.einsum('n,ni,nj->ij', 1 - visibility, rays, rays) / max(len(rays), 1)


def template_patch(gray, centre, radius=7):
    x, y = np.rint(np.asarray(centre) * [gray.shape[1], gray.shape[0]] - .5).astype(int)
    if x < radius or y < radius or x + radius >= gray.shape[1] or y + radius >= gray.shape[0]:
        return None
    return gray[y-radius:y+radius+1, x-radius:x+radius+1].copy()


def masked_patch(gray, centre, scatter, foreground=None):
    h,w=gray.shape
    rx,ry=np.clip(np.rint(1.25*np.sqrt(np.diag(scatter))*[w,h]),3,6).astype(int)
    x,y=np.rint(np.asarray(centre)*[w,h]-.5).astype(int)
    if x<rx or y<ry or x+rx>=w or y+ry>=h:return None,None,None
    ys,xs=slice(y-ry,y+ry+1),slice(x-rx,x+rx+1)
    yy,xx=np.mgrid[ys,xs]
    delta=np.stack(((xx+.5)/w-centre[0],(yy+.5)/h-centre[1]),axis=-1)
    mask=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(scatter),delta)<=1.5625
    if foreground is not None:mask &= np.asarray(foreground,bool)[ys,xs]
    return gray[ys,xs].copy(),mask,(ys,xs)


def verify_masked_stationary_lock(gray_grid, pred_centroid_norm, shape_scatter_norm,
        residual_flow_norm, template_patch, template_mask, v_eff, ncc_thresh=.78):
    patch,mask,slices=masked_patch(gray_grid,pred_centroid_norm,shape_scatter_norm)
    if patch is None or mask.sum()<9:return False,0.,patch,mask
    if template_patch is None or template_mask is None or template_patch.shape!=patch.shape or template_mask.shape!=mask.shape:
        return False,0.,patch,mask
    valid=mask & template_mask
    if valid.sum()<9:return False,0.,patch,mask
    flow=residual_flow_norm[slices][valid]
    if not np.isfinite(flow).all() or np.linalg.norm(flow,axis=-1).mean()>1.2*v_eff:return False,0.,patch,mask
    a=template_patch[valid].astype(np.float32);b=patch[valid].astype(np.float32)
    a-=a.mean();b-=b.mean();denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    if denom<1e-5:return False,0.,patch,mask
    ncc=float(a@b/denom)
    return ncc>=ncc_thresh,ncc,patch,mask


def verify_stationary_lock(gray_grid, pred_centroid_norm, shape_scatter_norm, residual_flow_norm, template, v_eff, ncc_thresh=.78):
    # Compatibility for callers without retained binary support; production stores proposal masks.
    _,mask,_=masked_patch(gray_grid,pred_centroid_norm,shape_scatter_norm)
    result=verify_masked_stationary_lock(gray_grid,pred_centroid_norm,shape_scatter_norm,residual_flow_norm,template,mask,v_eff,ncc_thresh)
    return result[:3]


def containment_association_cost(parent_centroid,parent_scatter,parent_sig,child_centroid,child_scatter,child_sig,
                                 max_mahal_sq=5.06,parent_area=None,child_area=None):
    parent_area=np.sqrt(np.linalg.det(parent_scatter)) if parent_area is None else parent_area
    child_area=np.sqrt(np.linalg.det(child_scatter)) if child_area is None else child_area
    if child_area>=.6*parent_area:return None
    delta=child_centroid-parent_centroid
    d=float(delta@np.linalg.solve(parent_scatter,delta))
    lab=float(np.linalg.norm(parent_sig[4:10]-child_sig[4:10])/100.)
    if d>max_mahal_sq or lab>.45:return None
    return .35*d/max_mahal_sq+.45*lab+.15


def condition_cloud(predicted,cloud,floor,adapt=False,alpha=.5):
    shape=(1-alpha)*predicted.shape_scatter+alpha*cloud.shape_scatter if adapt else predicted.shape_scatter
    updated=condition(predicted,cloud.centroid,cloud.obs_cov_c+np.eye(2)*floor**2,shape)
    if adapt:
        mean=updated.mean.copy();mean[:2]=(1-alpha)*predicted.mean[:2]+alpha*cloud.centroid
        return Belief(updated.time,mean,updated.covariance,shape,'OBSERVED')
    return updated


def bind_anchor(tracks, point_norm, image_hw=(1080,1920)):
    """Reference-only initialization: centroid proximity OR approximate shape support."""
    choices=[]
    for tid, state in tracks.items():
        if state.status != 'OBSERVED': continue
        delta=np.asarray(point_norm)-state.mean[:2]
        distance=float(delta @ np.linalg.solve(state.shape_scatter,delta))
        if distance <= 4 or np.linalg.norm(delta*np.array(image_hw[::-1])) <= 50:
            choices.append((distance,tid))
    return min(choices)[1] if choices else None


@dataclass
class DormantEntity:
    state: Belief
    last_observed_ts: float
    signature: np.ndarray
    cloud: object
    patch: np.ndarray | None
    observed_count: int
    template_mask: np.ndarray | None = None


class BeliefTracker:
    """Bounded proposal association and coasting; IDs are provisional."""
    def __init__(self, acceleration_noise=.001, gate=9., max_tracks=40, measurement_floor=0., ncc_threshold=.78, containment_gate=5.06, adaptation_rate=.5):
        if acceleration_noise < 0 or gate <= 0 or max_tracks < 1 or not np.isfinite(measurement_floor) or measurement_floor < 0:
            raise ValueError('Invalid tracker configuration')
        self.noise, self.gate, self.max_tracks = acceleration_noise, gate, max_tracks
        if not np.isfinite(ncc_threshold) or not -1<=ncc_threshold<=1:
            raise ValueError("Invalid NCC threshold")
        if not np.isfinite(containment_gate) or containment_gate<=0 or not 0<adaptation_rate<=1:raise ValueError("Invalid containment/adaptation settings")
        self.containment_gate=containment_gate
        self.adaptation_rate=adaptation_rate
        self.ncc_threshold = float(ncc_threshold)
        self.measurement_floor = float(measurement_floor)
        self.tracks = {}
        self.signatures = {}
        self.observed_count = {}
        self.misses = {}
        self._inverse_shapes = {}
        self._prediction_key = None
        self._prediction_cache = {}
        self.last_observed = {}
        self.templates = {}
        self.template_masks = {}
        self.last_clouds = {}
        self.dormant_entities = {}
        self.lifecycle = {}
        self.next_id = 0
        self.last_time = None

    def predict_at(self, time, ids=None):
        """Reuse exact current-frame predictions, bound to source object/time/noise."""
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError('Times must increase')
        key = (float(time), float(self.noise))
        if key != self._prediction_key:
            self._prediction_key = key
            self._prediction_cache.clear()
        result = {}
        for track_id in self.tracks if ids is None else ids:
            source = self.tracks[track_id]
            cached = self._prediction_cache.get(track_id)
            if cached is None or cached[0] is not source:
                cached = (source, advance(source, time, self.noise))
                self._prediction_cache[track_id] = cached
            result[track_id] = cached[1]
        return result

    def inverse_shape_scatter(self, track_id):
        """Cache regularized float32 inverse; invalidate when the actual shape changes."""
        shape = self.tracks[track_id].shape_scatter
        cached = self._inverse_shapes.get(track_id)
        if cached is None or not np.array_equal(cached[0], shape):
            inverse = np.linalg.inv(shape.astype(np.float32) + np.eye(2, dtype=np.float32) * 1e-8)
            inverse.setflags(write=False)
            self._inverse_shapes[track_id] = (shape, inverse)
        return self._inverse_shapes[track_id][1]

    def update(self, time, clouds, *, birth_kinematics=None, flow_advection=None, visual_context=None, reid_similarity=None, reid_compare=None, foreground_masks=None):
        from scipy.optimize import linear_sum_assignment
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError('Times must increase')
        births = {}
        for index, (velocity, covariance) in (birth_kinematics or {}).items():
            v = np.asarray(velocity, np.float64)
            if not isinstance(index, int) or not 0 <= index < len(clouds) or v.shape != (2,) or not np.isfinite(v).all():
                raise ValueError('Birth kinematics require a valid cloud index and finite velocity/sec')
            births[index] = (v.copy(), _cov(covariance, 2, positive=True))
        advected = {}
        for track_id, (displacement, covariance) in (flow_advection or {}).items():
            d = np.asarray(displacement, np.float64)
            if track_id not in self.tracks or d.shape != (2,) or not np.isfinite(d).all():
                raise ValueError('Flow advection requires a known track and finite image displacement/frame')
            advected[track_id] = (d, _cov(covariance, 2, positive=True))
        for tid in list(self.dormant_entities):
            if time-self.dormant_entities[tid].last_observed_ts > 10.:
                del self.dormant_entities[tid]
                self.lifecycle.pop(tid, None)
        ids = list(self.tracks)
        predictions = self.predict_at(time)
        for track_id, (displacement, noise) in advected.items():
            prior = self.tracks[track_id]
            dt = time - prior.time
            mean = np.r_[prior.mean[:2] + displacement, displacement / dt]
            lift = np.r_[np.eye(2), np.eye(2) / dt]
            covariance = predictions[track_id].covariance + lift @ noise @ lift.T
            predictions[track_id] = Belief(float(time), mean, covariance, prior.shape_scatter, 'PREDICTED')
        predicted = [predictions[i] for i in ids]
        track_ellipses = [_ellipse_polygon(state.mean[:2], state.shape_scatter) for state in predicted]
        cloud_ellipses = [_ellipse_polygon(cloud.centroid, cloud.shape_scatter) for cloud in clouds]
        cost = np.full((len(ids), len(clouds)), 1e6)
        for i, state in enumerate(predicted):
            for j, cloud in enumerate(clouds):
                d = association_distance(state, cloud, self.measurement_floor)
                if d <= self.gate:
                    overlap = _polygon_iou(track_ellipses[i], cloud_ellipses[j])
                    lab = np.linalg.norm((self.signatures[ids[i]][4:7] - cloud.algebraic_sig[4:7]) / 255.) / np.sqrt(3)
                    cost[i, j] = .5 * np.sqrt(d) + .3 * (1 - overlap) + .2 * float(lab)
        part_pairs=set()
        for i,state in enumerate(predicted):
            tid=ids[i]
            if self.observed_count.get(tid,1)<3:continue
            parent=self.last_clouds.get(tid)
            for j,cloud in enumerate(clouds):
                value=containment_association_cost(state.mean[:2],state.shape_scatter,self.signatures[tid],
                    cloud.centroid,cloud.shape_scatter,cloud.algebraic_sig,max_mahal_sq=self.containment_gate,
                    parent_area=None if parent is None else parent.mass,child_area=cloud.mass if parent is not None else None)
                if value is not None and value<cost[i,j]:cost[i,j]=value;part_pairs.add((i,j))
        pairs = []
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            pairs = [(i, j) for i, j in zip(rows, columns) if cost[i, j] < 1e6]
        assigned = {i: j for i, j in pairs}
        used = {j for _, j in pairs}
        associated = {}
        measurement_sources = {}
        archived = []
        for i, track_id in enumerate(ids):
            self.observed_count.setdefault(track_id, 1)
            self.misses.setdefault(track_id, 0)
            if i in assigned:
                cloud = clouds[assigned[i]]
                associated[track_id] = cloud
                # Current visible scatter may change with occlusion, so preserve prior shape.
                self.tracks[track_id] = condition_cloud(predictions[track_id],cloud,self.measurement_floor,(i,assigned[i]) in part_pairs,self.adaptation_rate)
                self.signatures[track_id] = cloud.algebraic_sig.copy()
                self.observed_count[track_id] += 1
                self.misses[track_id] = 0
                measurement_sources[track_id] = 'proposal'
            else:
                self.tracks[track_id] = predicted[i]
                self.misses[track_id] += 1
                if self.observed_count[track_id] >= 3 and visual_context is not None:
                    gray, residual, v_eff, sigma_base = visual_context
                    state = predictions[track_id]
                    locked, _, patch, _ = verify_masked_stationary_lock(gray,state.mean[:2],state.shape_scatter,
                        residual,self.templates.get(track_id),self.template_masks.get(track_id),v_eff,self.ncc_threshold)
                    if locked and np.all((state.mean[:2]>=0)&(state.mean[:2]<=1)):
                        from dataclasses import replace
                        cloud = replace(self.last_clouds[track_id], centroid=state.mean[:2],
                                        obs_cov_c=np.eye(2)*2*sigma_base**2)
                        self.tracks[track_id] = condition(state, cloud.centroid, cloud.obs_cov_c)
                        associated[track_id] = cloud
                        measurement_sources[track_id] = 'template_verified'
                        self.observed_count[track_id] += 1
                        self.misses[track_id] = 0
                if track_id not in associated and self.observed_count[track_id]>=3:
                    state = self.tracks[track_id]
                    mean = state.mean.copy()
                    mean[2:] = self.tracks[track_id].mean[2:] * np.exp(-1.5*(time-self.last_time))
                    self.tracks[track_id] = Belief(time,mean,state.covariance,state.shape_scatter,'PREDICTED')
        # Evict tentative tracks before admitting births; retain confirmed entities separately.
        evicted = [i for i in ids if self.observed_count[i] < 3 and self.misses[i] >= 3]
        for track_id in evicted:
            self.tracks.pop(track_id)
            self.signatures.pop(track_id, None)
            self.observed_count.pop(track_id, None)
            self.misses.pop(track_id, None)
            self._inverse_shapes.pop(track_id, None)
        for tid in ids:
            if tid in evicted: continue
            if tid in associated:
                self.last_observed[tid] = float(time)
                self.lifecycle[tid] = 'CONFIRMED' if self.observed_count[tid]>=3 else 'TENTATIVE'
            elif self.observed_count[tid]>=3:
                self.lifecycle[tid] = 'COASTING'
                state=self.tracks[tid]
                if time-self.last_observed.get(tid,time)>1. or np.sqrt(np.trace(state.covariance[:2,:2]))>.25:
                    if self.observed_count[tid]<5:
                        self.tracks.pop(tid);evicted.append(tid);continue
                    self.dormant_entities[tid]=DormantEntity(state,self.last_observed[tid],self.signatures[tid],
                        self.last_clouds.get(tid),self.templates.get(tid),self.observed_count[tid],self.template_masks.get(tid))
                    self.lifecycle[tid]='DORMANT';archived.append(tid)
                    self.tracks.pop(tid)
        while len(self.dormant_entities)>64:
            oldest=min(self.dormant_entities,key=lambda tid:self.dormant_entities[tid].last_observed_ts)
            del self.dormant_entities[oldest];self.lifecycle.pop(oldest,None)
        # Second association pass uses the repository's actual Fourier/Lab layout.
        dormant_ids=list(self.dormant_entities)
        remaining=[j for j in range(len(clouds)) if j not in used]
        reid_cost=np.full((len(dormant_ids),len(remaining)),1e6)
        for row,tid in enumerate(dormant_ids):
            ent=self.dormant_entities[tid];radius=min(.35,.08+.15*(time-ent.last_observed_ts))
            for col,j in enumerate(remaining):
                cloud=clouds[j];distance=np.linalg.norm(cloud.centroid-(ent.cloud.centroid if ent.cloud is not None else ent.state.mean[:2]))
                if distance>radius:continue
                shape=np.linalg.norm(cloud.algebraic_sig[:4]-ent.signature[:4])/max(np.linalg.norm(ent.signature[:4]),1e-4)
                lab=np.linalg.norm(cloud.algebraic_sig[4:10]-ent.signature[4:10])/255.
                value=.45*distance/radius+.35*min(lab,1.5)+.20*min(shape,1.5)
                if value<=.65:reid_cost[row,col]=value
        dormant_parts=set()
        for row,tid in enumerate(dormant_ids):
            ent=self.dormant_entities[tid]
            for col,j in enumerate(remaining):
                cloud=clouds[j]
                value=containment_association_cost(ent.cloud.centroid if ent.cloud is not None else ent.state.mean[:2],
                    ent.state.shape_scatter,ent.signature,cloud.centroid,cloud.shape_scatter,cloud.algebraic_sig,max_mahal_sq=self.containment_gate,
                    parent_area=None if ent.cloud is None else ent.cloud.mass,child_area=None if ent.cloud is None else cloud.mass)
                if value is not None and value<reid_cost[row,col]:reid_cost[row,col]=value;dormant_parts.add((row,col))
        # Use real SAVPE cosine evidence only for competing dormant candidates.
        embedding_resolved=set()
        for col,j in enumerate(remaining):
            candidates=np.flatnonzero(reid_cost[:,col]<=.65)
            candidates=sorted(candidates,key=lambda row:reid_cost[row,col])[:2]
            if len(candidates)<2 or abs(reid_cost[candidates[0],col]-reid_cost[candidates[1],col])>.12:continue
            scores=[]
            if reid_compare is not None:
                similarities=reid_compare([dormant_ids[row] for row in candidates],j)
                for row in candidates:
                    similarity=similarities.get(dormant_ids[row])
                    if similarity is not None and np.isfinite(similarity):scores.append((float(similarity),int(row)))
            elif reid_similarity is not None:
                for row in candidates:
                    similarity=reid_similarity(dormant_ids[row],j)
                    if similarity is not None and np.isfinite(similarity):scores.append((float(similarity),int(row)))
            scores.sort(reverse=True)
            if len(scores)==len(candidates) and scores[0][0]>=.8 and scores[0][0]-scores[1][0]>=.05:
                winner=scores[0][1]
                for row in candidates:
                    if row!=winner:reid_cost[row,col]=1e6
                reid_cost[winner,col]=.7*reid_cost[winner,col]+.3*(1-scores[0][0])
                embedding_resolved.add(col)
        resurrected=[]
        if reid_cost.size:
            rr,cc=linear_sum_assignment(reid_cost)
            for row,col in zip(rr,cc):
                if reid_cost[row,col]>.65 or len(self.tracks)>=self.max_tracks:continue
                # Ambiguous appearance-only candidates abstain; no unavailable SAVPE embedding is invented.
                competitors=np.sort(reid_cost[:,col])
                if col not in embedding_resolved and len(competitors)>1 and competitors[1]-competitors[0]<.05:continue
                tid=dormant_ids[row];j=remaining[col];ent=self.dormant_entities.pop(tid);cloud=clouds[j]
                pred=advance(ent.state,time,self.noise) if time>ent.state.time else ent.state
                self.tracks[tid]=condition_cloud(pred,cloud,self.measurement_floor,(row,col) in dormant_parts,self.adaptation_rate)
                self.observed_count[tid]=ent.observed_count+1;self.misses[tid]=0
                self.last_observed[tid]=time;self.lifecycle[tid]='CONFIRMED'
                self.templates[tid]=ent.patch;self.template_masks[tid]=ent.template_mask;self.last_clouds[tid]=ent.cloud
                associated[tid]=cloud;measurement_sources[tid]='reid_proposal'
                self.signatures[tid]=cloud.algebraic_sig.copy();used.add(j);resurrected.append(tid)
                predictions[tid]=pred
        rejected = 0
        for j, cloud in enumerate(clouds):
            if j in used:
                continue
            if len(self.tracks) >= self.max_tracks:
                rejected += 1
                continue
            velocity, velocity_cov = births.get(j, (np.zeros(2), np.eye(2) * .01))
            mean = np.r_[cloud.centroid, velocity]
            covariance = np.zeros((4, 4))
            # A region's centroid is not an accurately localized physical point.
            # This fixed floor is uncalibrated and must be evaluated on held-out references.
            covariance[:2, :2] = cloud.obs_cov_c + np.eye(2) * (1e-4 + self.measurement_floor**2)
            covariance[2:, 2:] = velocity_cov
            self.tracks[self.next_id] = Belief(float(time), mean, covariance, cloud.shape_scatter, 'OBSERVED')
            associated[self.next_id] = cloud
            self.signatures[self.next_id] = cloud.algebraic_sig.copy()
            self.observed_count[self.next_id] = 1
            self.misses[self.next_id] = 0
            self.last_observed[self.next_id] = time
            self.lifecycle[self.next_id] = "TENTATIVE"
            measurement_sources[self.next_id] = "proposal"
            self.inverse_shape_scatter(self.next_id)
            self.next_id += 1
        for tid,cloud in associated.items():
            self.last_clouds[tid]=cloud
            if visual_context is not None and measurement_sources[tid]!='template_verified':
                index=next((j for j,c in enumerate(clouds) if c is cloud),None)
                foreground=(foreground_masks or {}).get(index)
                self.templates[tid],self.template_masks[tid],_=masked_patch(visual_context[0],self.tracks[tid].mean[:2],self.tracks[tid].shape_scatter,foreground)
        for tid in evicted:
            self.lifecycle.pop(tid,None)
        for tid in evicted+archived:
            for mapping in (self.signatures,self.observed_count,self.misses,self._inverse_shapes,self.last_observed,self.templates,self.template_masks,self.last_clouds):
                if tid not in resurrected:mapping.pop(tid,None)
        self.last_time = float(time)
        self._prediction_cache.clear()
        return dict(tracks=self.tracks.copy(), clouds=associated, matched=len(pairs), rejected_capacity=rejected,
                    predictions=predictions, archived=archived, resurrected=resurrected, measurement_sources=measurement_sources, flow_advected=len(advected), assignment_cost='0.5*Mahalanobis_distance+0.3*(1-ellipse_IoU)+0.2*normalized_Lab_distance', evicted_working_set=evicted, identity_verified=False, covariance_calibrated=False)
