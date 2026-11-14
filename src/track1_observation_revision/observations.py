"""Local material-point transport; a baseline, not neural re-identification."""
import json
import numpy as np
from pipeline.common import ROOT, atomic_json, digest
from track1_belief.anchors import similarity_fit
from track1_belief.common import arrays, save


def transport(points, visible, initial, reference, timestamps, available, minimum=6):
    """Compose adjacent fits only along an unbroken accepted chain.

    A failed chain may restart only with a successful original-anchor fit.
    Surviving point IDs never get replaced by new unverified corners. Output
    describes the transported initialization point, not a deforming centroid.
    Covariance ignores unknown inter-step correlation and is explicitly not
    calibrated; it must not be used as a certified measurement uncertainty.
    """
    points, visible, timestamps = np.asarray(points), np.asarray(visible, bool), np.asarray(timestamps)
    initial, reference = np.asarray(initial), np.asarray(reference)
    available = np.asarray(available, bool)
    n = len(timestamps)
    if (points.ndim != 3 or points.shape[2] != 2 or visible.shape != points.shape[:2]
            or len(points) != n or initial.shape != points.shape[1:] or reference.shape != (2,)
            or available.shape != (n,) or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('Incompatible point ledger or non-increasing timestamps')
    positions = np.zeros((n, 2))
    covariance = np.zeros((n, 2, 2))
    valid = np.zeros(n, bool)
    reasons = np.full(n, 'no_decoded_frame', dtype='<U40')
    for i in range(n):
        if not available[i]:
            continue
        keep = visible[i]
        fit = None
        if i and valid[i - 1] and timestamps[i] - timestamps[i - 1] <= 100000:
            shared = keep & visible[i - 1]
            fit = similarity_fit(points[i - 1, shared], points[i, shared], positions[i - 1], minimum=minimum)
            if fit is not None:
                linear = fit['affine'][:, :2]
                covariance[i] = linear @ covariance[i - 1] @ linear.T
                reasons[i] = 'adjacent_material_transport'
        if fit is None:
            fit = similarity_fit(initial[keep], points[i, keep], reference, minimum=minimum)
            reasons[i] = 'original_anchor_recovery' if fit is not None else 'no_supported_chain'
        if fit is not None:
            positions[i], valid[i] = fit['position'], True
            covariance[i] += fit['covariance'] + np.eye(2) * fit['deformation_rmse']**2
    return dict(position=positions, conditional_covariance=covariance, valid=valid, reason=reasons)


def replay():
    report = ROOT / 'reports/track1_observation_revision'
    data = ROOT / 'derived/track1_observation_revision/observations'
    metadata = json.loads((ROOT / 'reports/track1_belief/anchor_data.json').read_text())
    rows = []
    for meta in metadata['episodes']:
        source = ROOT / meta['path']
        if digest(source) != meta['sha256']:
            raise ValueError('Original point ledger changed')
        with np.load(source, allow_pickle=False) as z:
            x = {k: z[k] for k in z.files}
        fit_ids = x['point_roles'] == 0
        result = transport(x['point_positions'][:, fit_ids], x['point_valid'][:, fit_ids],
                           x['point_initial'][fit_ids], x['seed_reference'],
                           x['timestamp_us'], x['latest_evidence_us'] >= 0)
        arrays(data / (meta['episode_id'] + '.npz'), **result, timestamp_us=x['timestamp_us'])
        for i in range(len(x['timestamp_us'])):
            rows.append(dict(episode_id=meta['episode_id'], video=meta['video_id'], split=meta['split'],
                             timestamp_us=int(x['timestamp_us'][i]), decoded=bool(x['latest_evidence_us'][i] >= 0),
                             original_supported=bool(x['valid'][i]), local_supported=bool(result['valid'][i]),
                             reason=str(result['reason'][i])))
    save(report / 'observations.parquet', rows)
    atomic_json(report / 'observations.json', dict(
        proposed=len(rows), decoded=sum(r['decoded'] for r in rows),
        original_supported=sum(r['original_supported'] for r in rows),
        local_supported=sum(r['local_supported'] for r in rows),
        accuracy_measured=False, neural_tracker_used=False, reidentification=False,
        scope='Same recorded point tracks; local transport support only. Neither physical accuracy nor calibrated covariance established.',
        input_hashes={m['path']: m['sha256'] for m in metadata['episodes']}))
