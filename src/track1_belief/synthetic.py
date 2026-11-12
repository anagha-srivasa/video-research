"""Exact motion references with separately generated observation failures."""
from __future__ import annotations

import numpy as np
from .common import C


KINDS = ('stationary', 'linear', 'pickup', 'burst', 'reversal')


def truth(times, kind, onset=2., duration=.3):
    t = np.asarray(times, float)
    u = (t - onset) / duration
    p, v = np.zeros_like(t), np.zeros_like(t)
    if kind == 'linear':
        p, v = 80 * t, np.full_like(t, 80.)
    elif kind == 'pickup':
        z = np.clip(u, 0, 1)
        p = 100 * (10*z**3 - 15*z**4 + 6*z**5)
        v = 100/duration * (30*z**2 - 60*z**3 + 30*z**4)
    elif kind == 'burst':
        active = (u > 0) & (u < 1)
        p[active] = 65 * np.sin(np.pi*u[active])**2
        v[active] = 65*np.pi/duration * np.sin(2*np.pi*u[active])
    elif kind == 'reversal':
        z = (t - onset) / .12
        p = 80*.12 * (np.logaddexp(z, -z) - np.log(2))
        v = 80 * np.tanh(z)
    elif kind != 'stationary':
        raise ValueError('Unknown analytic motion')
    return np.c_[700+p, 450+.3*p], np.c_[v, .3*v]


def episode(seed):
    rng = np.random.default_rng(C['synthetic_seed_offset'] + seed)
    kind = KINDS[seed % len(KINDS)]
    duration = (.12, .3, .6)[seed // 5 % 3]
    onset = 1.8 + rng.uniform(-.12, .12)
    t = np.cumsum(rng.uniform(.015, .025, 205))
    t -= t[0]
    corruption = ('clean', 'shared_bias', 'occlusion', 'outlier', 'missing', 'camera')[seed // 5 % 6]
    if corruption == 'missing':
        t = t[~((t > onset-.03) & (t < onset+.12))]
    position, velocity = truth(t, kind, onset, duration)
    sigma = (.7, 2., 4.)[seed // 15 % 3]
    groups = np.floor(t/.5).astype(int)
    biases = rng.normal(0, 4., (groups.max()+1, 2)) if corruption == 'shared_bias' else np.zeros((groups.max()+1, 2))
    # Sixteen noisy point displacement estimates. The reference is analytic,
    # never the mean of the input points. Common camera error is kept separate.
    points = position[:, None, :] + biases[groups, None, :] + rng.normal(0, sigma, (len(t), 16, 2))
    if corruption == 'outlier':
        k = np.argmin(abs(t - onset))
        points[k, 0] += [100., -70.]
    anchor = np.median(points, axis=1)
    local = np.tile(np.eye(2)*max(sigma*sigma/8, .05), (len(t), 1, 1))
    shared = np.tile(np.eye(2)*(16. if corruption == 'shared_bias' else 0.), (len(t), 1, 1))
    centroid = position + biases[groups] + rng.normal(0, sigma, position.shape)
    if corruption == 'occlusion':
        hidden = (t > onset-.1) & (t < onset+.4)
        centroid[hidden] += [55., -35.]
    if corruption == 'outlier':
        centroid[np.argmin(abs(t-onset))] += [100., -70.]
    camera = np.zeros_like(position)
    if corruption == 'camera':
        camera = np.c_[22*np.sin(t*2), 12*np.sin(t*3)]
        anchor += camera
        centroid += camera
    # Coherence is measured from agreement of point displacements, not truth.
    delta = np.diff(points, axis=0, prepend=points[:1])
    centre_delta = np.median(delta, axis=1)
    deviation = np.linalg.norm(delta-centre_delta[:, None], axis=2)
    coherence = np.mean(deviation < max(3*sigma, 1.), axis=1) >= .7
    return dict(episode_id=f'synthetic_{seed:04d}', seed=seed, kind=kind,
                corruption=corruption, participant=f'synthetic_{seed%5}',
                split='train' if seed < C['synthetic_training_episodes'] else 'validation',
                time=t, timestamp_us=np.rint(t*1e6).astype(np.int64),
                truth=position+camera, material_truth=position, velocity=velocity + np.c_[44*np.cos(2*t),36*np.cos(3*t)]*(corruption=='camera'),
                anchor_positions=anchor, centroid_positions=centroid,
                covariance_local=local, covariance_shared=shared,
                centroid_covariance=np.tile(np.eye(2)*sigma*sigma, (len(t),1,1)),
                valid=np.ones(len(t),bool), centroid_valid=np.ones(len(t),bool),
                proxy_positions=position+camera, proxy_valid=np.ones(len(t),bool),
                groups=groups.astype(str), coherence=coherence, onset=onset,
                duration=duration, reference_kind='analytic_exact',
                latest_evidence_us=np.rint(t*1e6).astype(np.int64))


def episodes(part=None):
    count = C['synthetic_training_episodes'] + C['synthetic_validation_episodes']
    for seed in range(count):
        row = episode(seed)
        if part is None or row['split'] == part:
            yield row
