"""Balanced corruption/motion experiments and common causal event scoring."""
from itertools import product
import numpy as np
from track1_belief.synthetic import truth, KINDS
from track1_belief.estimation import FilterConfig, process

CORRUPTIONS = ('clean', 'shared_bias', 'occlusion', 'outlier', 'missing', 'drift')
ARMS = tuple(f'd{d}_r{r}_j{j}' for d, r, j in product((0, 1), repeat=3))
METHODS = ('nearest', 'cv', *ARMS)


def episodes(split):
    """Full motion x corruption x duration factorial with independent RNGs.

    Episode is the randomization unit, never a fictional human participant.
    Coherent drift is deliberately unidentifiable from physical motion. Its
    source is not revealed to any estimator. Sensor variance and declared
    anchor-reset schedule are supplied to every arm equally.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Unknown split')
    for index, (kind, corruption, duration) in enumerate(product(KINDS, CORRUPTIONS, (.18, .6))):
        rng = np.random.default_rng(20261112 + index + 10000 * (split == 'validation'))
        t = np.arange(301) / 50
        onset = rng.uniform(1.5, 3.)
        p, v = truth(t, kind, onset, duration)
        angle = rng.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        gain = rng.uniform(.7, 1.4)
        p = (p - [700, 450]) @ rotation.T * gain + [700, 450]
        v = v @ rotation.T * gain
        sigma = rng.uniform(1., 3.)
        points = p[:, None] + rng.normal(0, sigma, (len(t), 16, 2))
        groups = np.floor(t / .8).astype(int)
        valid = np.ones(len(t), bool)
        at = int(np.searchsorted(t, onset))
        if corruption == 'shared_bias':
            points += rng.normal(0, 2., (groups.max() + 1, 2))[groups, None]
        elif corruption == 'occlusion':
            valid[(t >= onset - .1) & (t <= onset + .35)] = False
        elif corruption == 'outlier':
            # Corrupt a majority so robust point aggregation cannot erase it.
            points[at:at + 3, :12] += [70., -50.]
        elif corruption == 'missing':
            valid = rng.uniform(size=len(t)) > .3
            valid[0] = True
        elif corruption == 'drift':
            points += (np.maximum(t - onset, 0)[:, None] * np.array([18., -12.]))[:, None]
        y = np.median(points, axis=1)
        delta = np.diff(points, axis=0, prepend=points[:1])
        coherence = np.mean(np.linalg.norm(delta - np.median(delta, axis=1)[:, None], axis=2) < 3 * sigma, axis=1) >= .7
        yield dict(id=f'{split}_{index:03}', split=split, kind=kind, corruption=corruption,
                   duration=duration, onset=onset, time=t, truth=p, velocity=v,
                   y=y, valid=valid, groups=groups.astype(str), coherence=coherence,
                   noise_variance=np.pi * sigma**2 / (2 * 16))


def estimate(ep, method, parameter):
    """No truth, future, or corruption label is consulted by the estimator."""
    t, y = ep['time'], ep['y']
    ids = np.flatnonzero(ep['valid'])
    mean = np.zeros((len(t), 4))
    valid = np.zeros(len(t), bool)
    if method in ('nearest', 'cv'):
        for i in ids:
            past = ids[(t[ids] <= t[i]) & (t[ids] >= t[i] - parameter)]
            if method == 'nearest':
                mean[i, :2], valid[i] = y[i], True
            elif len(past) >= 3 and np.ptp(t[past]) >= .06:
                mean[i] = np.linalg.lstsq(np.c_[np.ones(len(past)), t[past] - t[i]], y[past], rcond=None)[0].ravel()
                valid[i] = True
    else:
        if method not in ARMS:
            raise ValueError('Unknown estimator')
        dependent, robust, jump = [int(s[1]) for s in method.split('_')]
        local = np.tile(np.eye(2) * ep['noise_variance'], (len(ids), 1, 1))
        shared = np.tile(np.eye(2) * 4., (len(ids), 1, 1))
        if not dependent:
            local += shared
            shared *= 0
        cfg = FilterConfig(acceleration_spectral_density=parameter,
                           jump_spectral_density=100 * parameter,
                           shared_bias_variance=0., initial_velocity_variance=10000.,
                           maximum_gap=.15, minimum_span=.06, memory_timescales=(),
                           robust_measurement=bool(robust),
                           process_mode='coherent_jump' if jump else 'gaussian',
                           jump_nis_threshold=25.)
        result = process(t[ids], y[ids], local, ep['groups'][ids], cfg,
                         ancestor_max_time=t[ids], coherence=ep['coherence'][ids],
                         shared_covariance=shared)
        mean[ids], valid[ids] = result['mean'], result['valid']
    return mean, valid


def event_head(times, velocities, valid, speed=20., dwell=.04):
    """One common causal hysteresis detector, independent of filter innovations.

    Onset: speed>=threshold; release: speed<threshold/2; reversal: opposite
    direction to the last established direction, sustained for dwell seconds.
    Unknown gaps reset evidence; they do not manufacture release events.
    """
    events = []
    moving = False
    direction = None
    released_at = None
    pending, since = None, None
    last_time = None
    for t, velocity, ok in zip(times, velocities, valid):
        if not ok or (last_time is not None and t - last_time > .15):
            moving, direction, pending, since = False, None, None, None
            released_at = None
            last_time = None
            if not ok:
                continue
        last_time = t
        magnitude = np.linalg.norm(velocity)
        event = None
        if not moving and magnitude >= speed:
            event = ('reversal' if direction is not None and released_at is not None
                     and t - released_at <= .5 and velocity @ direction < -.5 * magnitude else 'onset')
        elif moving and magnitude < speed / 2:
            event = 'release'
        elif moving and magnitude >= speed and direction is not None and velocity @ direction < -.5 * magnitude:
            event = 'reversal'
        if event != pending:
            pending, since = event, t
        if event is not None and t - since >= dwell - 1e-10:
            events.append((event, float(t)))
            moving = event != 'release'
            if moving:
                direction = velocity / max(magnitude, 1e-12)
                released_at = None
            else:
                released_at = t
            pending, since = None, None
    return events


def score_events(ep, predicted, tolerance=.25):
    """One-to-one causal matches; all unmatched alarms count, including motion."""
    reference = [(k, t) for k, t in event_head(ep['time'], ep['velocity'], np.ones(len(ep['time']), bool), dwell=0.) if t >= .5]
    alarms = [(k, t) for k, t in predicted if t >= .5]
    used = set()
    delays = []
    hits = dict.fromkeys(('onset', 'release', 'reversal'), 0)
    for kind, time in reference:
        matches = [(t - time, i) for i, (k, t) in enumerate(alarms)
                   if i not in used and k == kind and 0 <= t - time <= tolerance]
        if matches:
            delay, i = min(matches)
            used.add(i)
            hits[kind] += 1
            delays.append(delay)
    quiet = np.linalg.norm(ep['velocity'], axis=1) < 5.
    eligible = ep['time'] >= .5
    quiet_seconds = float(np.diff(ep['time'])[quiet[:-1] & eligible[:-1]].sum())
    false = [t for i, (_, t) in enumerate(alarms) if i not in used]
    quiet_false = sum(bool(quiet[min(np.searchsorted(ep['time'], t), len(quiet) - 1)]) for t in false)
    return dict(reference_events=len(reference), matched_events=len(used), false_alarms=len(false),
                quiet_false_alarms=quiet_false, quiet_seconds=quiet_seconds, delay_seconds=delays,
                **{k + '_references': sum(e[0] == k for e in reference) for k in hits},
                **{k + '_hits': n for k, n in hits.items()})


def continuation(mean, horizon, decay):
    """E[position] with exponential velocity damping; infinity is ordinary CV."""
    if horizon < 0 or decay <= 0:
        raise ValueError('Nonnegative horizon and positive decay required')
    distance = horizon if np.isinf(decay) else -decay * np.expm1(-horizon / decay)
    return mean[..., :2] + distance * mean[..., 2:]
