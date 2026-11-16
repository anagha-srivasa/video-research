"""Variable-epoch collocation interface; no Track 2 model training."""
import numpy as np


def sample(segment):
    nodes, weights = np.polynomial.legendre.leggauss(4)
    times = segment.t_start + (nodes + 1) * (segment.t_end - segment.t_start) / 2
    return dict(times=times, nodes=nodes, weights=weights,
                position=np.stack([segment.evaluate(t) for t in times]),
                velocity=np.stack([segment.velocity(t) for t in times]))


def pushforward(encoder, position, velocity):
    """Exact autodifferentiated tangent; physical contact is not synthesized."""
    import torch
    if position.shape != velocity.shape:
        raise ValueError('Position and velocity shapes must match')
    return torch.func.jvp(encoder, (position,), (velocity,))
