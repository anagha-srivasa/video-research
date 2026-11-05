"""Four causal pose/velocity states; learned correction to the CV baseline."""
import json
import numpy as np
import torch
from torch import nn
from pipeline.common import ROOT

CFG = json.loads((ROOT/'configs/track2_kinematics.json').read_text())
HORIZONS = np.array(CFG['horizons_us'], dtype=np.float64) / 1e6


def baseline(history, constant_position=False):
    """Position and physical-time source velocity, at each declared horizon."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (4, 4) or not np.isfinite(history).all():
        raise ValueError('Expected finite four-state position/velocity history')
    result = np.repeat(history[:, -1:, :], len(HORIZONS), axis=1)
    if constant_position:
        result[:, :, 2:] = 0
    else:
        result[:, :, :2] += HORIZONS[None, :, None] * history[:, -1:, 2:]
    return result


def relative(history):
    value = np.array(history, dtype=np.float64, copy=True)
    value[:, :, :2] -= value[:, -1:, :2]
    return value


def normalization(data, ids):
    if not len(ids) or any(data.split[ids] != 'train'):
        raise ValueError('Normalization accepts training participants only')
    history = np.asarray(data.x[ids, :, 140:144], dtype=np.float64)
    x = relative(history)
    residual = np.asarray(data.y[ids, :, :4], dtype=np.float64) - baseline(history)
    observed = residual[data.mask[ids]]
    if not len(observed):
        raise ValueError('No supported training targets')
    # Fixed isotropic scales within position and velocity preserve coordinate
    # balance. No offset is learned: a zero head exactly retains CV.
    xs = [max(float(np.sqrt(np.mean(x[:, :, a:b]**2))), 1e-3) for a, b in [(0, 2), (2, 4)]]
    ys = [max(float(np.sqrt(np.mean(observed[:, a:b]**2))), 1e-3) for a, b in [(0, 2), (2, 4)]]
    return dict(x_scale=np.repeat(xs, 2).tolist(), y_scale=np.repeat(ys, 2).tolist(),
                training_participants=sorted(set(data.participants[ids].tolist())),
                training_histories=len(ids), supported_targets=len(observed))


def tensors(data, norm):
    history = np.asarray(data.x[:, :, 140:144], dtype=np.float64)
    x = relative(history) / np.asarray(norm['x_scale'])
    y = (np.asarray(data.y[:, :, :4], dtype=np.float64) - baseline(history)) / np.asarray(norm['y_scale'])
    y[~data.mask] = 0
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(data.mask)


class Kinematics(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        hidden = CFG['hidden_size']
        if family == 'kinematic_gru':
            self.encoder = nn.GRU(4, hidden, batch_first=True)
        elif family == 'kinematic_mlp':
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(16, CFG['mlp_width']), nn.GELU(),
                                         nn.Linear(CFG['mlp_width'], hidden), nn.GELU())
        else:
            raise ValueError('Unknown kinematic model')
        self.head = nn.Linear(hidden, 12)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (4, 4):
            raise ValueError('Expected four kinematic states')
        z = self.encoder(x)
        if self.family == 'kinematic_gru':
            z = z[1][-1]
        return self.head(z).reshape(-1, 3, 4)


def loss(prediction, target, mask, weights=None):
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2] or not mask.any():
        raise ValueError('Explicit supported target mask required')
    # Mask before arithmetic, including nonfinite placeholders in external callers.
    difference = torch.where(mask[:, :, None], prediction-target, 0.)
    error = difference[:, :, :2].square().mean(-1) + .25*difference[:, :, 2:].square().mean(-1)
    weights = prediction.new_ones(len(prediction)) if weights is None else weights
    w = mask * weights[:, None]
    return (error*w).sum()/w.sum()


def predict(model, norm, history):
    model.eval()
    x = relative(history)/np.asarray(norm['x_scale'])
    with torch.no_grad():
        correction = model(torch.from_numpy(x.astype(np.float32))).numpy().astype(np.float64)
    return baseline(history) + correction*np.asarray(norm['y_scale'])
