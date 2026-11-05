"""Training-only stopping and deterministic all-training refits."""
import json
import logging
import time
import numpy as np
import torch
from pipeline.common import ROOT, atomic_json
from track2_feasibility.data import Data
from track2_feasibility.train import seed_all, weight_hash, participant_weights
from track1_retrain.common import guard
from .model import CFG, Kinematics, normalization, tensors, loss, predict

WORK = ROOT/'work/track2_kinematics'
DATA = ROOT/'derived/track2_kinematics'
REPORT = ROOT/'reports/track2_kinematics'
LOG = logging.getLogger(__name__)


def fit(family, seed, held=None, epochs=None):
    seed_all(seed)
    data = Data()
    ids = data.indices(exclude=held)
    stop_ids = data.indices(only=held) if held else None
    norm = normalization(data, ids)
    x, y, mask = tensors(data, norm)
    model = Kinematics(family)
    initial = weight_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    weights = torch.from_numpy(participant_weights(data, ids))
    rng = np.random.default_rng(seed)
    best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    history = []
    started = time.monotonic()
    for epoch in range(1, (epochs or CFG['maximum_epochs'])+1):
        model.train()
        total, count = 0., 0
        order = rng.permutation(len(ids))
        for offset in range(0, len(order), CFG['batch_size']):
            chosen = order[offset:offset+CFG['batch_size']]
            batch = ids[chosen]
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(x[batch]), y[batch], mask[batch], weights[chosen])
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(value.detach())*len(batch)
            count += len(batch)
        stopping = None
        if held:
            model.eval()
            with torch.no_grad():
                stopping = float(loss(model(x[stop_ids]), y[stop_ids], mask[stop_ids]))
            if not np.isfinite(stopping):
                raise FloatingPointError('Nonfinite stopping score')
            if stopping < best-CFG['minimum_improvement']:
                best, best_epoch, stale = stopping, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(dict(epoch=epoch, training_loss=total/count, stopping_loss=stopping))
        if epoch == 1 or epoch % 10 == 0:
            LOG.info('%s/%s/%s epoch=%d loss=%.6f stop=%s elapsed=%.1fs', family, seed, held or 'final', epoch, total/count, stopping, time.monotonic()-started)
            guard()
        if held and stale >= CFG['patience']:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    record = dict(family=family, seed=seed, held_participant=held, history=history,
                  best_epoch=best_epoch, completed_epochs=len(history), normalization=norm,
                  initial_weights_sha256=initial, final_weights_sha256=weight_hash(model),
                  seconds=time.monotonic()-started)
    return model, norm, record


def fold_job(job):
    family, seed, held = job
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    path = WORK/'folds'/f'{family}.{seed}.{held}.json'
    if path.exists():
        return json.loads(path.read_text())
    _, _, record = fit(family, seed, held=held)
    if held in record['normalization']['training_participants']:
        raise ValueError('Stopping participant leaked into normalization')
    atomic_json(path, record)
    return record


def final_job(job):
    family, seed, epochs = job
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    path = DATA/'models'/f'{family}.{seed}.pt'
    record_path = REPORT/f'training.{family}.{seed}.json'
    if path.exists() and record_path.exists():
        return json.loads(record_path.read_text())
    model, norm, record = fit(family, seed, epochs=epochs)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(family=family, seed=seed, epochs=epochs, state_dict=model.state_dict(), normalization=norm, record=record)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    restored, other, _ = load(path)
    data = Data()
    sample = data.x[data.indices()[:4], :, 140:144]
    if not np.array_equal(predict(model, norm, sample), predict(restored, other, sample)):
        raise ValueError('Checkpoint reload differs')
    record['reload_bitwise_equal'] = True
    atomic_json(record_path, record)
    return record


def load(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    model = Kinematics(value['family'])
    model.load_state_dict(value['state_dict'])
    model.eval()
    if weight_hash(model) != value['record']['final_weights_sha256']:
        raise ValueError('Model weight hash mismatch')
    return model, value['normalization'], value
