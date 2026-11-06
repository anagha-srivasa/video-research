import json
import numpy as np
import torch
from torch import nn
from pipeline.common import atomic_json,digest
from track1_retrain.common import guard,pool
from track2_feasibility.data import Data
from track2_feasibility.train import seed_all,weight_hash,participant_weights
from track2_measurement.anchors import Model,inputs,base,normalization,loss
from .model import Dynamics
from .common import C,D,R,W,OLD,validate


def fit(family,seed,held=None,epochs=None):
    if epochs is not None and (not isinstance(epochs,int) or epochs<0):raise ValueError('Epoch count must be nonnegative or None')
    seed_all(seed);data=Data();raw=np.load(OLD/'recent_observation.npy');ok=np.load(OLD/'input_mask.npy')
    ids=data.indices(exclude=held);ids=ids[ok[ids]];stop=data.indices(only=held) if held else None
    if held:stop=stop[ok[stop]]
    norm=normalization(data,raw,ids);x=torch.tensor(inputs(data,raw)/norm['x_scale'],dtype=torch.float32)
    structured=family.startswith('anchored_');anchor='nearest' if structured else family.split('_')[0];vw=.25 if family.endswith('joint') else 0.
    if structured:norm['y_scale']=[norm['y_scale'][0]]*4
    target=(data.y[:,:,:4]-base(data,raw,anchor))/norm['y_scale'];target[~data.mask]=0
    y=torch.tensor(target,dtype=torch.float32);mask=torch.tensor(data.mask);model=Dynamics(family) if structured else Model()
    initial=weight_hash(model);optimizer=torch.optim.AdamW(model.parameters(),lr=C['learning_rate'],weight_decay=C['weight_decay'])
    weights=torch.tensor(participant_weights(data,ids));rng=np.random.default_rng(seed);history=[];best_epoch=0;stale=0
    model.eval()
    with torch.no_grad():
        initial_loss=float(loss(model(x[ids]),y[ids],mask[ids],vw,weights))
        best=float(loss(model(x[stop]),y[stop],mask[stop],0.)) if held else None
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    history.append(dict(epoch=0,training_loss=initial_loss,held_position_loss=best))
    for epoch in range(1,(C['maximum_epochs'] if epochs is None else epochs)+1):
        model.train();total=0.;order=rng.permutation(len(ids))
        for start in range(0,len(ids),C['batch_size']):
            at=order[start:start+C['batch_size']];ix=ids[at];optimizer.zero_grad(set_to_none=True)
            value=loss(model(x[ix]),y[ix],mask[ix],vw,weights[at])
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
            value.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(value.detach())*len(ix)
        model.eval();score=None
        if held:
            with torch.no_grad():score=float(loss(model(x[stop]),y[stop],mask[stop],0.))
            if score<best-C['minimum_improvement']:
                best=score;best_epoch=epoch;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        else:best_epoch=epoch
        history.append(dict(epoch=epoch,training_loss=total/len(ids),held_position_loss=score))
        if epoch%10==0:guard()
        if held and stale>=C['patience']:break
    if held:model.load_state_dict(state)
    return model,dict(family=family,seed=seed,held=held,selected_epochs=best_epoch,history=history,normalization=norm,initial_hash=initial,final_hash=weight_hash(model),parameters=sum(p.numel() for p in model.parameters()))


def job(task):
    family,seed,held,epochs=task;tag=f'{family}.{seed}.{held or "final"}';record=W/f'{tag}.json';receipt=W/f'{tag}.receipt.json';modelpath=D/f'{family}.{seed}.pt'
    if receipt.exists():validate(json.loads(receipt.read_text()));return json.loads(record.read_text())
    model,result=fit(family,seed,held,epochs);atomic_json(record,result);files=[record]
    if held is None:
        temp=modelpath.with_suffix('.tmp');torch.save(dict(state=model.state_dict(),record=result),temp);temp.replace(modelpath);files.append(modelpath)
    atomic_json(receipt,{str(p.relative_to(D.parents[1])):digest(p) for p in files});return result


def train(monitor):
    tasks=[(f,s,p,None) for f in C['families'] for s in C['seeds'] for p in C['training_participants']]
    records=pool(job,tasks,monitor,workers=C['workers'])
    selection=[(f,s,None,int(np.median([r['selected_epochs'] for r in records if r['family']==f and r['seed']==s]))) for f in C['families'] for s in C['seeds']]
    atomic_json(R/'epoch_selection.json',dict(tasks=selection,epoch_zero_eligible=True,validation_used=False))
    pool(job,selection,monitor,workers=C['workers'])
