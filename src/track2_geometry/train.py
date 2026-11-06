import json,time
import numpy as np
import torch
from torch import nn
from pipeline.common import atomic_json
from track1_retrain.common import guard
from track2_feasibility.data import Data
from track2_feasibility.train import seed_all,weight_hash,participant_weights
from track2_kinematics.model import baseline,loss
from .data import CFG,D,R,W,normalize,features

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(4*29,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.head=nn.Linear(32,12);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return self.head(self.encoder(x)).reshape(-1,3,4)


def fit(family,seed,held=None,epochs=None):
    seed_all(seed);data=Data();geo=np.load(D/'geometry.npy');mask=np.load(D/'geometry_mask.npy')
    ids=data.indices(exclude=held);stop=data.indices(only=held) if held else None
    norm=normalize(data,ids,geo,mask);x=torch.from_numpy(features(data,norm,geo,mask,family))
    target=(data.y[:,:,:4].astype(float)-baseline(data.x[:,:,140:144]))/norm['y_scale'];target[~data.mask]=0
    y=torch.from_numpy(target.astype(np.float32));valid=torch.from_numpy(data.mask)
    model=Model();initial=weight_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    weights=torch.from_numpy(participant_weights(data,ids));rng=np.random.default_rng(seed)
    best=float('inf');best_epoch=0;stale=0;best_state=None;history=[];started=time.time()
    for epoch in range(1,(epochs or CFG['maximum_epochs'])+1):
        model.train();order=rng.permutation(len(ids));total=0.;count=0
        for lo in range(0,len(order),CFG['batch_size']):
            at=order[lo:lo+CFG['batch_size']];batch=ids[at];opt.zero_grad(set_to_none=True)
            value=loss(model(x[batch]),y[batch],valid[batch],weights[at])
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
            value.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(value.detach())*len(batch);count+=len(batch)
        score=None
        if held:
            model.eval()
            with torch.no_grad():score=float(loss(model(x[stop]),y[stop],valid[stop]))
            if not np.isfinite(score):raise FloatingPointError('Nonfinite stopping score')
            if score<best-CFG['minimum_improvement']:
                best=score;best_epoch=epoch;stale=0;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        else:best_epoch=epoch
        history.append(dict(epoch=epoch,loss=total/count,held_training_loss=score))
        if epoch%10==0:guard()
        if held and stale>=CFG['patience']:break
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    record=dict(family=family,seed=seed,held=held,best_epoch=best_epoch,history=history,
        seconds=time.time()-started,initial_hash=initial,final_hash=weight_hash(model),parameters=sum(p.numel() for p in model.parameters()),
        training_participants=norm['training_participants'])
    return model,norm,record


def fold(job):
    family,seed,held=job;path=W/f'fold.{family}.{seed}.{held}.json'
    if path.exists():return json.loads(path.read_text())
    _,_,record=fit(family,seed,held=held);atomic_json(path,record);return record


def final(job):
    family,seed,epochs=job;p=D/f'{family}.{seed}.pt';recordpath=R/f'training.{family}.{seed}.json'
    if p.exists() and recordpath.exists():return json.loads(recordpath.read_text())
    model,norm,record=fit(family,seed,epochs=epochs)
    tmp=p.with_suffix('.tmp');torch.save(dict(state=model.state_dict(),normalization=norm,record=record),tmp);tmp.replace(p)
    loaded=torch.load(p,map_location='cpu',weights_only=True);other=Model();other.load_state_dict(loaded['state'])
    if weight_hash(other)!=record['final_hash']:raise ValueError('Reload mismatch')
    atomic_json(recordpath,record);return record
