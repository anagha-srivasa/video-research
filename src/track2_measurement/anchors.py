"""Identical information and scales; anchor and velocity-loss interventions only."""
import json,time
import numpy as np
import torch
from torch import nn
from pipeline.common import atomic_json
from track1_retrain.common import guard
from track2_feasibility.data import Data
from track2_feasibility.train import seed_all,weight_hash,participant_weights
from track2_kinematics.model import baseline
from .data import C,D,R,W


def inputs(data,raw):
    history=data.x[:,:,140:144].astype(float).copy();last=raw[:,-1:,:2]
    history[:,:,:2]-=last
    return np.concatenate([history,raw[:,:,:2]-last,raw[:,:,2:]],axis=2)


def base(data,raw,anchor):
    if anchor=='cv':return baseline(data.x[:,:,140:144])
    if anchor!='nearest':raise ValueError('Unknown anchor')
    result=np.zeros((len(data.samples),3,4));result[:,:,:2]=raw[:,-1:, :2];return result


def normalization(data,raw,ids):
    if not len(ids) or any(data.split[ids]!='train'):raise ValueError('Training participants only')
    x=inputs(data,raw)[ids];y=(data.y[:,:,:4].astype(float)-base(data,raw,'nearest'))[ids][data.mask[ids]]
    return dict(x_scale=np.maximum(np.sqrt(np.mean(x*x,axis=(0,1))),.001).tolist(),
        y_scale=np.repeat([max(float(np.sqrt(np.mean(y[:,:2]**2))),.001),max(float(np.sqrt(np.mean(y[:,2:]**2))),.001)],2).tolist(),
        participants=sorted(set(data.participants[ids].tolist())))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.head=nn.Linear(32,12);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return self.head(self.net(x)).reshape(-1,3,4)


def loss(pred,target,mask,velocity_weight,weights=None):
    if not mask.any():raise ValueError('No target support')
    # Select before arithmetic so unsupported placeholder labels cannot affect gradients.
    p=pred[mask];y=target[mask];error=(p[:,:2]-y[:,:2]).square().mean(-1)
    if velocity_weight:error=error+velocity_weight*(p[:,2:]-y[:,2:]).square().mean(-1)
    w=torch.ones(len(pred),dtype=pred.dtype) if weights is None else weights
    w=w[:,None].expand_as(mask)[mask]
    return (error*w).sum()/w.sum()


def fit(family,seed,held=None,epochs=None):
    seed_all(seed);data=Data();raw=np.load(D/'recent_observation.npy');supported=np.load(D/'input_mask.npy')
    ids=data.indices(exclude=held);ids=ids[supported[ids]];stop=data.indices(only=held) if held else None
    if held:stop=stop[supported[stop]]
    norm=normalization(data,raw,ids);x=torch.tensor(inputs(data,raw)/norm['x_scale'],dtype=torch.float32)
    anchor=family.split('_')[0];vw=.25 if family.endswith('joint') else 0.
    target=(data.y[:,:,:4].astype(float)-base(data,raw,anchor))/norm['y_scale'];target[~data.mask]=0
    y=torch.tensor(target,dtype=torch.float32);mask=torch.tensor(data.mask)
    model=Model();initial=weight_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=C['learning_rate'],weight_decay=C['weight_decay'])
    weights=torch.tensor(participant_weights(data,ids));rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;stale=0;state=None;history=[]
    for epoch in range(1,(epochs or C['maximum_epochs'])+1):
        total=0.;order=rng.permutation(len(ids));model.train()
        for start in range(0,len(ids),C['batch_size']):
            at=order[start:start+C['batch_size']];batch=ids[at];opt.zero_grad(set_to_none=True)
            value=loss(model(x[batch]),y[batch],mask[batch],vw,weights[at])
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
            value.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(value.detach())*len(batch)
        score=None
        if held:
            model.eval()
            # Identical position-only stopping criterion isolates training-loss interference.
            with torch.no_grad():score=float(loss(model(x[stop]),y[stop],mask[stop],0.))
            if score<best-C['minimum_improvement']:
                best=score;best_epoch=epoch;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        else:best_epoch=epoch
        history.append(dict(epoch=epoch,training_loss=total/len(ids),held_position_loss=score))
        if epoch%10==0:guard()
        if held and stale>=C['patience']:break
    if state is not None:model.load_state_dict(state)
    model.eval();record=dict(family=family,seed=seed,held=held,epochs=best_epoch,history=history,
        initial_hash=initial,final_hash=weight_hash(model),normalization=norm,parameters=sum(p.numel() for p in model.parameters()))
    return model,record


def fold(job):
    f,s,p=job;path=W/f'fold.{f}.{s}.{p}.json'
    if path.exists():return json.loads(path.read_text())
    _,record=fit(f,s,held=p);atomic_json(path,record);return record


def final(job):
    f,s,epochs=job;path=D/f'{f}.{s}.pt';record_path=R/f'training.{f}.{s}.json'
    if path.exists() and record_path.exists():return json.loads(record_path.read_text())
    model,record=fit(f,s,epochs=epochs);temp=path.with_suffix('.tmp');torch.save(dict(state=model.state_dict(),record=record),temp);temp.replace(path)
    loaded=torch.load(path,map_location='cpu',weights_only=True);other=Model();other.load_state_dict(loaded['state']);assert weight_hash(other)==record['final_hash']
    atomic_json(record_path,record);return record


def linear_fit(data,raw,ids,regularization):
    norm=normalization(data,raw,ids);x=(inputs(data,raw)/norm['x_scale']).reshape(len(data.samples),-1);x=np.c_[x,np.ones(len(x))]
    residual=(data.y[:,:,:4].astype(float)-base(data,raw,'nearest'))/norm['y_scale'];models=[]
    for h in range(3):
        selected=ids[data.mask[ids,h]];a=x[selected];w=participant_weights(data,selected).astype(float);w/=w.sum()
        ridge=np.eye(a.shape[1])*regularization;ridge[-1,-1]=0
        models.append(np.linalg.solve(a.T@(w[:,None]*a)+ridge,a.T@(w[:,None]*residual[selected,h])))
    return norm,np.array(models)


def linear_predict(data,raw,norm,coefficients):
    x=(inputs(data,raw)/norm['x_scale']).reshape(len(data.samples),-1);x=np.c_[x,np.ones(len(x))]
    return base(data,raw,'nearest')+np.einsum('nd,hdk->nhk',x,coefficients)*norm['y_scale']


def shrinkage(data,raw,ids):
    result=[];v=data.x[:,-1,142:144].astype(float);p=raw[:,-1,:2]
    for hi,h in enumerate(C['horizons_us']):
        ix=ids[data.mask[ids,hi]];w=participant_weights(data,ix);x=v[ix]*h/1e6;y=data.y[ix,hi,:2]-p[ix]
        den=np.sum(w[:,None]*x*x);result.append(float(np.clip(np.sum(w[:,None]*x*y)/den,0,1)) if den>0 else 0.)
    return np.array(result)


def shrink_predict(data,raw,alpha=None,beta=None):
    out=base(data,raw,'nearest');v=data.x[:,-1,142:144].astype(float)
    for hi,h in enumerate(np.array(C['horizons_us'])/1e6):
        if alpha is not None:distance=alpha[hi]*h;speed=alpha[hi]
        elif beta is not None:distance=h if beta==0 else -np.expm1(-beta*h)/beta;speed=np.exp(-beta*h)
        else:raise ValueError('A shrinkage model is required')
        out[:,hi,:2]+=distance*v;out[:,hi,2:]=speed*v
    return out


def position_score(data,ids,pred):
    scores=[]
    for p in sorted(set(data.participants[ids])):
        for h in range(3):
            ix=ids[(data.participants[ids]==p)&data.mask[ids,h]]
            scores.append(float(np.sqrt(np.mean(np.sum((pred[ix,h,:2]-data.y[ix,h,:2])**2,axis=1)))))
    return float(np.mean(scores))


def baselines():
    data=Data();raw=np.load(D/'recent_observation.npy');supported=np.load(D/'input_mask.npy');records=[]
    ids=data.indices();ids=ids[supported[ids]]
    for held in C['training_participants']:
        train=ids[data.participants[ids]!=held];test=ids[data.participants[ids]==held]
        for ridge in C['ridge_grid']:
            norm,coef=linear_fit(data,raw,train,ridge)
            records.append(dict(held=held,kind='ridge',value=ridge,score=position_score(data,test,linear_predict(data,raw,norm,coef))))
        for beta in C['decay_beta_grid']:
            records.append(dict(held=held,kind='decay',value=beta,score=position_score(data,test,shrink_predict(data,raw,beta=beta))))
    selections={kind:min(grid,key=lambda value:np.mean([r['score'] for r in records if r['kind']==kind and r['value']==value])) for kind,grid in [('ridge',C['ridge_grid']),('decay',C['decay_beta_grid'])]}
    norm,coef=linear_fit(data,raw,ids,selections['ridge']);np.save(D/'ridge.npy',coef,allow_pickle=False)
    atomic_json(R/'baseline_selection.json',dict(records=records,selected=selections,normalization=norm,shrinkage=shrinkage(data,raw,ids).tolist(),validation_used=False))
