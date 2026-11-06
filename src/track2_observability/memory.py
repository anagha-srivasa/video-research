"""Uniform-measure overlap retention; a diagnostic for the irregular fitted store."""
import numpy as np
from numpy.polynomial.legendre import leggauss,legvander,legfit,legval
from pipeline.common import atomic_json
from track2_feasibility.data import Data
from track2_feasibility.train import participant_weights
from track2_measurement.math_checks import transport_matrix
from .common import C,D,R,OLD


def overlap(advance):
    h=float(advance)
    if not 0<=h<=1:raise ValueError('Advance must lie in [0,1] window lengths')
    z,w=leggauss(8);factors=(2*np.arange(4)+1)/2
    left=-1.;right=1-2*h;t=left+(z+1)*(right-left)/2
    f=factors[:,None]*(legvander(t,3).T@((w*(right-left)/2)[:,None]*legvander(t+2*h,3)))
    t=right+(z+1)*(1-right)/2;b=factors*(legvander(t,3).T@(w*(1-right)/2))
    return f,b


def filled(c,h):
    f,b=overlap(h);return np.einsum('ij,njk->nik',f,c)+b[None,:,None]*c.sum(1)[:,None,:]


def fit_incoming(c,y,mask,data,ids,regularization):
    mean=c[ids].mean(0);scale=np.maximum(c[ids].std(0),.001);state_scale=np.maximum(c[ids].sum(1).std(0),.001)
    x=np.c_[((c-mean)/scale).reshape(len(c),-1),np.ones(len(c))];models=[]
    for hi,h in enumerate(C['horizons_seconds']):
        ix=ids[mask[ids,hi]];_,b=overlap(h);target=(y[ix,hi].sum(1)-filled(c[ix],h).sum(1))/b.sum()/state_scale
        a=x[ix];w=participant_weights(data,ix).astype(float);w/=w.sum();penalty=np.eye(a.shape[1])*regularization;penalty[-1,-1]=0
        models.append(np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w[:,None]*target)))
    return mean,scale,state_scale,np.array(models)


def predict(c,norm):
    mean,scale,state_scale,weights=norm;x=np.c_[((c-mean)/scale).reshape(len(c),-1),np.ones(len(c))]
    return np.stack([filled(c,h)+overlap(h)[1][None,:,None]*(x@weights[hi]*state_scale)[:,None,:] for hi,h in enumerate(C['horizons_seconds'])],1)


def run():
    data=Data();c=data.x[:,-1,:140].reshape(-1,4,35).astype(float);y=data.y[:,:,4:].reshape(-1,3,4,35).astype(float);ok=np.load(OLD/'input_mask.npy');ids=data.indices();ids=ids[ok[ids]];folds=[]
    for held in C['training_participants']:
        train=ids[data.participants[ids]!=held];stop=ids[data.participants[ids]==held]
        for ridge in C['ridge_grid']:
            norm=fit_incoming(c,y,data.mask,data,train,ridge);pred=predict(c,norm);scores=[]
            for hi in range(3):
                ix=stop[data.mask[stop,hi]];scores.append(float(np.mean(((pred[ix,hi].sum(1)-y[ix,hi].sum(1))/norm[2])**2)))
            folds.append(dict(held=held,ridge=ridge,score=float(np.mean(scores))))
    selected=min(C['ridge_grid'],key=lambda a:np.mean([r['score'] for r in folds if r['ridge']==a]));norm=fit_incoming(c,y,data.mask,data,ids,selected)
    for name,value in zip(['mean','scale','state_scale','weights'],norm):np.save(D/f'overlap.{name}.npy',value,allow_pickle=False)
    pred={'copy':np.repeat(c[:,None],3,axis=1),'extension':np.stack([np.einsum('ij,njk->nik',transport_matrix(2*h),c) for h in C['horizons_seconds']],1),
          'overlap_persistence':np.stack([filled(c,h) for h in C['horizons_seconds']],1),'overlap_incoming_ridge':predict(c,norm)}
    val=data.indices('validation',usable=False);val=val[ok[val]];results=[]
    for method,value in pred.items():
        for hi,h in enumerate(C['horizons_seconds']):
            for participant in C['validation_participants']:
                ix=val[(data.participants[val]==participant)&data.mask[val,hi]];error=value[ix,hi].sum(1)-y[ix,hi].sum(1)
                results.append(dict(method=method,horizon_seconds=h,participant=participant,queries=len(ix),endpoint_position_rmse=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,1)))),endpoint_state_nrmse=float(np.sqrt(np.mean((error/norm[2])**2)))))
    synthetic=[];z=np.linspace(-1,1,256);functions={'constant':lambda t:np.ones_like(t),'linear':lambda t:t,'quadratic':lambda t:t*t,'cubic':lambda t:t**3,'periodic':lambda t:np.sin(2*t)}
    metric=np.sqrt(2/(2*np.arange(4)+1));operators=[]
    for h in C['horizons_seconds']:
        f,b=overlap(h);ext=transport_matrix(2*h)
        operators.append(dict(horizon_seconds=h,retention_norm=float(np.linalg.norm(metric[:,None]*f/metric[None,:],2)),extension_norm=float(np.linalg.norm(metric[:,None]*ext/metric[None,:],2))))
        for name,fun in functions.items():
            coef=legfit(z,fun(z),3);truth=float(fun(np.array(1+2*h)));filled_coef=filled(coef[None,:,None],h)[0,:,0]
            synthetic.append(dict(trajectory=name,horizon_seconds=h,copy_endpoint_error=float(abs(coef.sum()-truth)),extension_endpoint_error=float(abs((ext@coef).sum()-truth)),overlap_persistence_endpoint_error=float(abs(filled_coef.sum()-truth))))
    atomic_json(R/'overlap_memory.json',dict(selection=folds,selected_ridge=selected,records=results,synthetic=synthetic,operators=operators,
        learned_parameters=int(norm[3].size),scope='Uniform projection diagnostic, not an exact irregular/reliability-weighted store update. Persistence fill uses old endpoint only. Future data are training targets only.',promotion=False))
