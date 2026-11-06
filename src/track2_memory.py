"""Training-only ridge innovation following an exact polynomial coordinate shift."""
import json
import numpy as np
from numpy.polynomial.legendre import legfit,legval,legvander,leggauss
from pipeline.common import ROOT,atomic_json,digest
from track2_feasibility.data import Data,check_frozen
from track2_feasibility.train import participant_weights
from track1.numerics import spline_basis,wavelet_matrix
from track2_measurement.data import C,D,R
from track2_measurement.math_checks import transport_matrix


def fit(c,y,mask,data,ids,ridge):
    mean=c[ids].mean(0);scale=np.maximum(c[ids].std(0),.001);a=np.c_[((c-mean)/scale).reshape(len(c),-1),np.ones(len(c))];models=[]
    for hi,h in enumerate(np.array(C['horizons_us'])/1e6):
        selected=ids[mask[ids,hi]];weights=participant_weights(data,selected).astype(float);weights/=weights.sum();x=a[selected]
        shifted=np.einsum('ij,njk->nik',transport_matrix(2*h),c[selected]);residual=((y[selected,hi]-shifted)/scale).reshape(len(selected),-1)
        penalty=np.eye(a.shape[1])*ridge;penalty[-1,-1]=0
        models.append(np.linalg.solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights[:,None]*residual)))
    return mean,scale,np.array(models)


def predict(c,mean,scale,models):
    a=np.c_[((c-mean)/scale).reshape(len(c),-1),np.ones(len(c))]
    return np.stack([np.einsum('ij,njk->nik',transport_matrix(2*h),c)+(a@models[hi]).reshape(-1,4,35)*scale for hi,h in enumerate(np.array(C['horizons_us'])/1e6)],axis=1)


def decoded_error(pred,target):
    # Normalized shape coordinates only: no world pose, identity, certificate or physical geometry claim.
    phi=legvander(np.linspace(-1,1,9),3);p=np.einsum('ti,nik->ntk',phi,pred);q=np.einsum('ti,nik->ntk',phi,target)
    decode=spline_basis(16,64)@np.linalg.inv(wavelet_matrix(16,'db2'))
    shape=np.einsum('sm,ntmc->ntsc',decode,(p[:,:,2:-1]-q[:,:,2:-1]).reshape(len(p),9,16,2))
    return float(np.sqrt(np.mean(np.sum((p[:,:,:2]-q[:,:,:2])**2,axis=-1)))),float(np.sqrt(np.mean(np.sum(shape**2,axis=-1))))


def run():
    path=R/'memory_innovation.json'
    if path.exists():return
    check_frozen();atomic_json(R/'memory_protocol.json',dict(source_sha256=digest(ROOT/'src/track2_memory.py'),ridge_grid=C['memory_ridge_grid'],train_only_selection=True,validation_exploratory=True))
    data=Data();c=data.x[:,-1,:140].astype(float).reshape(-1,4,35);y=data.y[:,:,4:].astype(float).reshape(-1,3,4,35);mask=data.mask
    supported=np.load(D/'input_mask.npy');ids=data.indices();ids=ids[supported[ids]];records=[]
    for held in C['training_participants']:
        train=ids[data.participants[ids]!=held];stop=ids[data.participants[ids]==held]
        for ridge in C['memory_ridge_grid']:
            mean,scale,weights=fit(c,y,mask,data,train,ridge);pred=predict(c,mean,scale,weights);scores=[]
            for hi in range(3):
                ix=stop[mask[stop,hi]];scores.append(float(np.mean(((pred[ix,hi]-y[ix,hi])/scale)**2)))
            records.append(dict(held=held,ridge=ridge,score=float(np.mean(scores))))
    selected=min(C['memory_ridge_grid'],key=lambda r:np.mean([a['score'] for a in records if a['ridge']==r]))
    mean,scale,weights=fit(c,y,mask,data,ids,selected);np.save(D/'memory_ridge.npy',weights,allow_pickle=False);np.save(D/'memory_mean.npy',mean,allow_pickle=False);np.save(D/'memory_scale.npy',scale,allow_pickle=False)
    predictions={'copy':np.repeat(c[:,None],3,axis=1),'polynomial_reindex':np.stack([np.einsum('ij,njk->nik',transport_matrix(2*h),c) for h in np.array(C['horizons_us'])/1e6],axis=1),'reindex_ridge':predict(c,mean,scale,weights)}
    val=data.indices('validation',usable=False);val=val[supported[val]];results=[]
    for method,pred in predictions.items():
        for hi,h in enumerate(C['horizons_us']):
            for participant in C['validation_participants']:
                ix=val[(data.participants[val]==participant)&mask[val,hi]];position,shape=decoded_error(pred[ix,hi],y[ix,hi])
                results.append(dict(method=method,horizon_us=h,participant=participant,queries=len(ix),coefficient_nrmse=float(np.sqrt(np.mean(((pred[ix,hi]-y[ix,hi])/scale)**2))),decoded_history_position_rmse=position,decoded_normalized_shape_rmse=shape))
    synthetic=[];z=np.linspace(-1,1,128);functions={'constant':lambda x:np.ones_like(x),'linear':lambda x:x,'quadratic':lambda x:x*x,'cubic':lambda x:x**3,'periodic':lambda x:np.sin(2*x)}
    for name,f in functions.items():
        coef=legfit(z,f(z),3)
        for h in [.25,.5,1.]:
            target=f(z+2*h)
            synthetic.append(dict(trajectory=name,horizon_seconds=h,copy_rmse=float(np.sqrt(np.mean((legval(z,coef)-target)**2))),reindex_rmse=float(np.sqrt(np.mean((legval(z,transport_matrix(2*h)@coef)-target)**2)))))
    # Nonlinear composition needs its own quadrature check even when the history is cubic.
    quadrature=[]
    for nodes in [4,8,16,128]:
        z,w=leggauss(nodes);state=np.tanh(2*z**3+z);psi=1-z*z
        quadrature.append(dict(nodes=nodes,weak_left_integral=float(np.sum(w*2*z*state)),derivative_integral=float(np.sum(w*psi*(6*z*z+1)*(1-state*state)))))
    atomic_json(path,dict(selection=records,selected_ridge=selected,results=results,synthetic=synthetic,nonlinear_quadrature=quadrature,
        learning='One linear innovation map per horizon; 59220 learned weights. No new neural architecture or weak-form training objective.',
        scope='Same-epoch frozen functional targets. Decoded coordinates are representation diagnostics, not physical velocity, certified geometry, or independent boundaries.',
        validation_used_for_selection=False,promotion=False))
    check_frozen()

if __name__=='__main__':run()
