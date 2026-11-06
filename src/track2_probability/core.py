import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from track2_signal.probes import probabilistic_fit, design


def scores(p,y):
    p=np.clip(p,1e-8,1-1e-8);pos=y>0; n=pos.sum();m=len(y)-n
    auc=float((rankdata(p)[pos].sum()-n*(n+1)/2)/(n*m)) if n and m else None
    bins=[]
    for lo in np.arange(0,1,.2):
        mask=(p>=lo)&(p<lo+.2)
        if mask.any():bins.append(dict(lower=float(lo),count=int(mask.sum()),prediction=float(p[mask].mean()),frequency=float(y[mask].mean())))
    return dict(count=len(y),frequency=float(y.mean()),brier=float(np.mean((p-y)**2)),log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),auc=auc,reliability=bins)


def arms(raw,coeff,anchor):
    position=np.c_[raw[:,:,:2].reshape(len(raw),-1),np.linalg.norm(raw[:,:,:2],axis=2)]
    motion=np.c_[raw[:,:,2:4].reshape(len(raw),-1),np.linalg.norm(raw[:,:,2:4],axis=2)]
    metadata=raw[:,:,[4,5,6,7,20]].reshape(len(raw),-1)
    both=np.c_[position,motion,metadata]
    c=coeff.copy();c[:,:,0,:]-=anchor[:,None,:]
    # Past-only 250ms projection at each input hop, never future observations.
    from track2_signal.memory import update
    memory=np.stack([update(coeff[:,j],anchor+raw[:,j,:2], .25) for j in range(4)],axis=1)
    memory[:,:,0,:]-=anchor[:,None,:]
    return dict(energy=np.mean(np.sum(raw[:,:,:2]**2,axis=2),axis=1)[:,None],position=position,position_motion=np.c_[position,motion],position_metadata=np.c_[position,metadata],both=both,relation=np.c_[both,raw[:,:,8:20].reshape(len(raw),-1)],functional=np.c_[both,c.reshape(len(raw),-1)],memory=np.c_[both,memory.reshape(len(raw),-1)])


def fit(x,squared,participants,ids):
    a,scale=design(x,ids);w=np.array([1/np.sum(participants[ids]==p) for p in participants[ids]]);w/=w.sum()
    threshold=float(np.quantile(np.sqrt(squared[ids]),.75));event=(squared>threshold**2).astype(float)
    bp,logit,rp=probabilistic_fit(a[ids],event[ids],w,'probability',1.)
    bs,base,rs=probabilistic_fit(a[ids],squared[ids],w,'scale',1.)
    if not rp['success'] or not rs['success']:raise RuntimeError(str((rp,rs)))
    return dict(probability=expit(a@bp),variance=np.exp(base+3*np.tanh(a@bs/3)),constant_variance=float(np.exp(base)),frequency=float(expit(logit)),event=event,threshold=threshold,scale=scale,bp=bp,bs=bs)
