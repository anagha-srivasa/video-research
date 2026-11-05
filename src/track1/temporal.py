"""Fixed Legendre coordinates, covariance, derivatives and physical collocation."""
from dataclasses import dataclass
import numpy as np
from .numerics import basis,huber_weights,weighted_solve

@dataclass
class FunctionalEntityState:
    coefficients:np.ndarray
    coefficient_cov:np.ndarray
    endpoint:np.ndarray
    endpoint_velocity:np.ndarray
    endpoint_accel:np.ndarray
    endpoint_cov:np.ndarray
    endpoint_cross_cov:np.ndarray
    effective_samples:float
    max_gap_seconds:float
    span_seconds:float
    weighted_fit_rmse:float
    condition_score:float
    valid:bool
    solver:str

def evaluate(coefficients,tau,seconds=1.):
    return tuple(basis(tau,k)@coefficients*(2/seconds)**k for k in range(3))

def collocation(coefficients,seconds=1.):
    nodes,weights=np.polynomial.legendre.leggauss(4)
    h,v,a=evaluate(coefficients,nodes,seconds)
    xi=np.concatenate([v[:,:2],h[:,2:34],v[:,2:34],h[:,34:35],v[:,34:35]],axis=-1)
    xidot=np.concatenate([a[:,:2],v[:,2:34],a[:,2:34],v[:,34:35],a[:,34:35]],axis=-1)
    return dict(nodes=nodes,weights=weights*seconds/2,h=h,velocity=v,acceleration=a,xi=xi,xi_dot=xidot)

def fit_function(times,h,variance,provenance,cfg,start,observation_quality=None):
    seconds=cfg['seconds'];t=np.asarray(times)-start;n,d=h.shape;degree=cfg['degree'];k=degree+1
    tau=2*t/seconds-1;phi=basis(tau,degree=degree)
    if n==1:qtime=np.ones(1)*seconds
    else:
        # Voronoi cells clipped to observed extent: unsupported edges get no invented mass.
        edges=np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]];qtime=np.diff(edges)
    quality=np.ones(n) if observation_quality is None else observation_quality
    base=qtime[:,None]*provenance[:,None]*quality[:,None]/(variance+cfg['variance_floor'])
    prior=np.diag(cfg['lambda_modes']*np.arange(k)**2+cfg['lambda_ridge'])
    w=base.copy();condition=0.;method='cholesky';c=np.zeros((k,d));cov=np.zeros((d,k,k))
    for _ in range(cfg['irls_iterations']):
        gram=np.einsum('ni,nd,nj->dij',phi,w,phi)+prior[None]
        rhs=np.einsum('ni,nd,nd->di',phi,w,h)
        conditions=np.linalg.cond(gram);condition=float(conditions.max())
        try:
            if condition>1e10:raise np.linalg.LinAlgError()
            chol=np.linalg.cholesky(gram)
            c=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,rhs[...,None]))[...,0].T
            cov=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,np.broadcast_to(np.eye(k),(d,k,k))))
        except np.linalg.LinAlgError:
            for j in range(d):c[:,j],cov[j],_,_=weighted_solve(phi,h[:,j],w[:,j],prior,force_qr=True)
            method='qr'
        residual=h-phi@c
        w=base*huber_weights(residual/np.sqrt(variance+cfg['variance_floor']),cfg['huber_delta'])
    residual=h-phi@c
    # Calibrate approximate inverse-precision covariance for misspecification; never collapse below declared noise.
    reduced=np.sum(w*residual**2,axis=0)/max(n-k,1)
    cov*=np.maximum(reduced,1)[:,None,None]
    ell=np.stack([basis(1.,i)*(2/seconds)**i for i in range(3)])
    endpoint=ell@c;endpoint_cov=np.einsum('ai,dij,aj->da',ell,cov,ell)
    cross=np.einsum('i,dij,j->d',ell[0],cov,ell[1])
    neff=(w.sum(0)**2/np.maximum((w*w).sum(0),1e-30)).min()
    maxgap=float(max(np.diff(np.r_[0,t,seconds])))
    span=float(t[-1]-t[0])
    valid=n>=k and neff>=cfg['min_effective_samples'] and span>=cfg['min_span_seconds'] and maxgap<=cfg['max_gap_seconds'] and np.isfinite(c).all() and condition<1e10
    rmse=float(np.sqrt(np.sum(w*residual**2)/max(w.sum(),1e-30)))
    return FunctionalEntityState(c,cov,*endpoint,endpoint_cov,cross,float(neff),maxgap,span,rmse,condition,bool(valid),method)
