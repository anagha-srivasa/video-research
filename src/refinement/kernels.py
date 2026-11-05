"""Fixed-capacity cubic models with bounded fitting and explicit uncertainty."""
import numpy as np
from functools import lru_cache
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from shapely.geometry import LinearRing
from track1.numerics import weighted_solve,basis,huber_weights

def periodic_design(u,m=16,derivative=0):
    return BSpline(np.arange(-3,m+4,dtype=float)/m,np.eye(m)[np.arange(m+3)%m],3,extrapolate='periodic')(np.asarray(u),nu=derivative)

@lru_cache(maxsize=16)
def evaluation_design(samples,derivative=0):return periodic_design(np.arange(samples)/samples,derivative=derivative)

def nearest_curve(points,p,samples=256):
    """Exact polyline distance, with a certified nearest-vertex pruning bound.

    Any segment closer than current d has an endpoint within d + max_segment_length.
    If all omitted vertices are farther away, the candidate minimum is exact.
    Uncertified points expand their search, up to every segment.
    """
    a=evaluation_design(samples)@p;ab=np.roll(a,-1,axis=0)-a;length=np.linalg.norm(ab,axis=1);tree=cKDTree(a)
    distance=np.empty(len(points));u=np.empty(len(points));remaining=np.arange(len(points))
    for count in [min(8,samples),min(32,samples),samples]:
        vd,vi=tree.query(points[remaining],k=count)
        if count==1:vd=vd[:,None];vi=vi[:,None]
        candidates=np.concatenate([vi,(vi-1)%samples],axis=1)
        aa=a[candidates];bb=ab[candidates];delta=points[remaining,None,:]-aa
        fraction=np.clip(np.sum(delta*bb,axis=2)/np.maximum(np.sum(bb*bb,axis=2),1e-20),0,1)
        squared=np.sum((delta-fraction[:,:,None]*bb)**2,axis=2);local=squared.argmin(1);best=np.sqrt(squared[np.arange(len(remaining)),local])
        certified=np.ones(len(remaining),bool) if count==samples else vd[:,-1]>best+length.max()+1e-12
        take=remaining[certified];distance[take]=best[certified]
        u[take]=(candidates[np.arange(len(remaining)),local][certified]+fraction[np.arange(len(remaining)),local][certified])/samples
        remaining=remaining[~certified]
        if not len(remaining):break
    return distance,u

def spatial_fit(y,options,reliability=1.,sample_weights=None):
    y=np.asarray(y,float);n=len(y);m=16
    base=np.full(n,reliability)*(np.ones(n) if sample_weights is None else sample_weights)
    centre=np.average(y,axis=0,weights=base);target=y-centre;u=np.arange(n)/n
    d2=np.roll(np.eye(m),1,axis=0)-2*np.eye(m)+np.roll(np.eye(m),-1,axis=0)
    best=None;iterations=0
    for outer in range(options.get('reparameterize',0)+1):
        phi=periodic_design(u);w=base.copy()
        for _ in range(options.get('irls',3)):
            data=phi.T@(w[:,None]*phi)
            prior=options['curvature']*np.trace(data)/(np.trace(d2.T@d2)+1e-20)*(d2.T@d2)+1e-6*np.eye(m)
            p,cov,condition,solver=weighted_solve(phi,target,w,prior);p+=centre
            residual=np.linalg.norm(phi@p-y,axis=1)
            sigma=max(float(np.median(residual[base>0])/.67448975),1.)
            robust=np.ones(n) if options['loss']=='l2' else huber_weights(residual/(sigma+1),options.get('delta',1.5))
            w=base*robust
        distance,_=nearest_curve(y[base>0],p)
        score=float(np.average(distance**2,weights=base[base>0]));simple=LinearRing(evaluation_design(256)@p).is_simple
        # Refinement is monotonic on observed fitting data and cannot introduce a crossing.
        if best is not None and (score>best['score'] or not simple):break
        residual2=np.sum((phi@p-y)**2,axis=1)
        sigma2=max(float(np.sum(w*residual2)/(2*max(np.sum(w)-m,1))),1.)
        best=dict(controls=p,covariance=cov*sigma2,score=score,condition=condition,solver=solver,valid=bool(simple),parameters=u.copy(),iterations=outer)
        curve=phi@p;tangent=periodic_design(u,derivative=1)@p
        step=np.sum((y-curve)*tangent,axis=1)/np.maximum(np.sum(tangent*tangent,axis=1),1e-12)
        step=np.clip(step,-1/64,1/64)*options.get('parameter_step',.5);step[base==0]=0;step[0]=0
        candidate=u+step
        # Keep correspondence ordered; the first seam stays anchored.
        candidate=np.clip(candidate,0,1-1e-6)
        for j in range(1,n):candidate[j]=max(candidate[j],candidate[j-1]+.01/n)
        if candidate[-1]>=1:candidate*=((1-.01/n)/candidate[-1])
        candidate[0]=0
        if np.max(abs(candidate-u))<1e-5:break
        u=candidate;iterations+=1
    return best

def spatial_errors(y,fit):
    distance,u=nearest_curve(y,fit['controls'],512);phi=periodic_design(u)
    tangent=periodic_design(u,derivative=1)@fit['controls'];normal=np.stack([tangent[:,1],-tangent[:,0]],axis=1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    error=np.sum((y-phi@fit['controls'])*normal,axis=1)
    variance=np.maximum(np.einsum('ni,ij,nj->n',phi,fit['covariance'],phi),1e-20)
    return error,variance,distance

def temporal_fit(t,h,variance,prior,quality,options,feature_scale):
    t=np.asarray(t,float);h=np.asarray(h,float);n,d=h.shape;phi=basis(2*t-1)
    qtime=np.diff(np.r_[t[0],(t[1:]+t[:-1])/2,t[-1]]) if n>1 else np.ones(1)
    effective=np.maximum(variance,1e-12)+options.get('floor',0)*feature_scale[None]**2
    clip=options.get('variance_clip',0)
    if clip:
        middle=np.median(effective,axis=0);effective=np.clip(effective,middle/clip,middle*clip)
    base=qtime[:,None]*prior[:,None]**options.get('provenance_power',1)*quality[:,None]/effective**options.get('variance_power',1)
    base*=n/np.maximum(base.sum(0),1e-20)
    w=base.copy();reg=np.diag(np.array(options.get('mode_penalty',[0,1,4,9]))*options.get('regularization',0)+1e-8)
    for _ in range(options.get('irls',3)):
        g=np.einsum('ni,nd,nj->dij',phi,w,phi)+reg[None];rhs=np.einsum('ni,nd,nd->di',phi,w,h)
        chol=np.linalg.cholesky(g);c=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,rhs[...,None]))[...,0].T
        residual=h-phi@c
        if options['loss']=='l2':break
        scale=np.sqrt(effective+options.get('robust_floor',.01)*feature_scale[None]**2)
        r=residual/np.maximum(scale,1e-10)
        robust=huber_weights(r,options.get('delta',1.5)) if options['loss']=='huber' else 1/(1+r*r)
        w=base*robust
    cov=np.linalg.solve(chol.swapaxes(-1,-2),np.linalg.solve(chol,np.broadcast_to(np.eye(4),(d,4,4))))
    sigma2=np.maximum(np.sum(w*residual**2,axis=0)/max(n-4,1),1e-10)
    cov*=sigma2[:,None,None]
    neff=float(np.min(w.sum(0)**2/np.maximum(np.sum(w*w,axis=0),1e-30)))
    return dict(coefficients=c,covariance=cov,neff=neff,condition=float(np.linalg.cond(g).max()),weights=w)
