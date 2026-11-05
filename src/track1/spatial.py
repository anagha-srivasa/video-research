"""Contour correspondence, robust periodic splines, covariance and ROI ribbons."""
import json
import numpy as np
from scipy.special import expit
from shapely.geometry import LinearRing
from .numerics import spline_basis,wavelet_matrix,weighted_solve,huber_weights

def resample(points,n=64):
    p=np.asarray(points,dtype=float)
    p=p[np.r_[True,np.any(np.diff(p,axis=0)!=0,axis=1)]]
    if len(p)<3:raise ValueError('degenerate_ring')
    signed=np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])
    if signed<0:p=p[::-1]
    closed=np.concatenate([p,p[:1]]);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    if length[-1]<=0:raise ValueError('zero_perimeter')
    u=np.arange(n)*length[-1]/n
    return np.stack([np.interp(u,length,closed[:,d]) for d in range(2)],axis=1)

def align_seam(y,previous=None):
    centre=y.mean(0);shape=(y-centre)/max(np.linalg.norm(y-centre),1e-12)
    if previous is None:
        angle=np.arctan2(shape[:,1],shape[:,0])
        shift=int(np.argmin(np.abs(np.angle(np.exp(1j*(angle-np.pi/2))))))
    else:
        prior=(previous-previous.mean(0))/max(np.linalg.norm(previous-previous.mean(0)),1e-12)
        shift=int(np.argmin([np.sum((np.roll(shape,-i,axis=0)-prior)**2) for i in range(len(y))]))
    return np.roll(y,-shift,axis=0),shift

def fit_spline(y,cfg,reliability=1.,force_qr=False,sample_weights=None):
    m=cfg['control_points'];phi=spline_basis(m,len(y),cfg['degree'])
    d2=np.roll(np.eye(m),1,axis=0)-2*np.eye(m)+np.roll(np.eye(m),-1,axis=0)
    base=np.full(len(y),reliability)
    if sample_weights is not None:base*=sample_weights
    w=base.copy();last=None
    for _ in range(cfg['irls_iterations']):
        data=phi.T@(w[:,None]*phi)
        lam=cfg['lambda_curvature']*np.trace(data)/(np.trace(d2.T@d2)+1e-12)
        prior=lam*d2.T@d2+cfg['lambda_ridge']*np.eye(m)
        p,cov,condition,solver=weighted_solve(phi,y,w,prior,force_qr)
        r=np.linalg.norm(phi@p-y,axis=1)
        sigma=max(float(np.median(r[base>0])/.67448975),cfg['sigma_floor_pixels'])
        if last is not None and np.linalg.norm(p-last)/(np.linalg.norm(last)+1e-12)<1e-4:break
        last=p.copy();w=base*huber_weights(r/(sigma+cfg['sigma_floor_pixels']),cfg['huber_delta'])
    # Relative reliability -> residual-scaled approximate posterior, explicitly calibrated later.
    variance=max(float(np.sum(w*np.sum((phi@p-y)**2,axis=1))/(2*max(w.sum()-m,1))),cfg['sigma_floor_pixels']**2)
    cov=cov*variance
    curve=phi@p;tangent=spline_basis(m,len(y),cfg['degree'],1)@p
    normals=np.stack([tangent[:,1],-tangent[:,0]],axis=1);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    normal_var=np.maximum(np.einsum('qi,ij,qj->q',phi,cov,phi),0)
    return dict(controls=p,covariance=cov,curve=curve,normals=normals,normal_var=normal_var,
                residual_normal=np.sum((y-curve)*normals,axis=1),rmse=float(np.sqrt(np.mean((curve-y)**2))),
                condition=condition,solver=solver,self_intersection=not LinearRing(curve).is_simple)

def dominant_ring(row,transform):
    rings=json.loads(row['contours_json']);outer=[r for r in rings if r['role']=='outer' and len(r['points'])>=3]
    if not outer:raise ValueError('no_external_ring')
    ring=max(outer,key=lambda r:abs(r['signed_area']))
    points=np.asarray(ring['points'],float)
    native=np.c_[points,np.ones(len(points))]@transform.T
    return native[:,:2]/native[:,2:],len(outer),sum(r['role']=='hole' for r in rings)

def soft_masks(curve,normal_std,grid,core=1.,halo=2.,tau=1.):
    """Exact segment distance + winding on ROI cells; supports concave contours."""
    a=curve;b=np.roll(curve,-1,axis=0);ab=b-a
    delta=grid[:,None,:]-a[None,:,:]
    t=np.clip(np.sum(delta*ab,axis=-1)/np.maximum(np.sum(ab*ab,axis=-1),1e-12),0,1)
    distance=np.linalg.norm(delta-t[:,:,None]*ab,axis=-1);nearest=distance.argmin(axis=1)
    d=distance[np.arange(len(grid)),nearest]
    x,y=grid[:,0,None],grid[:,1,None]
    inside=np.sum(((a[:,1]>y)!=(b[:,1]>y)) & (x<(b[:,0]-a[:,0])*(y-a[:,1])/(b[:,1]-a[:,1]+1e-20)+a[:,0]),axis=1)%2==1
    d=np.where(inside,-d,d);u=t[np.arange(len(grid)),nearest]
    sigma=(1-u)*normal_std[nearest]+u*normal_std[(nearest+1)%len(curve)]
    return expit(-(d+core*sigma)/tau),expit(-(d-halo*sigma)/tau)

def wavelet_state(p,covariance,centre,scale,name='db2'):
    w=wavelet_matrix(len(p),name)
    return w@((p-centre)/scale),w@covariance@w.T/(scale*scale)
