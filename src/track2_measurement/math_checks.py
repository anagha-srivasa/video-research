"""Exact coordinate transport and region moments, separate from frozen features."""
import numpy as np


def transport_matrix(shift):
    d=float(shift)
    if not np.isfinite(d):raise ValueError('Finite normalized-time shift required')
    return np.array([[1,d,1.5*d*d,d+2.5*d**3],[0,1,3*d,7.5*d*d],[0,0,1,5*d],[0,0,0,1]],float)


def polygon_moments(rings,roles):
    if not rings or len(rings)!=len(roles):raise ValueError('Rings and roles required')
    offset=np.asarray(rings[0],float)[0];total=np.zeros(6)
    for ring,role in zip(rings,roles):
        p=np.asarray(ring,float)-offset
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all() or role not in ('outer','hole'):raise ValueError('Invalid polygon ring')
        x,y=p.T;u,v=np.roll(p,-1,axis=0).T;cross=x*v-u*y;a=cross.sum()/2
        if abs(a)<1e-16:raise ValueError('Degenerate polygon ring')
        values=np.array([a,np.sum((x+u)*cross)/6,np.sum((y+v)*cross)/6,
            np.sum((x*x+x*u+u*u)*cross)/12,np.sum((y*y+y*v+v*v)*cross)/12,
            np.sum((2*x*y+x*v+u*y+2*u*v)*cross)/24])
        total+=values*np.sign(a)*(1 if role=='outer' else -1)
    a,mx,my,xx,yy,xy=total
    if a<=0:raise ValueError('Positive occupied region required')
    centre=np.array([mx,my])/a;cov=np.array([[xx,xy],[xy,yy]])/a-np.outer(centre,centre)
    if np.linalg.eigvalsh(cov).min() < -1e-9*max(1.,abs(cov).max()):raise ValueError('Invalid region moments')
    return a,centre+offset,cov


def region_descriptors(rings,roles,scale):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive scale required')
    rings=[np.asarray(r,float)*scale for r in rings];a,_,cov=polygon_moments(rings,roles)
    points=np.concatenate(rings);extent=np.ptp(points,axis=0)
    outer=sum(polygon_moments([p],['outer'])[0] for p,role in zip(rings,roles) if role=='outer')
    return np.array([np.log(max(extent[0],1e-8)),np.log(max(extent[1],1e-8)),np.log(a),
        cov[0,0]/np.trace(cov),cov[0,1]/np.trace(cov),(outer-a)/outer,roles.count('outer'),roles.count('hole')])
