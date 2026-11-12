"""Exact linear CV/exponential-memory transitions and joint covariance propagation.

Memory is a posterior linear functional of the latent position process. It is
neither an independent sensor nor a new motion estimate. All times are seconds.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class MemoryConfig:
    timescales: tuple[float, ...] = (.125, .5, 2.)

    def __post_init__(self):
        if any(not np.isfinite(t) or t <= 0 for t in self.timescales):
            raise ValueError('Memory timescales must be finite and positive')


def transition(dt, timescales=(), acceleration_spectral_density=0.):
    """Return exact discrete transition/noise for [px,py,vx,vy,m1x,m1y,...]."""
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(acceleration_spectral_density) or acceleration_spectral_density < 0:
        raise ValueError('Nonnegative finite duration and process spectral density required')
    MemoryConfig(tuple(timescales))
    return _transition_cached(float(dt),tuple(timescales),float(acceleration_spectral_density))


@lru_cache(maxsize=512)
def _transition_cached(dt,timescales,acceleration_spectral_density):
    n=4+2*len(timescales);a=np.zeros((n,n));a[:2,2:4]=np.eye(2)
    for j,tau in enumerate(timescales):
        sl=slice(4+2*j,6+2*j);a[sl,:2]=np.eye(2)/tau;a[sl,sl]=-np.eye(2)/tau
    q=np.zeros_like(a);q[2:4,2:4]=np.eye(2)*acceleration_spectral_density
    # Van Loan block exponential integrates continuous white acceleration noise.
    van=np.block([[a,q],[np.zeros_like(a),-a.T]])*dt;e=expm(van);f=e[:n,:n];qd=e[:n,n:]@f.T
    qd=(qd+qd.T)/2;f.setflags(write=False);qd.setflags(write=False)
    return f,qd


def propagate(mean,covariance,dt,timescales=(),acceleration_spectral_density=0.):
    f,q=transition(dt,timescales,acceleration_spectral_density)
    m=np.asarray(mean,float);p=np.asarray(covariance,float)
    if m.shape!=(len(f),) or p.shape!=f.shape:raise ValueError('Joint state shape mismatch')
    result=f@p@f.T+q
    return f@m,(result+result.T)/2


def decode_history(coefficients,time,end_time,derivative=0):
    """Evaluate endpoint-power coefficients (4,2); derivatives are physical time."""
    c=np.asarray(coefficients,float);u=np.asarray(time,float)-end_time
    if c.shape!=(4,2) or derivative not in (0,1,2):raise ValueError('Expected cubic coefficients and derivative 0, 1 or 2')
    if derivative==0:b=np.stack([np.ones_like(u),u,u*u,u*u*u],axis=-1)
    elif derivative==1:b=np.stack([np.zeros_like(u),np.ones_like(u),2*u,3*u*u],axis=-1)
    else:b=np.stack([np.zeros_like(u),np.zeros_like(u),np.full_like(u,2),6*u],axis=-1)
    return b@c
