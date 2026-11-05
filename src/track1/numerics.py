"""Double precision reference kernels; no explicit matrix inverse."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve, lstsq
import pywt

@lru_cache(maxsize=32)
def spline_basis(m=16, n=64, degree=3, derivative=0):
    # Fold the extended cardinal basis onto M periodic coefficients.
    knots=np.arange(-degree,m+degree+1,dtype=float)/m
    coefficients=np.eye(m)[np.arange(m+degree)%m]
    spline=BSpline(knots,coefficients,degree,extrapolate='periodic')
    return spline(np.arange(n)/n,nu=derivative)

@lru_cache(maxsize=8)
def wavelet_matrix(m=16, name='db2'):
    a,d=pywt.dwt(np.eye(m),name,mode='periodization',axis=0)
    return np.concatenate([a,d],axis=0)

def weighted_solve(phi,y,weights,prior,force_qr=False):
    """Solve regularised WLS and posterior covariance with QR fallback."""
    weighted=phi*np.sqrt(weights[:,None]);rhs=y*np.sqrt(weights)[:,None] if y.ndim==2 else y*np.sqrt(weights)
    gram=weighted.T@weighted+prior
    condition=float(np.linalg.cond(gram));method='cholesky'
    if not force_qr and np.isfinite(condition) and condition<1e10:
        try:
            factor=cho_factor(gram,lower=True,check_finite=False)
            fit=cho_solve(factor,weighted.T@rhs,check_finite=False)
            cov=cho_solve(factor,np.eye(len(prior)),check_finite=False)
            return fit,cov,condition,method
        except np.linalg.LinAlgError:pass
    # Eigen square root also permits semidefinite (e.g. unpenalised constant) priors.
    values,vectors=np.linalg.eigh(prior)
    root=np.sqrt(np.maximum(values,0))[:,None]*vectors.T
    design=np.concatenate([weighted,root]);target=np.concatenate([rhs,np.zeros((len(prior),)+y.shape[1:])])
    fit=lstsq(design,target,lapack_driver='gelsy')[0]
    # Covariance from the augmented least-squares factor, not normal-equation inversion.
    _,r=np.linalg.qr(design,mode='reduced')
    from scipy.linalg import solve_triangular
    ri=solve_triangular(r,np.eye(len(prior)))
    cov=ri@ri.T
    return fit,cov,condition,'qr'

def huber_weights(residual,delta=1.5):
    return np.minimum(1.,delta/np.maximum(np.abs(residual),1e-12))

def basis(tau,derivative=0,degree=3):
    from numpy.polynomial.legendre import legval,legder
    x=np.asarray(tau)
    return np.stack([legval(x,legder(np.eye(degree+1)[i],m=derivative)) for i in range(degree+1)],axis=-1)
