"""Small joint position/velocity block, including coarse/residual cross terms.

This is a conditional perturbation ensemble, not independently certified label
uncertainty. Only held-out 2D position can be calibrated with existing labels.
"""
import numpy as np
from scipy.stats import chi2,norm
from track1.numerics import basis
from .residual import atoms,fit_residual

def augmented(model):
 effective=model['coarse'][:,:2]-model['projection']@model['residual']
 return np.concatenate([effective.ravel(),model['residual'].ravel()])

def jacobian(time,count,derivative=0):
 row=np.r_[basis(2*np.asarray(time)-1,derivative)*2**derivative,atoms(time,count,derivative)]
 return np.kron(row,np.eye(2))

def ensemble(t,h,variance,prior,quality,options,scale,parameters,seed,members=24):
 rng=np.random.default_rng(seed);models=[]
 for _ in range(members):
  noisy=np.asarray(h).copy();noisy[:,:2]+=rng.normal(size=noisy[:,:2].shape)*np.sqrt(np.maximum(variance[:,:2],0));weights=np.asarray(quality)*rng.exponential(1.,len(t));models.append(augmented(fit_residual(t,noisy,variance,prior,weights,options,scale,**parameters)))
 values=np.stack(models);covariance=np.cov(values,rowvar=False);floor=max(float(np.trace(covariance))/len(covariance)*1e-8,1e-14);covariance+=np.eye(len(covariance))*floor
 return covariance

def score(error,covariance):
 error=np.asarray(error,float);covariance=np.asarray(covariance,float);chol=np.linalg.cholesky(covariance);white=np.linalg.solve(chol,error);nees=float(white@white);d=len(error);diagonal=np.sqrt(np.diag(covariance));out=dict(nees=nees,nll=float(.5*(nees+2*np.log(np.diag(chol)).sum()+d*np.log(2*np.pi))),width95=float(np.mean(2*norm.ppf(.975)*diagonal)),variance=float(np.trace(covariance)),squared_error=float(error@error))
 for probability in [.5,.8,.9,.95]:out[f'joint_coverage_{probability}']=bool(nees<=chi2.ppf(probability,d));out[f'marginal_coverage_{probability}']=float(np.mean(abs(error)<=norm.ppf((1+probability)/2)*diagonal))
 return out
