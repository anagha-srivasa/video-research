"""A fixed coarse state with a bounded, weighted-orthogonal causal residual.

The first experiment corrects position only: no unguarded shape residual is
introduced. No target after the requested state timestamp enters the fitter.
"""
import numpy as np
from scipy.interpolate import BSpline
from track1.numerics import basis
from .temporal import fit

KNOTS=np.r_[np.zeros(4),[.25,.5,.75],np.ones(4)]

def atoms(t,count,derivative=0):
 spline=BSpline(KNOTS,np.eye(7),3,extrapolate=False)
 return np.nan_to_num(spline(np.asarray(t),nu=derivative))[...,-count:]

def fit_residual(time,h,variance,prior,quality,options,scale,cutoff=1.,count=2,strength=1.,threshold=4.):
 time=np.asarray(time,float);keep=(time<=cutoff)&(time>=cutoff-1.-1e-12);t=time[keep]+(1.-cutoff);h=np.asarray(h)[keep];variance=np.asarray(variance)[keep];prior=np.asarray(prior)[keep];quality=np.asarray(quality)[keep]
 if len(t)<8:raise ValueError('Insufficient past observations for coarse/residual fit')
 coarse=fit(t,h,variance,prior,quality,options,scale);phi=basis(2*t-1);raw=atoms(t,count);w=np.median(coarse['weights'][:,:2],axis=1);root=np.sqrt(w);projection=np.linalg.lstsq(phi*root[:,None],raw*root[:,None],rcond=None)[0];psi=raw-phi@projection;error=h[:,:2]-phi@coarse['coefficients'][:,:2];gram=psi.T@(w[:,None]*psi);noise=np.maximum(np.median(variance[:,:2],axis=0),1e-12);energy=np.sum(w[:,None]*error**2,axis=0)/w.sum();excess=np.maximum(energy-noise,0);gate=excess/(excess+threshold*noise+1e-20)
 coefficients=np.linalg.solve(gram+np.eye(count)*(strength+1e-8),(psi.T@(w[:,None]*error)));coefficients*=gate[None]
 # Bound weighted residual energy by the unexplained coarse observation energy.
 power=np.sum(w[:,None]*(psi@coefficients)**2,axis=0)/w.sum();coefficients*=np.minimum(1.,np.sqrt(energy/np.maximum(power,1e-20)))[None]
 return dict(coarse=coarse['coefficients'],coarse_covariance=coarse['covariance'],residual=coefficients,projection=projection,gate=gate,cutoff=float(cutoff),count=count,orthogonality_error=float(np.max(abs(phi.T@(w[:,None]*psi)))),observations=int(keep.sum()),weights=w)

def evaluate(model,time,derivative=0,linear_tail=True):
 time=np.asarray(time,float);relative=time-model['cutoff']+1.;clipped=np.minimum(relative,1.) if linear_tail else relative;phi=basis(2*clipped-1,derivative)*2**derivative;psi=atoms(clipped,model['count'],derivative)-phi@model['projection'];value=phi@model['coarse'];value[...,:2]+=psi@model['residual']
 if linear_tail and derivative<2:
  future=relative>1.
  if np.any(future):
   endpoint=evaluate(model,np.asarray([model['cutoff']]),1,False)[0]
   if derivative==0:value[future]+=((relative[future]-1.)[:,None])*endpoint
   else:value[future]=endpoint
 if linear_tail and derivative>=2:value[relative>1.]=0
 return value
