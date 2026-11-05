"""Retained-only circular gauge refinement; hidden targets never enter this API."""
import numpy as np
from scipy.optimize import minimize_scalar
from refinement.kernels import periodic_design
from track1.numerics import wavelet_matrix

def circular_matrix(m,shift):
 # Real periodic interpolation, including a real Nyquist contribution for even m.
 return np.fft.ifft(np.fft.fft(np.eye(m),axis=0)*np.exp(2j*np.pi*np.fft.fftfreq(m)*shift)[:,None],axis=0).real

def align(prepared,native_scales,*,limit=.5,smoothness=.001,max_displacement_px=.25,geometry_guard=None):
 import copy
 p=copy.deepcopy(prepared);h=np.array(p['h'],float);variance=np.array(p['variance'],float);offsets=np.cumsum([2]+[2*m for m in p['allocation']]);previous=None;last_phase=np.zeros(len(p['allocation']));phases=[];rejected=0;proposed=0;max_error=0.
 for i in range(len(h)):
  original=h[i].copy();current=[];new_phase=[]
  for ring,m in enumerate(p['allocation']):
   sl=slice(offsets[ring],offsets[ring+1]);w=wavelet_matrix(m);controls=w.T@h[i,sl].reshape(m,2);shift=0.
   if previous is not None:
    a=(controls-controls.mean(0))/max(np.linalg.norm(controls-controls.mean(0)),1e-12);b=previous[ring];b=(b-b.mean(0))/max(np.linalg.norm(b-b.mean(0)),1e-12)
    result=minimize_scalar(lambda x:float(np.sum((circular_matrix(m,x)@a-b)**2)+smoothness*(x-last_phase[ring])**2),bounds=(-limit,limit),method='bounded',options={'xatol':1e-4,'maxiter':24});shift=float(result.x)
   transform=circular_matrix(m,shift);candidate=transform@controls;u=np.arange(128)/128
   error=float(np.max(np.linalg.norm(periodic_design(u,m=m)@candidate-periodic_design((u+shift/m)%1,m=m)@controls,axis=1))*native_scales[i]);max_error=max(max_error,error)
   if error>max_displacement_px:shift=0.;transform=np.eye(m);candidate=controls;rejected+=1
   if abs(shift)>1e-4:proposed+=1
   h[i,sl]=(w@candidate).ravel();channel=np.kron(w@transform@w.T,np.eye(2));variance[i,sl]=(channel**2)@variance[i,sl]
   current.append(candidate);new_phase.append(shift)
  if geometry_guard is not None and not geometry_guard(i,h[i],original):
   h[i]=original;variance[i]=p['variance'][i];current=[wavelet_matrix(m).T@original[offsets[r]:offsets[r+1]].reshape(m,2) for r,m in enumerate(p['allocation'])];new_phase=[0.]*len(current);rejected+=1
  phases.append(new_phase);previous=current;last_phase=np.array(new_phase)
 p['h']=h;p['variance']=variance
 return p,dict(offsets_controls=phases,proposed_fractional_shifts=proposed,rejected_shifts_or_frames=rejected,maximum_sampled_reparameterization_error_px=max_error,static_mask_guard_applied=geometry_guard is not None,phase_uncertainty_calibrated=False,topology_inferred_from_retained_only=True)

def phase_discontinuity(phases):
 x=np.asarray(phases,float)
 return float(np.mean(abs(np.diff(x,axis=0))>.5)) if len(x)>1 else 0.
