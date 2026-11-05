"""Exact Jacobian covariance propagation conditional on supplied input covariance."""
import numpy as np
from scipy.linalg import block_diag
from track1.numerics import wavelet_matrix

def features_and_jacobian(controls,centre,scale,position_jacobian=None):
 controls=[np.asarray(p,float) for p in controls];centre=np.asarray(centre,float);scale=float(scale)
 if scale<=0 or not np.isfinite(scale) or centre.shape!=(2,) or not all(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() for p in controls) or not np.isfinite(centre).all():raise ValueError('Finite controls/centre and positive scale required')
 count=sum(p.size for p in controls);j=np.zeros((count+3,count+3));a=np.eye(2) if position_jacobian is None else np.asarray(position_jacobian,float)
 if a.shape!=(2,2) or not np.isfinite(a).all():raise ValueError('Invalid position Jacobian')
 j[:2,count:count+2]=a;shape=[];offset=0
 for p in controls:
  w=wavelet_matrix(len(p));b=np.kron(w,np.eye(2));z=(b@(p-centre).ravel())/scale;shape.extend(z);q=p.size;j[2+offset:2+offset+q,offset:offset+q]=b/scale;j[2+offset:2+offset+q,count:count+2]=-b@np.tile(np.eye(2),(len(p),1))/scale;j[2+offset:2+offset+q,-1]=-z/scale;offset+=q
 j[-1,-1]=1/scale
 return np.r_[a@centre,shape,np.log(scale)],j

def propagate(controls,centre,scale,joint_covariance,position_jacobian=None):
 h,j=features_and_jacobian(controls,centre,scale,position_jacobian);cov=np.asarray(joint_covariance,float)
 if cov.shape!=(j.shape[1],j.shape[1]) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-10):raise ValueError('Invalid joint input covariance')
 e=np.linalg.eigvalsh(cov)
 if e.min()< -1e-9*max(abs(e).max(),1.):raise ValueError('Input covariance is not PSD')
 out=j@cov@j.T;out=(out+out.T)/2
 return dict(features=h,covariance=out,jacobian=j,conditional_on_input_covariance=True,phase_topology_identity_uncertainty_included=False,geometry_covariance_after_repair_available=False)

def independent_input_covariance(control_covariances,centre_covariance,scale_variance):
 if scale_variance<0:raise ValueError('Negative scale variance')
 # Fitter covariances are m x m under independent x/y noise. Off-diagonal control terms survive.
 return block_diag(*[np.kron(np.asarray(c,float),np.eye(2)) for c in control_covariances],np.asarray(centre_covariance,float),np.array([[scale_variance]]))
