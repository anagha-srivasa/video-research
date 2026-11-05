import numpy as np
import pytest
from numpy.testing import assert_allclose
from track1.numerics import basis
from refinement.kernels import temporal_fit as reference_fit,periodic_design
from track1_lab.temporal import fit
from track1_lab.spatial import closest_polyline
from track1_lab.common import allowed,config
from track1_lab.motion import project,velocity_native
from track1_lab.uncertainty import variance_features


def test_lab_rejects_exposed_test_participants_and_keeps_finalization_disabled():
 assert config()['finalization_authorized'] is False and config()['track2_authorized'] is False
 allowed(config()['train']+config()['validation'])
 for video in ['P06_108','P21_01','P27_105','P30_07']:
  with pytest.raises(AssertionError):allowed([video])


def test_family_kernel_reproduces_frozen_temporal_fitter_without_new_priors():
 rng=np.random.default_rng(25);t=np.sort(rng.uniform(0,1,51));h=rng.normal(size=(51,35));var=np.exp(rng.normal(size=h.shape));prior=rng.uniform(.5,1,len(t));quality=rng.uniform(.7,1,len(t));scale=np.linspace(.1,1,35);o=dict(regularization=1/3,variance_power=.25,variance_clip=100,provenance_power=.5,floor=.01,loss='l2',irls=1,mode_penalty=[0,0,1,9]);a=reference_fit(t,h,var,prior,quality,o,scale);b=fit(t,h,var,prior,quality,o,scale)
 assert_allclose(a['coefficients'],b['coefficients'],atol=1e-13);assert_allclose(a['covariance'],b['covariance'],atol=1e-13)


def test_slope_prior_preserves_exact_linear_motion_and_channel_separation():
 rng=np.random.default_rng(28);t=np.linspace(0,1,61);c=np.zeros((4,35));c[:2]=rng.normal(size=(2,35));h=basis(2*t-1)@c;o=dict(regularization=0,variance_power=0,provenance_power=0,floor=0,loss='l2',irls=1);args=(t,h,np.ones_like(h),np.ones(len(t)),np.ones(len(t)));a=fit(*args,o,np.ones(35));b=fit(*args,dict(o,slope_strength=1,endpoint_value_strength=.1,slope_window=.25,slope_groups=['position']),np.ones(35))
 assert_allclose(b['coefficients'],c,atol=1e-7)
 assert_allclose(a['coefficients'][:,2:],b['coefficients'][:,2:],atol=0)
 assert np.linalg.eigvalsh(b['covariance']).min()>0
 assert_allclose(2*basis(1.,1)@b['coefficients'],2*c[1],atol=1e-7)


def test_polyline_projection_matches_exhaustive_and_handles_repeated_vertices():
 q=np.array([[1.,1.],[3.,1.],[-2.,0.],[0.,0.]])
 p=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.],[0.,2.]])
 projected,distance=closest_polyline(q,p)
 assert_allclose(projected,[[1,0],[2,1],[0,0],[0,0]],atol=1e-12)
 assert_allclose(distance,[1,1,2,0],atol=1e-12)


def test_camera_projection_derivatives_are_in_native_pixel_seconds():
 camera=dict(intrinsic_native=[[1000.,0,960],[0,900.,540],[0,0,1]],distortion=[0.,0.,0.,0.]);position=np.array([.2,-.1]);velocity=np.array([.3,.4])
 assert_allclose(project(position,camera),[1160,450],atol=1e-10)
 assert_allclose(velocity_native(position,velocity,camera),[300,360],rtol=1e-8)


def test_local_variance_features_use_retained_contour_only_and_are_nonnegative():
 angle=np.arange(16)*2*np.pi/16;p=np.c_[30+12*np.cos(angle),20+9*np.sin(angle)];u=np.arange(128)/128;y=periodic_design(u)@p;kept=np.arange(128)%4!=0;query=y[~kept];result=dict(controls=p,covariance=np.eye(16)*.1);x=variance_features(y[kept],result,query)
 assert x.shape==(32,3) and np.all(x>=0);assert np.max(x[:,1])<1e-8;assert_allclose(x[:,2],1)


def test_topology_guard_preserves_position_and_repairs_invalid_constant_shape():
 from track1_lab.topology import guard,validity
 from track1.numerics import wavelet_matrix
 w=wavelet_matrix();angle=np.arange(16)*2*np.pi/16;circle=np.c_[np.cos(angle),np.sin(angle)];good=np.zeros(35);good[2:34]=(w@circle).reshape(-1);observed=np.tile(good,(20,1));c=np.zeros((4,35));c[0]=good;c[1,:2]=[.5,-.3];cov=np.broadcast_to(np.eye(4),(35,4,4)).copy()*.01
 unchanged=guard(c,cov,observed);assert not unchanged['guard_applied'];assert_allclose(unchanged['coefficients'],c,atol=0)
 crossing=circle[np.r_[np.arange(0,16,2),np.arange(1,16,2)]];bad=c.copy();bad[0,2:34]=(w@crossing).reshape(-1)
 assert not validity(bad).all()
 fixed=guard(bad,cov,observed);assert fixed['valid'] and fixed['guard_applied'] and validity(fixed['coefficients']).all();assert_allclose(fixed['coefficients'][:,:2],bad[:,:2],atol=0);assert np.linalg.eigvalsh(fixed['covariance']).min()>=-1e-12


def test_architecture_legendre_units_and_fixed_feature_dimension():
 from track1_lab.architecture import polynomial_fit
 rng=np.random.default_rng(56)
 for window,degree in [(.5,3),(1.,5),(.5,5)]:
  t=np.linspace(1-window,1,71);c=rng.normal(size=(degree+1,35))*.1;h=basis(2*(t-(1-window))/window-1,degree=degree)@c;result=polynomial_fit(t,h,np.ones_like(h),np.ones(len(t)),np.ones(len(t)),window,degree,0.,np.ones(35));assert result['coefficients'].shape==(degree+1,35);assert_allclose(result['coefficients'],c,atol=1e-7);assert_allclose((2/window)*basis(1.,1,degree=degree)@result['coefficients'],(2/window)*basis(1.,1,degree=degree)@c,atol=1e-6)


def test_bounded_spatial_architecture_closes_at_each_control_count():
 from track1_lab.architecture import general_fit
 angle=np.arange(128)*2*np.pi/128;y=np.c_[30+12*np.cos(angle),20+9*np.sin(angle)]
 for m in [16,24,32]:
  result=general_fit(y,m=m,steps=2);assert result['controls'].shape==(m,2) and result['valid'];curve=periodic_design(np.arange(256)/256,m=m)@result['controls'];assert np.sqrt(np.mean(closest_polyline(y,curve)[1]**2))<.1;assert np.linalg.eigvalsh(result['covariance']).min()>0


def test_degree5_collocation_uses_six_nodes_and_physical_quadrature_weights():
 from track1_lab.variants import collocation
 c=np.zeros((6,67));c[:,0]=np.polynomial.legendre.poly2leg([0,0,0,0,0,1]);nodes=collocation(c,.5)
 assert nodes['xi'].shape==(6,132)
 assert_allclose(nodes['weights'].sum(),.5,atol=1e-14)
 assert_allclose(np.sum(nodes['weights']*nodes['h'][:,0]**2),.5/11,atol=1e-14)
 assert_allclose(nodes['velocity'][:,0],20*nodes['nodes']**4,atol=1e-12)


def test_compound_correspondence_uses_retained_frames_only():
 from track1_lab.compound import encode_sequence,match_rings
 angle=np.arange(32)*2*np.pi/32;p=np.c_[np.cos(angle),np.sin(angle)]
 ring=dict(role='outer',area=3.,points=p,index=0)
 frames={i:dict(uid=str(i),allocation=(32,),roles=('outer',),rings=[ring],centre=np.zeros(2),scale=1.,position=np.zeros(2),log_scale=0.) for i in [0,1,2]}
 def fitted(frame,source,m):
  assert frame['uid'] in ['0','2'];return dict(controls=p,valid=True)
 a=encode_sequence([0,2],((32,),('outer',)),frames,fitted);frames[1]=None;b=encode_sequence([0,2],((32,),('outer',)),frames,fitted)
 assert_allclose(a[1],b[1],atol=0)
 assert a[1].shape==(2,67)
 previous=[dict(role='outer',area=1.,relative_centre=np.array([-1.,0.])),dict(role='outer',area=1.,relative_centre=np.array([1.,0.]))]
 left=dict(role='outer',area=1.,points=np.array([[-1.,0.]]));right=dict(role='outer',area=1.,points=np.array([[1.,0.]]));ordered,ambiguous=match_rings(previous,[right,left],np.zeros(2),1.)
 assert not ambiguous and ordered[0] is left and ordered[1] is right


def test_compound_render_preserves_holes_and_global_pixel_coordinates():
 from track1_lab.compound import render
 from track1.numerics import wavelet_matrix
 centre=np.array([60.,45.]);scale=10.;a=np.arange(24)*2*np.pi/24;b=np.arange(8)*2*np.pi/8;outer=centre+np.c_[10*np.cos(a),10*np.sin(a)];inner=centre+np.c_[3*np.cos(b),3*np.sin(b)];shape=np.r_[(wavelet_matrix(24)@((outer-centre)/scale)).ravel(),(wavelet_matrix(8)@((inner-centre)/scale)).ravel()];h=np.r_[.1,.05,shape,np.log(.1)];camera=dict(intrinsic_native=[[100.,0,50],[0,100.,40],[0,0,1]],distortion=[0.,0.,0.,0.],native_size=[100,100]);mask,simple,clipped=render(h,(24,8),('outer','hole'),camera,(100,100))
 assert simple and not clipped and mask[45,60]==0 and mask[45,68]==1
 assert 200<mask.sum()<400


def test_compound_guard_changes_only_the_crossing_loop():
 from track1_lab.compound import guard_loops,loop_validity
 from track1.numerics import wavelet_matrix
 a=np.arange(24)*2*np.pi/24;b=np.arange(8)*2*np.pi/8;outer=np.c_[np.cos(a),np.sin(a)];hole=.2*np.c_[np.cos(b),np.sin(b)];h=np.r_[0.,0.,(wavelet_matrix(24)@outer).ravel(),(wavelet_matrix(8)@hole).ravel(),0.];observed=np.tile(h,(12,1));c=np.zeros((4,67));c[0]=h;c[1,:2]=[.5,.2];cov=np.broadcast_to(np.eye(4),(67,4,4)).copy()*.01
 bad=c.copy();crossing=hole[[0,2,4,6,1,3,5,7]];bad[0,50:66]=(wavelet_matrix(8)@crossing).ravel();assert not loop_validity(bad[:,50:66],8).all()
 fixed=guard_loops(bad,cov,observed,(24,8));assert fixed['valid'] and fixed['applied'];assert loop_validity(fixed['coefficients'][:,50:66],8).all();assert_allclose(fixed['coefficients'][:,:50],bad[:,:50],atol=0);assert_allclose(fixed['coefficients'][:,-1],bad[:,-1],atol=0);assert fixed['factors'][0]==1.


def test_robust_local_slope_rejects_displaced_annotation_without_changing_default():
 from track1_lab.temporal import local_linear
 t=np.linspace(.8,1,11);h=np.c_[2*t,3*t];h[-2]+=[2,-3];variance=np.full_like(h,.0001);weights=np.linspace(1,2,len(t));design=np.c_[np.ones(len(t)),t-1];root=np.sqrt(weights);expected=np.linalg.lstsq(design*root[:,None],h*root[:,None],rcond=None)[0]
 ordinary=local_linear(t,h,weights,variance);robust=local_linear(t,h,weights,variance,2.)
 assert_allclose(ordinary,expected,atol=0,rtol=0)
 assert np.linalg.norm(robust[1]-[2,3])<.15*np.linalg.norm(ordinary[1]-[2,3])


def test_analytic_motion_derivatives_and_stationary_endpoints():
 from track1_lab.kinematics import truth,KINDS
 t=np.linspace(1.5,2.7,101);eps=1e-6
 for kind in KINDS:
  position,velocity=truth(t,kind,2.,.25);left,_=truth(t-eps,kind,2.,.25);right,_=truth(t+eps,kind,2.,.25)
  assert_allclose((right-left)/(2*eps),velocity,atol=1e-5,rtol=1e-6)
 for kind in ['pickup','burst']:
  _,velocity=truth(np.array([1.,2.,2.25,3.]),kind,2.,.25);assert_allclose(velocity,0,atol=1e-12)
