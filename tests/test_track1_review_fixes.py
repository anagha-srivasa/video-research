import numpy as np
from numpy.testing import assert_allclose
from track1.numerics import wavelet_matrix,basis
from track1_lab.ring_topology import predicate,curves,check,guard
from track1_lab.residual import fit_residual,evaluate
from track1_lab.joint_covariance import augmented,jacobian,score

def circle(centre,radius,m=64):
 a=np.arange(m)*2*np.pi/m;return np.asarray(centre)+radius*np.c_[np.cos(a),np.sin(a)]

def test_joint_topology_rejects_orphan_touching_and_nested_same_role():
 outer=circle([0,0],3);hole=circle([0,0],1)[::-1];far=circle([5,0],1)[::-1]
 assert predicate([outer,hole],['outer','hole'])['ring_topology_valid']
 assert predicate([outer,hole],['outer','hole'])['ring_parent']==[-1,0]
 assert not predicate([outer,far],['outer','hole'])['ring_topology_valid']
 assert not predicate([outer,hole[::-1]],['outer','outer'])['ring_topology_valid']
 assert not predicate([circle([0,0],1),circle([2,0],1)],['outer','outer'])['ring_topology_valid']
 island=circle([0,0],.3);assert predicate([outer,hole,island],['outer','hole','outer'])['ring_parent']==[-1,0,1]

def test_joint_guard_checks_hierarchy_and_preserves_position_scale():
 allocation=(24,8);outer=circle([0,0],3,24);hole=circle([0,0],.4,8);h=np.r_[0.,0.,(wavelet_matrix(24)@outer).ravel(),(wavelet_matrix(8)@hole).ravel(),0.];c=np.zeros((4,67));c[0]=h;c[1,:2]=[.2,.3]
 translation=wavelet_matrix(8)@np.tile([3.,0.],(8,1));c[1,50:66]=translation.ravel();assert not check(c,allocation,('outer','hole'))['ring_topology_valid']
 fixed=guard(c,allocation,('outer','hole'));assert fixed['quality']['ring_topology_valid'];assert check(fixed['coefficients'],allocation,('outer','hole'),samples=2048)['ring_topology_valid'];assert_allclose(fixed['coefficients'][:,:2],c[:,:2],atol=1e-8);assert_allclose(fixed['coefficients'][:,-1],c[:,-1],atol=0)
 c[1,50:66]=0;c[0,50:66]=(wavelet_matrix(8)@circle([6,0],.4,8)).ravel();assert not guard(c,allocation,('outer','hole'))['quality']['ring_topology_valid']

def residual_arguments():
 t=np.linspace(0,1.2,61);h=np.zeros((len(t),35));h[:,0]=.02*np.sin(9*t);h[:,1]=.01*t;var=np.full_like(h,1e-6);o=dict(regularization=1/3,variance_power=.25,provenance_power=.5,loss='l2');return t,h,var,np.ones(len(t)),np.ones(len(t)),o,np.ones(35)

def test_residual_is_causal_orthogonal_and_keeps_coarse_state():
 args=list(residual_arguments());a=fit_residual(*args,count=2);args[1]=args[1].copy();args[1][args[0]>1]=1e8;b=fit_residual(*args,count=2)
 assert_allclose(a['coarse'],b['coarse'],atol=0);assert_allclose(a['residual'],b['residual'],atol=0);assert a['orthogonality_error']<1e-10
 from track1_lab.temporal import fit
 keep=args[0]<=1;c=fit(args[0][keep],args[1][keep],args[2][keep],args[3][keep],args[4][keep],args[5],args[6]);assert_allclose(a['coarse'],c['coefficients'],atol=0)
 prediction=evaluate(a,np.linspace(0,1,21));assert_allclose(prediction[:,2:],basis(2*np.linspace(0,1,21)-1)@a['coarse'][:,2:],atol=0)

def test_augmented_cross_terms_match_residual_derivatives_and_psd_propagation():
 model=fit_residual(*residual_arguments(),count=3);x=np.array([.63]);eps=1e-6
 numerical=(evaluate(model,x+eps)-evaluate(model,x-eps))/(2*eps);analytic=evaluate(model,x,1);assert_allclose(numerical,analytic,atol=1e-8)
 for d in [0,1]:assert_allclose(jacobian(float(x[0]),3,d)@augmented(model),evaluate(model,x,d)[0,:2],atol=1e-12)
 rng=np.random.default_rng(3);a=rng.normal(size=(14,14));cov=a@a.T;j=np.vstack([jacobian(1.,3),jacobian(1.,3,1)]);assert np.linalg.eigvalsh(j@cov@j.T).min()>0
 future=evaluate(model,np.array([1.,1.1]));assert_allclose(future[1],future[0]+.1*evaluate(model,np.array([1.]),1)[0],atol=1e-12)
 result=score(np.zeros(2),np.eye(2));assert result['joint_coverage_0.95'] and result['nees']==0

def test_conditional_joint_ensemble_is_repeatable_and_keeps_cross_terms():
 from track1_lab.joint_covariance import ensemble
 args=list(residual_arguments());keep=args[0]<=1.;args[:5]=[x[keep] for x in args[:5]];parameters=dict(count=2,strength=1.,threshold=4.)
 a=ensemble(*args,parameters,seed=12,members=6);b=ensemble(*args,parameters,seed=12,members=6)
 assert_allclose(a,b,atol=0,rtol=0);assert a.shape==(12,12) and np.linalg.eigvalsh(a).min()>0;assert np.linalg.norm(a[:8,8:])>0

def test_unknown_quality_blocks_training_and_reference_derivatives_use_seconds():
 from track1_lab.scientific import training_eligible,derivative_references
 assert not training_eligible(dict(identity_continuous=None),['identity_continuous'])
 assert not training_eligible(dict(identity_continuous=False),['identity_continuous'])
 assert training_eligible(dict(identity_continuous=True),['identity_continuous'])
 t=np.arange(21)*50000+1000000;position=np.c_[3+4*np.arange(21)*.05,8-2*np.arange(21)*.05];refs=derivative_references(t,position,.1)
 for key in ['local_polynomial','regularized_spline']:assert_allclose(refs[key],np.tile([4.,-2.],(21,1)),atol=1e-10)

def test_hierarchy_renderer_preserves_island_inside_hole():
 from track1_lab.ring_topology import render
 allocation=(16,8,8);roles=('outer','hole','outer');shapes=[circle([0,0],radius,m) for radius,m in zip([2.5,1.5,.5],allocation)];h=np.r_[0.,0.,np.concatenate([(wavelet_matrix(m)@p).ravel() for p,m in zip(shapes,allocation)]),np.log(.1)];camera=dict(intrinsic_native=[[100.,0,40],[0,100.,40],[0,0,1]],distortion=[0.,0.,0.,0.],native_size=[100,100]);mask=render(h,allocation,roles,[-1,0,1],camera,(100,100))
 assert mask[40,40]==1 and mask[40,50]==0 and mask[40,60]==1
