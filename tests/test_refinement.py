import numpy as np
from numpy.testing import assert_allclose
from refinement.kernels import periodic_design,spatial_fit,spatial_errors,temporal_fit
from refinement.data import holdout
from track1.numerics import basis,spline_basis
from refinement.kernels import nearest_curve

def test_refinement_preserves_fixed_spline_and_coefficient_budget():
    assert_allclose(periodic_design(np.arange(64)/64),spline_basis(),atol=1e-14)
    angle=np.arange(128)*2*np.pi/128;y=np.c_[50+20*np.cos(angle),40+12*np.sin(angle)]
    options=dict(samples=128,reparameterize=4,curvature=.001,loss='huber',delta=1.5,irls=3)
    fit=spatial_fit(y,options);assert fit['controls'].shape==(16,2) and fit['valid']
    assert np.all(np.diff(fit['parameters'])>0) and fit['parameters'][0]==0
    error,var,distance=spatial_errors(y,fit);assert np.sqrt(np.mean(error**2))<.1 and np.all(var>0)
    assert np.linalg.eigvalsh(fit['covariance']).min()>0

def test_refined_temporal_fit_exact_physical_polynomial_and_no_target_use():
    rng=np.random.default_rng(41);t=np.linspace(0,1,61);c=rng.normal(size=(4,35))*.1;y=basis(2*t-1)@c
    hidden=holdout(t,'gap100');keep=~hidden;options=dict(variance_power=0,regularization=0,loss='l2',provenance_power=0)
    result=temporal_fit(t[keep],y[keep],np.ones_like(y[keep]),np.ones(keep.sum()),np.ones(keep.sum()),options,np.ones(35))
    assert result['coefficients'].shape==(4,35)
    assert_allclose(result['coefficients'],c,atol=1e-8)
    assert_allclose(basis(2*t[hidden]-1)@result['coefficients'],y[hidden],atol=1e-8)
    assert result['neff']>=6
    # Targets are never a fitter input, including endpoint holdouts.
    assert (t[holdout(t,'endpoint100')]>=.9).all()

def test_certified_distance_matches_exhaustive_segments():
    rng=np.random.default_rng(19)
    for _ in range(10):
        p=rng.normal(size=(16,2))*10;points=rng.normal(size=(70,2))*20
        a=periodic_design(np.arange(256)/256)@p;ab=np.roll(a,-1,axis=0)-a
        delta=points[:,None]-a;f=np.clip(np.sum(delta*ab,axis=2)/np.maximum(np.sum(ab*ab,axis=1),1e-20),0,1)
        brute=np.sqrt(np.min(np.sum((delta-f[:,:,None]*ab)**2,axis=2),axis=1))
        distance,u=nearest_curve(points,p)
        assert_allclose(distance,brute,atol=1e-11)
        assert np.all((u>=0)&(u<=1))

def test_reliability_distinguishes_calibrated_and_overconfident_intervals():
    from refinement.diagnostics import reliability
    rng=np.random.default_rng(73);error=rng.normal(size=30000);variance=np.ones(len(error));calibrated=reliability(error,variance);overconfident=reliability(error,variance*.1)
    assert abs(calibrated['0.5']['coverage']-.5)<.015
    assert abs(calibrated['0.95']['coverage']-.95)<.01
    assert abs(calibrated['mean_scalar_nis']-1)<.04
    assert overconfident['0.95']['coverage']<.6
    assert calibrated['gaussian_nll']<overconfident['gaussian_nll']

def test_periodic_basis_closes_through_second_derivative():
    for derivative in range(3):
        p=periodic_design(np.array([0.,1.]),derivative=derivative)
        assert_allclose(p[0],p[1],atol=1e-12)
