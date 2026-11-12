import numpy as np
import pytest
from track1_belief.estimation import (FilterConfig,fit_history,process,structured_covariance,time_mass,psd)
from track1_belief.memory import transition,decode_history


def observations(n=15):
    t=np.linspace(0,1,n);y=np.c_[.2+.3*t,-.4+.1*t];r=np.tile(np.eye(2)*1e-4,(n,1,1));g=np.array(['chain']*n)
    return t,y,r,g


def test_gls_exact_duplicates_do_not_add_information():
    t,y,r,g=observations();a=fit_history(t,y,r,g,ancestor_max_time=t)
    b=fit_history(np.repeat(t,5),np.repeat(y,5,axis=0),np.repeat(r,5,axis=0),np.repeat(g,5),ancestor_max_time=np.repeat(t,5))
    np.testing.assert_allclose(a['coefficients'],b['coefficients'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a['coefficient_covariance'],b['coefficient_covariance'],rtol=0,atol=1e-12)
    assert len(b['unique_indices'])==len(t)


def test_shared_error_does_not_average_away():
    r=np.tile(np.eye(2)*.01,(100,1,1));sigma=structured_covariance(r,['same']*100,np.eye(2)*.2)
    mean=np.zeros(200);mean[::2]=.01
    assert mean@sigma@mean==pytest.approx(.2+.01/100)


def test_time_mass_is_time_not_count():
    t=np.array([0.,.1,.3,1.]);w=time_mass(t);assert w.sum()==pytest.approx(1.)
    duplicate=time_mass(np.repeat(t,3)).reshape(-1,3).sum(1)
    np.testing.assert_allclose(w,duplicate)


def test_history_exact_affine_irregular_time():
    t=np.array([0.,.03,.17,.4,.9,1.]);y=np.c_[2+3*t,-1+.2*t];r=np.tile(np.eye(2)*.01,(len(t),1,1))
    f=fit_history(t,y,r,['a']*len(t),ancestor_max_time=t)
    np.testing.assert_allclose(decode_history(f['coefficients'],t,1),y,atol=1e-9)
    np.testing.assert_allclose(f['coefficient_covariance'],f['linear_map']@f['joint_input_covariance']@f['linear_map'].T)


def test_endpoint_constraints_and_joint_covariance():
    t,y,r,g=observations();ep=np.array([.6,.7,2.,-1.]);cov=np.eye(4)*.01;cross=np.zeros((2*len(t),4))
    f=fit_history(t,y,r,g,ancestor_max_time=t,endpoint_mean=ep,endpoint_covariance=cov,observation_endpoint_cross=cross)
    np.testing.assert_array_equal(decode_history(f['coefficients'],1.,1.),ep[:2])
    np.testing.assert_array_equal(decode_history(f['coefficients'],1.,1.,1),ep[2:])
    np.testing.assert_allclose(f['coefficient_covariance'][:4,:4],cov)
    assert not f['conditional_covariance']
    assert np.linalg.eigvalsh(f['coefficient_covariance']).min()>-1e-10


def test_history_causality_and_explicit_retrospective():
    t,y,r,g=observations()
    with pytest.raises(ValueError,match='ancestry'):fit_history(t,y,r,g)
    with pytest.raises(ValueError,match='future'):fit_history(t,y,r,g,end_time=.8,mode='retrospective')
    f=fit_history(t,y,r,g,mode='retrospective');assert not f['ancestry_verified']


def test_cv_transition_semigroup_and_joint_memory_noise():
    f1,q1=transition(.13,(.2,.7),.02);f2,q2=transition(.29,(.2,.7),.02);f,q=transition(.42,(.2,.7),.02)
    np.testing.assert_allclose(f2@f1,f,atol=1e-12)
    np.testing.assert_allclose(f2@q1@f2.T+q2,q,atol=1e-12)
    assert abs(q[0,4])>1e-8 and np.linalg.eigvalsh(q).min()>-1e-12


def test_filter_duplicate_and_future_perturbation_invariance():
    t,y,r,g=observations(20);cfg=FilterConfig(acceleration_spectral_density=0.,memory_timescales=(.2,))
    a=process(t,y,r,g,cfg,ancestor_max_time=t)
    b=process(np.repeat(t,3),np.repeat(y,3,axis=0),np.repeat(r,3,axis=0),np.repeat(g,3),cfg,ancestor_max_time=np.repeat(t,3))
    np.testing.assert_allclose(a['mean'],b['mean'][::3]);np.testing.assert_allclose(a['joint_covariance'],b['joint_covariance'][::3])
    yy=y.copy();yy[10:]+=100;c=process(t,yy,r,g,cfg,ancestor_max_time=t)
    np.testing.assert_array_equal(a['mean'][:10],c['mean'][:10]);assert a['valid'][-1]
    np.testing.assert_allclose(a['mean'][-1,2:4],[.3,.1],atol=.003)


def test_coherent_jump_not_single_anchor_outlier():
    t=np.arange(20)*.05;y=np.zeros((20,2));y[10:,0]=np.arange(10)*.1;r=np.tile(np.eye(2)*1e-5,(20,1,1));g=['one']*20
    cfg=FilterConfig(acceleration_spectral_density=1e-6,jump_spectral_density=100.,shared_bias_variance=0.,jump_nis_threshold=5.)
    isolated=process(t,y,r,g,cfg,ancestor_max_time=t)
    coherent=process(t,y,r,g,cfg,ancestor_max_time=t,coherence=np.ones(20,bool))
    assert not isolated['process_jump'].any();assert coherent['process_jump'].any()
    assert coherent['mean'][-1,0]>isolated['mean'][-1,0]
    assert isolated['measurement_weight'][11]<coherent['measurement_weight'][11]


def test_no_memory_baseline_and_unknown_ancestry():
    t,y,r,g=observations();a=process(t,y,r,g,FilterConfig(memory_timescales=()),ancestor_max_time=t)
    assert a['memory_mean'].shape==(len(t),0,2) and a['joint_covariance'].shape==(len(t),4,4)
    bad=process(t,y,r,g);assert not bad['valid'].any() and not bad['accepted'].any()
    retrospective=process(t,y,r,g,mode='retrospective');assert retrospective['valid'][-1] and not retrospective['ancestry_verified'].any()


def test_evicted_dependency_is_not_reused_as_new_evidence():
    t,y,r,g=observations(6);g=np.array(['a','b','c','a','c','c']);out=process(t,y,r,g,FilterConfig(maximum_bias_groups=2),ancestor_max_time=t)
    assert out['support_failure'][3]=='correlation_group_evicted' and not out['accepted'][3]
    assert np.isfinite(out['joint_covariance']).all()


def test_invalid_covariance_and_conflicting_provenance_fail():
    t,y,r,g=observations();r[2,0,0]=-1
    with pytest.raises(ValueError):process(t,y,r,g,ancestor_max_time=t)
    t,y,r,g=observations()
    with pytest.raises(ValueError,match='Conflicting'):fit_history(t,y,r,g,provenance_keys=['same']*len(t),ancestor_max_time=t)
    with pytest.raises(ValueError):psd(np.array([[1.,2.],[2.,1.]]))


def test_shared_covariance_floor_and_variable_loading():
    t,y,r,g=observations(50);r[:]=np.eye(2)*1e-5;shared=np.tile(np.diag([.2,.1]),(len(t),1,1))
    out=process(t,y,r,g,FilterConfig(acceleration_spectral_density=0.,shared_bias_variance=0.,robust_measurement=False),ancestor_max_time=t,shared_covariance=shared)
    assert out['covariance'][-1,0,0]>=.2 and out['covariance'][-1,1,1]>=.1
    shared[25:]*=2
    changed=process(t,y,r,g,FilterConfig(shared_bias_variance=0.),ancestor_max_time=t,shared_covariance=shared)
    assert np.linalg.eigvalsh(changed['joint_covariance']).min()>-1e-9


def test_duplicate_cannot_relabel_future_ancestry():
    t,y,r,g=observations(5);t=np.repeat(t,2);y=np.repeat(y,2,axis=0);r=np.repeat(r,2,axis=0);g=np.repeat(g,2);ancestry=t.copy();ancestry[1]+=1
    with pytest.raises(ValueError,match='Conflicting duplicate'):process(t,y,r,g,ancestor_max_time=ancestry)


def test_stale_unsupported_and_gap_reset_are_explicit():
    t=np.array([0.,.05,.1,.8,.85,.9]);y=np.zeros((6,2));r=np.tile(np.eye(2)*.01,(6,1,1));g=['a']*6;ancestry=t.copy();ancestry[3]=np.inf
    a=process(t,y,r,g,ancestor_max_time=ancestry)
    assert not a['valid'][3] and a['latest_evidence_time'][3]==.1
    assert not a['valid'][4] and a['latest_evidence_time'][4]==.85
    assert a['support_failure'][3]=='future_or_unknown_ancestry'
    assert a['support_failure'][4]=='insufficient_recent_support'
    assert not a['valid'][5]  # Only two observations since reset.


def test_transition_cache_is_immutable_and_no_memory_preserves_motion():
    f,q=transition(.1,(.2,),.1)
    with pytest.raises(ValueError):f[0,0]=3
    t,y,r,g=observations();a=process(t,y,r,g,FilterConfig(memory_timescales=()),ancestor_max_time=t);b=process(t,y,r,g,ancestor_max_time=t)
    np.testing.assert_allclose(a['mean'],b['mean'],atol=1e-10)
    np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-10)


def test_correlated_endpoint_covariance_matches_sampling():
    t,y,r,g=observations(8);sigma=structured_covariance(r,g,np.eye(2)*.002)
    endpoint_map=np.zeros((4,2*len(t)));endpoint_map[:2,-2:]=np.eye(2)
    endpoint_map[2:,-2:]=np.eye(2)/(t[-1]-t[-2]);endpoint_map[2:,-4:-2]=-np.eye(2)/(t[-1]-t[-2])
    endpoint=endpoint_map@y.ravel();epcov=endpoint_map@sigma@endpoint_map.T;cross=sigma@endpoint_map.T
    fit=fit_history(t,y,r,g,ancestor_max_time=t,shared_bias_covariance=np.eye(2)*.002,
                    endpoint_mean=endpoint,endpoint_covariance=epcov,observation_endpoint_cross=cross)
    rng=np.random.default_rng(903);noise=rng.multivariate_normal(np.zeros(2*len(t)),sigma,size=20000)
    samples=np.c_[noise,noise@endpoint_map.T]@fit['linear_map'].T
    empirical=np.cov(samples,rowvar=False)
    np.testing.assert_allclose(np.diag(empirical),np.diag(fit['coefficient_covariance']),rtol=.04,atol=1e-6)
    assert not fit['conditional_covariance']


def test_isolated_corruption_is_robustly_downweighted_without_event():
    t=np.arange(30)*.05;y=np.zeros((30,2));y[15]=[20.,-20.];r=np.tile(np.eye(2)*.01,(30,1,1))
    out=process(t,y,r,['a']*30,FilterConfig(acceleration_spectral_density=.01),ancestor_max_time=t)
    assert out['measurement_weight'][15]<.001
    assert not out['event'].any() and not out['process_jump'].any()
    assert np.linalg.norm(out['mean'][-1,:2])<.02
