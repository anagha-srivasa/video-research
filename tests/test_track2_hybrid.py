import numpy as np
import pytest
from track2_hybrid.imm import interaction,moments,initial,update,forecast,filter_history,transition_probability,covariance

P=dict(process_density=.01,measurement_multiplier=1.,switch_rate=.2)

def test_markov_semigroup_and_orientation():
    np.testing.assert_allclose(transition_probability(.2,.3)@transition_probability(.7,.3),transition_probability(.9,.3))
    x=np.array([[0.,0,0,0],[10.,0,0,0]]);p=np.array([np.eye(4),np.eye(4)])
    xx,pp,mu=interaction(x,p,np.array([.8,.2]),np.array([[.9,.1],[.4,.6]]))
    np.testing.assert_allclose(mu,[.8,.2]);np.testing.assert_allclose(xx[:,0],[1.,6.])
    np.testing.assert_allclose(pp[:,0,0],[10.,25.])

def test_between_mode_uncertainty():
    m,p=moments([[0,0,0,0],[2,0,0,0]],[np.eye(4),np.eye(4)],[.5,.5])
    assert m[0]==1 and p[0,0]==2

def test_causal_hidden_target_invariance():
    t=np.arange(0,1.21,.02);p=np.c_[t,2*t];v=np.ones_like(p)*1e-5
    a=filter_history(t,p,v,1.,P);p[t>1]=np.nan;v[t>1]=-100
    b=filter_history(t,p,v,1.,P)
    np.testing.assert_array_equal(forecast(a,.5)[0],forecast(b,.5)[0])

def test_linear_cv_accuracy_and_psd():
    t=np.arange(0,1.01,.02);p=np.c_[t,2*t];v=np.ones_like(p)*1e-8
    state=filter_history(t,p,v,1.,P,'kalman_cv')
    x,c,mu=forecast(state,.5)
    np.testing.assert_allclose(x,[1.5,3.,1.,2.],atol=1e-4)
    assert np.linalg.eigvalsh(c).min()>=0
    np.testing.assert_array_equal(mu,[0.,1.])

def test_invalid_and_missing_inputs_rejected():
    with pytest.raises(ValueError):covariance([[1,2],[0,1]])
    with pytest.raises(ValueError):covariance([[1,0],[0,-1]])
    with pytest.raises(ValueError):filter_history([0,.1,.9],np.zeros((3,2)),np.ones((3,2)),1.,P)
    with pytest.raises(ValueError):forecast(initial([0,0],[1,1],P),1.1)

def test_joseph_updates_and_zero_horizon():
    s=initial([0,0],[1e-12,1e-12],P)
    for i in range(20):
        s=update(s,.02,[i*.01,-i*.01],[1e-12,1e-12])
        assert np.linalg.eigvalsh(s['covariances']).min()>=-1e-12
        assert abs(s['weights'].sum()-1)<1e-12
    m,c=moments(s['means'],s['covariances'],s['weights'])
    np.testing.assert_array_equal(forecast(s,0)[0],m)
    np.testing.assert_array_equal(forecast(s,0)[1],c)
