import numpy as np
import pytest
import torch
from numpy.polynomial.legendre import legval
from track2_measurement.math_checks import transport_matrix,polygon_moments,region_descriptors
from track2_measurement.data import evidence_allowed
from track2_measurement.anchors import Model,loss


def test_exact_legendre_reindex_and_composition():
    c=np.random.default_rng(3).normal(size=(4,7));t=np.linspace(-1,1,101)
    for d in (-.7,0,.5,2):
        np.testing.assert_allclose(legval(t,transport_matrix(d)@c),legval(t+d,c),atol=1e-12)
    np.testing.assert_allclose(transport_matrix(.3)@transport_matrix(.7),transport_matrix(1.))


def test_occupied_moments_holes_orientation_and_subdivision():
    outer=np.array([[0,0],[4,0],[4,2],[0,2]],float);hole=np.array([[1,.5],[3,.5],[3,1.5],[1,1.5]],float)
    a,c,v=polygon_moments([outer,hole],['outer','hole'])
    assert a==6;np.testing.assert_allclose(c,[2,1]);np.testing.assert_allclose(np.diag(v),[(8*16/12-2*4/12)/6,(8*4/12-2/12)/6])
    def subdivide(p):return np.array([p[i]+f*(p[(i+1)%len(p)]-p[i]) for i in range(len(p)) for f in np.linspace(0,1,17+i,endpoint=False)])
    x=region_descriptors([outer,hole],['outer','hole'],2.)
    np.testing.assert_allclose(region_descriptors([subdivide(outer)[::-1]+100,subdivide(hole)+100],['outer','hole'],2.),x,atol=1e-11)


def test_unknown_ancestry_fails_closed():
    row=dict(timestamp_us=10,online_observation_allowed=True,available_at_us=10)
    assert not evidence_allowed(row,20)
    row['latest_evidence_us']=30;assert not evidence_allowed(row,20)
    row['latest_evidence_us']=15;assert evidence_allowed(row,20)


def test_position_loss_ignores_velocity_and_missing_labels():
    p=torch.zeros((2,3,4),requires_grad=True);y=torch.ones_like(p);mask=torch.tensor([[True,False,False],[True,True,False]])
    y[~mask]=float('nan');y[:,:,2:]=float('nan')
    value=loss(p,y,mask,0.);value.backward()
    assert torch.isfinite(value) and torch.isfinite(p.grad).all()
    assert torch.count_nonzero(p.grad[:,:,2:])==0
    model=Model();assert torch.count_nonzero(model(torch.zeros(2,4,8)))==0


def test_decoded_shape_excludes_position_and_log_scale():
    from track2_memory import decoded_error
    reference=np.zeros((2,4,35));pred=reference.copy();pred[:,0,0]=2;pred[:,0,-1]=7
    position,shape=decoded_error(pred,reference)
    assert position==2 and shape==0
    pred=reference.copy();pred[:,0,2]=1
    position,shape=decoded_error(pred,reference)
    assert position==0 and shape>0
