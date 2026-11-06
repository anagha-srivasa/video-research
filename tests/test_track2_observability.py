import numpy as np
import pytest
import torch
from track2_observability.model import Dynamics
from track2_observability.data import candidate_audit


def test_nearest_incomplete_does_not_hide_usable_alternative():
    records=[dict(timestamp_us=1003000,mask_uid='a',complete=False,position=[0.,0.]),dict(timestamp_us=1008000,mask_uid='b',complete=True,position=[1.,2.])]
    result=candidate_audit(records,900000,1000000,10000)
    assert not result['nearest_usable'] and result['any_usable'] and result['nearest_complete_mask_uid']=='b'
    assert not candidate_audit(records,1009000,1000000,10000)['any_usable']


@pytest.mark.parametrize('family',['anchored_linear','anchored_stable','anchored_polynomial'])
def test_exact_anchor_derivative_and_singular_generator(family):
    torch.manual_seed(42);model=Dynamics(family).double();x=torch.randn(3,4,8,dtype=torch.float64)
    with torch.no_grad():
        model.velocity.weight.normal_(std=.1);model.acceleration.weight.normal_(std=.1)
        if family=='anchored_polynomial':model.jerk.weight.normal_(std=.1)
    zero=model(x,torch.tensor([0.],dtype=torch.float64));torch.testing.assert_close(zero[:,:, :2],torch.zeros_like(zero[:,:,:2]),atol=0,rtol=0)
    h=.37;eps=1e-5;values=model(x,torch.tensor([h-eps,h,h+eps],dtype=torch.float64))
    derivative=(values[:,2,:2]-values[:,0,:2])/(2*eps)
    torch.testing.assert_close(derivative,values[:,1,2:],atol=1e-8,rtol=1e-7)
    assert torch.isfinite(values).all()
    if family=='anchored_stable':assert torch.linalg.eigvalsh(model.generator()+model.generator().T).max()<0


def test_overlap_retention_contract_and_exact_known_incoming_decomposition():
    from track2_observability.memory import overlap,filled
    from numpy.polynomial.legendre import leggauss,legvander
    from track2_measurement.math_checks import transport_matrix
    np.testing.assert_allclose(overlap(0)[0],np.eye(4),atol=1e-14)
    np.testing.assert_array_equal(overlap(1)[0],np.zeros((4,4)))
    coefficients=np.array([.7,-.3,.1,.02]);z,w=leggauss(8)
    metric=np.sqrt(2/(2*np.arange(4)+1))
    for h in [.25,.5,1.]:
        f,b=overlap(h);assert np.linalg.norm(metric[:,None]*f/metric[None,:],2)<=1+1e-12
        t=1-2*h+(z+1)*h
        incoming=(2*np.arange(4)+1)/2*(legvander(t,3).T@(w*h*(legvander(t+2*h,3)@coefficients)))
        np.testing.assert_allclose(f@coefficients+incoming,transport_matrix(2*h)@coefficients,atol=1e-13)
        constant=np.zeros((1,4,1));constant[:,0]=3
        np.testing.assert_allclose(filled(constant,h),constant,atol=1e-13)


def test_explicit_zero_epochs_and_zero_selection(monkeypatch):
    from track2_observability import train
    model,result=train.fit('nearest_position',17,epochs=0)
    assert result['selected_epochs']==0 and len(result['history'])==1
    assert result['initial_hash']==result['final_hash']
    monkeypatch.setitem(train.C,'maximum_epochs',1);monkeypatch.setitem(train.C,'minimum_improvement',1e12)
    _,result=train.fit('nearest_position',17,held='P02')
    assert result['selected_epochs']==0 and result['initial_hash']==result['final_hash']


def test_information_control_never_imports_other_queries():
    from track2_observability.relations import features
    x=np.arange(2*4*11,dtype=float).reshape(2,4,11)
    y=features(x,'shuffled').reshape(2,4,11)
    np.testing.assert_array_equal(y[:,:,:5],x[:,:,:5])
    for i in range(2):assert sorted(map(tuple,y[i,:,5:]))==sorted(map(tuple,x[i,:,5:]))


def test_stage_resume_rejects_partial_dependency_and_output_changes(tmp_path,monkeypatch):
    from track2_observability import common
    from pipeline.common import ROOT
    for name in ['R','D','W']:
        folder=tmp_path/name;folder.mkdir();monkeypatch.setattr(common,name,folder)
    identity={'helper':'original'};monkeypatch.setattr(common,'dependencies',lambda:dict(identity))
    def incomplete():raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):common.stage('partial',incomplete)
    identity['helper']='changed'
    with pytest.raises(ValueError,match='dependencies changed'):common.stage('partial',lambda:None)
    def completed():(common.D/'output.json').write_text('{}')
    completed()  # Existing identical output from an interrupted attempt must still be fingerprinted.
    common.stage('complete',completed)
    common.stage('complete',lambda:pytest.fail('Should reuse verified output'))
    (common.D/'output.json').write_text('{"corrupt":true}')
    with pytest.raises(ValueError,match='Integrity mismatch'):common.stage('complete',completed)
