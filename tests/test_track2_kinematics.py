import numpy as np
import pytest
import torch
from track2_kinematics.model import Kinematics, baseline, predict, normalization, tensors, loss


@pytest.mark.parametrize('family', ['kinematic_gru', 'kinematic_mlp'])
def test_initial_predictor_exactly_preserves_cv_and_translation(family):
    torch.manual_seed(5)
    model = Kinematics(family)
    history = np.arange(32).reshape(2,4,4)/8
    norm = dict(x_scale=[1.]*4, y_scale=[1.]*4)
    np.testing.assert_array_equal(predict(model,norm,history), baseline(history))
    with torch.no_grad():
        model.head.weight.normal_(0,.1)
    translated = history.copy()
    translated[:,:,:2] += [8.,-4.]
    a, b = predict(model,norm,history), predict(model,norm,translated)
    np.testing.assert_allclose(b[:,:,:2], a[:,:,:2]+[8.,-4.], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a[:,:,2:],b[:,:,2:])


def test_constant_position_has_zero_velocity_and_no_future_interpolation():
    history = np.ones((2,4,4))
    history[:,-1,:2] = [7.,3.]
    value = baseline(history,True)
    np.testing.assert_array_equal(value[:,:,2:],0)
    np.testing.assert_array_equal(value[:,:,:2],np.tile([7.,3.],(2,3,1)))
    with pytest.raises(ValueError):
        baseline(np.zeros((2,5,4)))


def test_hidden_nonfinite_targets_have_zero_gradient():
    prediction = torch.zeros(2,3,4,requires_grad=True)
    target = torch.ones_like(prediction)
    target[:,1:] = float('nan')
    mask = torch.tensor([[True,False,False]]*2)
    value = loss(prediction,target,mask)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(prediction.grad).all()
    assert prediction.grad[:,1:].abs().sum() == 0
    assert prediction.grad[:,0].abs().sum() > 0
    with pytest.raises(ValueError):
        loss(prediction,target,mask & False)


def test_normalization_excludes_validation_and_stopping_participant():
    class Data:
        def __init__(self):
            rng = np.random.default_rng(18)
            self.x = rng.normal(size=(3,4,157)).astype(np.float32)
            self.y = rng.normal(size=(3,3,144)).astype(np.float32)
            self.mask = np.ones((3,3),bool)
            self.participants = np.array(['P02','P03','P08'])
            self.split = np.array(['train','train','validation'])
    data = Data()
    norm = normalization(data,np.array([0]))
    data.x[1:] *= 1e5
    data.y[1:] *= -1e5
    assert normalization(data,np.array([0])) == norm
    assert norm['training_participants'] == ['P02']
    with pytest.raises(ValueError):
        normalization(data,np.array([2]))
    x,_,_ = tensors(data,norm)
    data.y *= 1e6
    changed,_,_ = tensors(data,norm)
    torch.testing.assert_close(x,changed,rtol=0,atol=0)


def test_only_kinematics_enters_predictor():
    # Full Track 1 coefficients/uncertainty never enter this deployed API.
    model = Kinematics('kinematic_mlp')
    with pytest.raises(ValueError):
        model(torch.zeros(2,4,157))
