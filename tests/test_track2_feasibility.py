import numpy as np
import pytest
import torch
from track2_feasibility.models import Dynamics,loss
from track2_feasibility.data import safe_covariance_code,target_reason,INPUT_DIM

def test_stable_generator_and_cached_propagation():
 torch.manual_seed(17);m=Dynamics('stable_koopman');x=torch.randn(2,4,INPUT_DIM);m.eval()
 with torch.no_grad():raw,_=m(x)
 eig=np.linalg.eigvals(m.generator().detach().numpy().astype(float));assert eig.real.max()<1e-5
 m.compile_inference()
 with torch.no_grad():cached,_=m(x)
 torch.testing.assert_close(cached,raw,rtol=0,atol=0)
 m.train();assert not m._cache.numel()


def test_missing_horizons_have_no_loss_gradient():
 prediction=torch.zeros(2,3,144,requires_grad=True);target=torch.ones_like(prediction);mask=torch.tensor([[True,False,False],[True,False,False]]);value=loss(prediction,target,mask,torch.tensor(0.));value.backward();assert prediction.grad[:,0].abs().sum()>0;assert prediction.grad[:,1:].abs().sum()==0
 with pytest.raises(ValueError):loss(prediction,target,torch.zeros_like(mask),torch.tensor(0.))


def test_target_ledger_cannot_cross_gap_or_epoch():
 origin=dict(window_end_us=1000000,episode_id='e',epoch='a');r=dict(valid=True,kinematics_valid=True,episode_id='e',epoch='a',window_end_us=1250000,latest_input_us=1250000,kinematics_latest_input_us=1250000);lut={('P02_122','t',1250000):r}
 assert target_reason(lut,'P02_122','t',origin,250000) is None
 assert target_reason(lut,'P02_122','t',origin,500000)=='missing_future_state'
 r['epoch']='b';assert target_reason(lut,'P02_122','t',origin,250000)=='source_epoch_boundary'
 code,info=safe_covariance_code(np.zeros((4,4)));assert np.isfinite(code).all() and info['repaired']
 with pytest.raises(ValueError):safe_covariance_code(-np.eye(4))


def test_normalization_never_uses_held_participant():
 from track2_feasibility.data import normalization
 class Data:
  def __init__(self):
   rng=np.random.default_rng(5);self.x=rng.normal(size=(3,4,157)).astype(np.float32);self.y=rng.normal(size=(3,3,144)).astype(np.float32);self.mask=np.ones((3,3),bool);self.participants=np.array(['P02','P03','P08']);self.samples=[dict(input_state_ids=[f'{i}:{t}' for t in range(4)],target_state_ids=[f'{i}:future:{t}' for t in range(3)]) for i in range(3)]
 d=Data();a=normalization(d,np.array([0]));d.x[1:]*=1e5;d.y[1:]*=-1e5;b=normalization(d,np.array([0]));assert a==b and a['training_participants']==['P02']


def test_stable_generator_energy_is_dissipative():
 torch.manual_seed(43);m=Dynamics('stable_koopman').double()
 with torch.no_grad():m.K.normal_(0,.1);m.L.normal_(0,.1);m.B.normal_(0,.1)
 a=m.generator();h=m.B@m.B.T+1e-4*torch.eye(32,dtype=torch.float64);energy=a.T@h+h@a
 assert torch.linalg.eigvalsh((energy+energy.T)/2).max()<1e-10
