import numpy as np
import pytest
import torch
from track2_geometry.data import descriptors,features,normalize
from track2_geometry.train import Model

class Fixture:
    def __init__(self):
        rng=np.random.default_rng(12)
        self.samples=[dict(input_state_ids=[f'{i}-{j}' for j in range(4)]) for i in range(12)]
        self.x=rng.normal(size=(12,4,157)).astype(np.float32)
        self.y=rng.normal(size=(12,3,144)).astype(np.float32)
        self.mask=np.ones((12,3),bool)
        self.split=np.array(['train']*10+['validation']*2)
        self.participants=np.array(['P02']*10+['P08']*2)


def test_shape_descriptors_translation_invariance():
    p=np.array([[0,0],[2,0],[2,1],[0,1]],float)
    np.testing.assert_allclose(descriptors([p],['outer'],2),descriptors([p+23],['outer'],2),atol=1e-12)
    with pytest.raises(ValueError):descriptors([p],['outer'],0)


def test_train_only_projection_and_missing_geometry():
    data=Fixture();g=np.random.default_rng(2).normal(size=(12,4,8));mask=np.ones((12,4),bool);mask[0,0]=False
    norm=normalize(data,np.arange(10),g,mask)
    first=features(data,norm,g,mask,'geometry');g[0,0]=np.nan
    np.testing.assert_array_equal(features(data,norm,g,mask,'geometry'),first)
    data.x[10:]=1e8;g[10:]=1e8
    assert normalize(data,np.arange(10),g,mask)==norm
    with pytest.raises(ValueError):normalize(data,np.arange(12),g,mask)


def test_arms_and_zero_residual():
    data=Fixture();g=np.ones((12,4,8));mask=np.ones((12,4),bool);norm=normalize(data,np.arange(10),g,mask)
    arrays={a:features(data,norm,g,mask,a) for a in ('kinematics','functional','availability','geometry')}
    assert np.count_nonzero(arrays['kinematics'][:,:,4:])==0
    assert np.count_nonzero(arrays['functional'][:,:,20:])==0
    np.testing.assert_array_equal(arrays['availability'][:,:,:20],arrays['functional'][:,:,:20])
    model=Model();assert sum(p.numel() for p in model.parameters())<10000
    assert torch.count_nonzero(model(torch.from_numpy(arrays['geometry'])))==0
