import numpy as np
import torch
from track2_signal.data import category,source_slope,relation_vector
from track2_signal.memory import joint_covariance
from track2_signal.probes import features,probability_scores
from track2_observability.memory import overlap
from track2_observability.model import Dynamics


def test_explicit_hand_policy_excludes_handles_and_ambiguous_gloves():
    assert category(191)=='object'  # handle
    assert category(300)==category(301)=='hand'
    assert category(303)==category(304)=='glove'
    assert category(60)=='object'  # generic glove does not establish a worn hand
    assert category(None)=='unknown'


def test_actual_timestamp_slope_deduplicates_and_excludes_future():
    rows=[dict(timestamp_us=t,mask_uid=str(i),fit16_valid=True,epoch16='a',position=np.array([2*t/1e6,-t/1e6]),quality=1.,class_id=301,name='right hand') for i,t in enumerate([0,50000,120000,120000,200000,250000])]
    result=source_slope(rows,200000)
    assert result['count']==4 and result['span_us']==200000 and result['latest_us']==200000
    np.testing.assert_allclose(result['velocity'],[2,-1],atol=1e-12)
    perturbed=[dict(r,position=np.array([1e9,1e9])) if r['timestamp_us']>200000 else r for r in rows]
    np.testing.assert_allclose(source_slope(perturbed,200000)['velocity'],result['velocity'])
    assert not source_slope(rows[-2:],200000)['velocity_valid']


def test_interaction_invariants_and_time_shuffle_masks():
    obj=dict(position=np.zeros(2),velocity=np.zeros(2),velocity_valid=True)
    hand=dict(position=np.array([2.,0.]),velocity=np.array([-1.,0.]),velocity_valid=True,age_us=0,count=4,rmse=0.)
    v=relation_vector(obj,hand);assert v[4]==2 and v[5]<0 and v[6]==1
    raw=np.arange(2*4*21,dtype=float).reshape(2,4,21);raw[:,:,15:17]=1
    x=features(raw,raw,'time_shuffle').reshape(2,4,23)
    np.testing.assert_array_equal(x[:,:,:8],raw[:,:,:8]);np.testing.assert_array_equal(x[:,:,15:21],raw[:,:,15:21])
    for i in range(2):assert sorted(map(tuple,x[i,:,8:15]))==sorted(map(tuple,raw[i,:,8:15]))


def test_complete_memory_operator_and_correlated_covariance():
    cov=np.diag([.1,.2,.3,.4]);cross=cov@np.ones(4);var=float(cross.sum());metric=np.sqrt(2/(2*np.arange(4)+1))
    f,b=overlap(1);m=f+np.outer(b,np.ones(4));assert abs(np.linalg.norm(metric[:,None]*m/metric[None,:],2)-4)<1e-12
    actual=joint_covariance(f,b,cov,var,cross);np.testing.assert_allclose(actual,m@cov@m.T,atol=1e-14)


def test_zero_readout_gradients_reach_generator_after_update():
    torch.manual_seed(17);model=Dynamics('anchored_stable');x=torch.randn(8,4,8);target=torch.randn(8,3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    loss=(model(x)[:,:,:2]-target).square().mean();loss.backward()
    assert model.A.grad.norm()==0 and model.encoder[0].weight.grad.norm()==0 and model.acceleration.weight.grad.norm()>0
    optimizer.step();optimizer.zero_grad();(model(x)[:,:,:2]-target).square().mean().backward()
    assert model.A.grad.norm()>0 and model.encoder[0].weight.grad.norm()>0


def test_probability_scores_reward_accurate_uncertainty():
    events=np.array([0.,1.]);good=probability_scores(np.array([.1,.9]),events);baseline=probability_scores(np.array([.5,.5]),events)
    assert good[0].mean()<baseline[0].mean() and good[1].mean()<baseline[1].mean()
