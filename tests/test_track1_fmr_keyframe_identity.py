import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest
from track1_fmr.keyframe_identity import KeyframeObservationIdentityTracker as Tracker
from track1_fmr.observation_identity import Observation,bbox_cloud


def observation(box=(.1,.1,.9,.9)):
    return Observation(np.array(box),bbox_cloud(box,np.zeros(6)),.9,'detector_bbox')


def texture():
    return np.random.default_rng(4).integers(0,255,(320,320,3),dtype=np.uint8)


def test_fb_measurement_is_real_and_stride_is_six():
    image=texture();calls=[]
    def detector(rgb):calls.append(1);return [observation((0,0,1,1))]
    model=Tracker();model.step(image,0,detector)
    for frame in range(1,7):
        shifted=cv2.warpAffine(image,np.array([[1,0,frame],[0,1,0]],np.float32),(320,320))
        states=model.step(shifted,frame/30,detector);entity=states[0]
        if frame<6:
            assert entity.evidence=='fb_lk' and len(entity.fb_errors)>=4
            assert entity.measurement[0]>.5
            assert entity.belief.mean[0]>.5
            assert entity.fb_errors.max()<1.5
            assert np.linalg.eigvalsh(entity.belief.covariance).min()>0
    assert len(calls)==2


def test_failed_flow_cannot_observe_or_revive_predicted_anchors(monkeypatch):
    image=texture();model=Tracker();model.step(image,0,lambda _: [observation()])
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lambda *a,**k:(None,None,None))
    states=model.step(image,1/30,lambda _:[])
    assert states[0].evidence=='predicted' and not states[0].observed
    assert len(states[0].anchors)==0 and 'flow_loss' in model.last_reasons
    model.step(image,2/30,lambda _:[])
    assert not model.entities[0].observed


def test_mesh_local_deformation_and_refresh_continuity():
    old=np.array([[0,0],[0,.1],[.1,0],[.1,.1],[.8,.8],[.8,.9],[.9,.8],[.9,.9]])
    new=old.copy();new[:4]+=[.1,0];new[4:]+=[-.1,0]
    point=np.array([.05,.05]);moved=Tracker.transport(point,old,new)
    assert moved[0]>.14  # Centroid displacement is zero; local mesh follows the near lobe.
    reseeded=old+.2
    assert np.allclose(Tracker.transport(moved,reseeded,reseeded),moved)
    assert np.allclose(Tracker.transport(point,old[:3],new[:3]),point)


def test_time_order_and_empty_trigger():
    model=Tracker();rgb=texture();model.step(rgb,0,lambda _:[])
    model.step(rgb,.03,lambda _:[])
    assert 'empty' in model.last_reasons
    with pytest.raises(ValueError):model.step(rgb,.03,lambda _:[])


def test_asymmetric_binding_rejects_giant_support():
    path=Path('scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py')
    spec=importlib.util.spec_from_file_location('calibration',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    states={('ep',1,0):(np.array([.5,.5,0,0]),np.eye(4)),('ep',1,1):(np.array([.51,.5,0,0]),np.eye(4))}
    supports={('ep',1,0):(np.eye(2)*.1,np.array([0,0,1,1]),3),('ep',1,1):(np.eye(2)*.0001,np.array([.49,.49,.53,.53]),3)}
    assert module.bind_keyframe_reference(states,supports,'ep',1,[.5,.5],[.48,.48,.54,.54],1)==1
    assert module.bind_keyframe_reference(states,supports,'ep',1,[.5,.5],[.48,.48,.54,.54],6) is None


def test_split_lineage_follows_model_point_without_reference_input():
    image=texture();model=Tracker(keyframe_stride=1)
    model.step(image,0,lambda _:[observation()]);model.entities[0].hits=3
    child=observation((.45,.45,.55,.55))
    states=model.step(image,1/30,lambda _:[observation(),child])
    assert model.lineage==[dict(time=1/30,parent=0,child=1,kind='split_hypothesis')]
    assert model.follow(0,np.array([.5,.5]))==1
    assert model.follow(0,np.array([.2,.2]))==0
    assert states[1].parent_ids==(0,)


def test_flow_spike_does_not_veto_real_detector(monkeypatch):
    image=texture();model=Tracker(keyframe_stride=1);model.step(image,0,lambda _:[observation()])
    monkeypatch.setattr(model,'_flow',lambda gray:{0:(np.array([.8,.8]),np.eye(2)*.01)})
    states=model.step(image,1/30,lambda _:[observation()])
    assert states[0].evidence=='detector' and np.allclose(states[0].belief.mean[:2],[.5,.5])
    assert len(states)==1


def test_dormant_recovery_and_ttl():
    image=texture();model=Tracker(keyframe_stride=1);model.step(image,0,lambda _:[observation()])
    model.step(image,1.1,lambda _:[])
    assert 0 in model.dormant and 0 not in model.entities
    model.step(image,1.2,lambda _:[observation()])
    assert model.entities[0].observed and not model.dormant
    model.step(image,12,lambda _:[])
    assert not model.entities and not model.dormant


def test_batched_calibration_matches_scalar_fixed_associations():
    path=Path('scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py')
    spec=importlib.util.spec_from_file_location('calibration_batch',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(11);values=[]
    for frame in range(10):
        for tid in range(3):
            if frame%3==tid:continue
            z=np.array([.2+tid*.1+frame*.003,.3])+rng.normal(0,.001,2)
            row=np.zeros(28);row[:4]=[frame/30,tid,*z];row[4:8]=[1e-5,1e-6,1e-6,2e-5];row[10:12]=[.1,0];values.append(row)
    archives=[('ep',np.array(values))]
    scalar=module.replay(archives,.1,.01);nis=module.replay.last_nis
    batched=module.replay_batched(archives,.1,.01)
    assert batched[1]==scalar[1] and np.isclose(batched[0],scalar[0],rtol=1e-12)
    assert np.isclose(module.replay_batched.last_nis,nis,rtol=1e-12)


def test_current_support_binding_uses_measured_warp_not_old_mask_or_filter():
    path=Path('scripts/exp_track1_fmr/keyframe_benchmark.py');spec=importlib.util.spec_from_file_location('keyframe_benchmark_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from track1_fmr.keyframe_identity import Entity
    from track1_fmr.belief_filter import Belief
    mask=np.zeros((160,160),bool);mask[32:48,32:48]=True
    obs=Observation(np.array([.2,.2,.3,.3]),bbox_cloud([.2,.2,.3,.3],np.zeros(6)),.9,'detector_mask',mask)
    e=Entity(0,Belief(0,np.array([.8,.8,0,0]),np.eye(4),np.eye(2)*.01,'OBSERVED'),obs.box.copy(),obs)
    assert np.array_equal(module.support(e),mask)  # Filter mean never moves a fresh detector mask.
    e.evidence='fb_lk';e.box=np.array([.4,.2,.5,.3])
    assert module.bind_current({0:e},np.array([.45,.25]))==0
    assert module.bind_current({0:e},np.array([.25,.25]))==-1
