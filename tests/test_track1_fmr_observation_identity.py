import numpy as np
import pytest
from dataclasses import replace
from track1_fmr.observation_identity import bbox_cloud,Observation,ObservationIdentityCarrier


def observation(box=(.25,.3,.4,.5),color=(130,160,110,10,10,10)):
    box=np.array(box,float);cloud=bbox_cloud(box,color)
    return Observation(box,cloud,.9,'detector_bbox')


def step(model,time,items,rgb=None):
    rgb=np.zeros((128,128,3),np.uint8) if rgb is None else rgb
    return model.step(rgb,time,[(o,o) for o in items])


def test_bbox_carrier_is_available_without_masks_or_valid_contours():
    o=observation();model=ObservationIdentityCarrier('motion','bbox')
    states=step(model,0.,[o]);carrier=states[0]
    assert carrier.observed and carrier.mode=='OBSERVED' and o.mask is None
    np.testing.assert_allclose(o.cloud.centroid,[.325,.4])
    np.testing.assert_allclose(np.diag(o.cloud.shape_scatter),np.array([.15,.2])**2/12+1e-9)
    assert not np.array_equal(carrier.belief.covariance[:2,:2],o.cloud.shape_scatter)
    assert set(step(model,.03,[o]))=={0}


def test_detector_only_has_no_persistent_identity_claim():
    model=ObservationIdentityCarrier('detector');o=observation()
    assert set(step(model,0.,[o]))=={0}
    assert set(step(model,.03,[o]))=={1}


def test_gap_deformation_coasting_and_redetection_keep_evidence_separate():
    model=ObservationIdentityCarrier('appearance');o=observation()
    for t in [0.,.03,.06,.09,.12]:assert set(step(model,t,[o]))=={0}
    changed=observation((.25,.28,.43,.54))
    states=step(model,.4,[changed])  # A >100 ms gap cannot permanently kill an entity.
    assert set(states)=={0} and states[0].observed
    np.testing.assert_array_equal(states[0].belief.shape_scatter,changed.cloud.shape_scatter)
    states=step(model,.6,[])
    assert not states[0].observed and states[0].mode=='COASTED'
    assert not step(model,1.6,[]) and model.last_telemetry['lost']==1
    states=step(model,1.7,[changed])
    assert set(states)=={0} and states[0].mode=='REDETECTED'


def test_optional_flow_is_not_observed_evidence():
    model=ObservationIdentityCarrier('flow');o=observation()
    rgb=np.random.default_rng(5).integers(0,256,(128,128,3),dtype=np.uint8)
    for t in [0.,.03,.06,.09,.12]:step(model,t,[o],rgb)
    states=step(model,.15,[],rgb)
    assert not states[0].observed and not model.last_observations
    assert model.last_telemetry['flow_predictions']>=1


def test_appearance_ablation_resolves_color_tie_only_when_enabled():
    red=observation(color=(100,200,140,10,10,10));green=observation(color=(160,40,180,10,10,10))
    for mode,expected in [('motion',green),('appearance',red)]:
        model=ObservationIdentityCarrier(mode);step(model,0.,[red,green]);states=step(model,.03,[green,red])
        np.testing.assert_array_equal(states[0].observation.cloud.algebraic_sig,expected.cloud.algebraic_sig)


def test_invalid_observation_and_timestamp_contracts():
    with pytest.raises(ValueError):observation((.5,.2,.4,.7))
    model=ObservationIdentityCarrier();step(model,0.,[observation()])
    with pytest.raises(ValueError):step(model,0.,[])


def test_detector_keeps_box_observations_when_model_returns_no_mask(tmp_path,monkeypatch):
    import track1_fmr.observation_identity as module
    from types import SimpleNamespace
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def cpu(self):return self
        def numpy(self):return self.value
    monkeypatch.setattr(module,'ROOT',tmp_path)
    checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'fake model fixture')
    detector=module.DetectorObserver(checkpoint)
    result=SimpleNamespace(boxes=SimpleNamespace(xyxyn=Tensor([[.2,.3,.5,.7]]),conf=Tensor([.8])),masks=None)
    detector.model=SimpleNamespace(predict=lambda **kwargs:[result])
    detections=detector(np.zeros((128,128,3),np.uint8))
    assert len(detections)==1 and detections[0][1].source=='detector_bbox'
    assert detections[0][1].mask is None and detections[0][1].cloud.mass>0


def test_common_denominator_counts_unmatched_and_never_credits_coasting():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/exp_track1_fmr/03_eval_observation_identity.py'
    spec=importlib.util.spec_from_file_location('observation_metrics',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    truth=np.full((1,4,2),.5);hidden=np.zeros((1,4),bool);first=np.array([0])
    row=module.point_row('test',truth,np.array([[True,False,True,False]]),np.ones((1,4),bool),truth,hidden,first,np.array([0]))
    assert row['visible_point_times']==3 and row['observed_correct_8px_times']==1
    assert row['carrier_coverage']==1 and row['observed_visible_coverage']==pytest.approx(1/3)
    unmatched=module.point_row('test',truth,np.zeros((1,4),bool),np.zeros((1,4),bool),truth,hidden,first,np.array([-1]))
    assert unmatched['visible_point_times']==3 and unmatched['observed_correct_8px_times']==0
