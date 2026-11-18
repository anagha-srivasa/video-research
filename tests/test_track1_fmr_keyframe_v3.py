import cv2
import numpy as np
from track1_fmr.keyframe_identity import KeyframeObservationIdentityTracker as Tracker,KeyframeDetector,PROMPTS_V3
from track1_fmr.observation_identity import Observation,bbox_cloud
from track1_fmr.belief_filter import advance


def obs():
    mask=np.zeros((160,160),bool);mask[32:128,32:128]=True
    return Observation(np.array([.2,.2,.8,.8]),bbox_cloud([.2,.2,.8,.8],np.zeros(6)),.9,'detector_mask',mask)


def image(seed=4):return np.random.default_rng(seed).integers(0,255,(320,320,3),dtype=np.uint8)


def test_v3_eroded_anchors_and_ncc_covariance_propagation(monkeypatch):
    rgb=image();model=Tracker(8,revision='v3');calls=[]
    def detector(_):calls.append(1);return [obs()]
    model.step(rgb,0,detector);e=model.entities[0]
    assert 4<=len(e.anchors)<=12
    assert np.all((e.anchors>.2+1/160)&(e.anchors<.8-1/160))
    prior=e.belief;expected=advance(prior,1/30,model.q)
    def reject_condition(*args,**kwargs):raise AssertionError('LK must not condition covariance')
    monkeypatch.setattr('track1_fmr.keyframe_identity.condition',reject_condition)
    e=model.step(rgb,1/30,detector)[0]
    assert e.evidence=='fb_lk' and e.ncc_scores.min()>=.65 and e.fb_errors.max()<1.2
    assert np.linalg.eigvalsh(e.belief.covariance-expected.covariance).min()>-1e-10
    assert np.trace(e.belief.covariance[:2,:2])>=np.trace(expected.covariance[:2,:2])
    assert len(calls)==1


def test_ncc_rejects_geometric_cycle_on_occluder(monkeypatch):
    model=Tracker(8,revision='v3');model.step(image(),0,lambda _:[obs()])
    def identity_lk(prev,cur,p,*args,**kwargs):return p.copy(),np.ones((len(p),1),np.uint8),np.zeros((len(p),1))
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',identity_lk)
    e=model.step(image(90),1/30,lambda _:[])[0]
    assert e.evidence=='predicted' and not e.observed and len(e.anchors)==0
    assert not model.last_reasons  # No flow-loss or flux-triggered detector burst.


def test_flat_patches_do_not_get_photometric_credit():
    assert np.all(Tracker.patch_ncc(np.zeros((4,5,5)),np.zeros((4,5,5)))==-1)


def test_single_detector_head_and_44_prompts():
    class Fake:
        calls=0
        def predict(self,*args,**kwargs):
            from types import SimpleNamespace
            self.calls+=1;assert kwargs['imgsz']==480
            return [SimpleNamespace(boxes=None)]
    d=KeyframeDetector(revision='v3');d.text_model=Fake()
    assert d.model is None and len(PROMPTS_V3)==44
    assert d(image())==[] and d.text_model.calls==1


def test_v3_stride_only_or_empty():
    model=Tracker(8,revision='v3');calls=[]
    def detector(_):calls.append(1);return [obs()]
    for k in range(17):model.step(image(),k/30,detector)
    assert len(calls)==3 and model.last_reasons==['stride']


def test_v3_batched_replay_matches_scalar_with_nonzero_lk_motion():
    from track1_fmr.calibration_replay import replay_v3,replay_v3_scalar
    rows=[]
    for frame in range(12):
        for tid in range(3):
            row=np.zeros(32);row[:4]=[frame/30,tid,.2+tid*.1+frame*.002,.3];row[4:8]=[1e-5,1e-6,1e-6,2e-5]
            row[28]=1 if frame%4==0 else 2;row[29:31]=[.002,0];row[31]=1/30
            rows.append(row)
    archives=[('ep',np.asarray(rows))]
    first=replay_v3(archives,.1,.012,True);nis=replay_v3.last_nis;second=replay_v3_scalar(archives,.1,.012,True)
    assert first[1]==second[1]==6 and np.isclose(first[0],second[0])
    assert np.isclose(nis,replay_v3_scalar.last_nis)
    for key in first[2]:
        for a,b in zip(first[2][key],second[2][key]):assert np.allclose(a,b)


def test_v3_gap_clears_correspondences_instead_of_reusing_old_motion():
    rgb=image();model=Tracker(8,revision='v3');model.step(rgb,0,lambda _:[obs()]);model.step(rgb,1/30,lambda _:[])
    assert len(model.entities[0].displacement_old)>=4
    e=model.step(rgb,.3,lambda _:[])[0]
    assert e.evidence=='predicted' and not len(e.displacement_old) and not len(e.anchor_patches)
