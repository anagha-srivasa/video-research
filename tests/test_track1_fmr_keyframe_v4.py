import cv2
import numpy as np
from track1_fmr.keyframe_identity import KeyframeDetector,KeyframeObservationIdentityTracker as Tracker
from track1_fmr.observation_identity import Observation,bbox_cloud


def observation(box=(.1,.1,.9,.9)):
    return Observation(np.array(box),bbox_cloud(box,np.zeros(6)),.9,'detector_bbox')


def image():return np.random.default_rng(4).integers(0,255,(256,256,3),dtype=np.uint8)


def test_v4_native_capped_detector_size():
    assert KeyframeDetector.recommended_detector_imgsz(256,256)==256
    assert KeyframeDetector.recommended_detector_imgsz(1080,1920)==480
    assert KeyframeDetector.recommended_detector_imgsz(16,16)==32


def test_v4_empty_does_not_burst_detector():
    m=Tracker(10,revision='v4');calls=[]
    def detector(_):calls.append(1);return []
    for k in range(23):m.step(image(),k/30,detector)
    assert len(calls)==3


def test_v4_one_forward_backward_batch_includes_query_and_anchors(monkeypatch):
    m=Tracker(10,revision='v4');rgb=image();m.step(rgb,0,lambda _:[observation()]);m.bind_query_point(0,0,np.array([.5,.5]))
    original=cv2.calcOpticalFlowPyrLK;counts=[]
    def counted(*args,**kwargs):counts.append(len(args[2]));assert kwargs['winSize']==(13,13);return original(*args,**kwargs)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',counted)
    moved=cv2.warpAffine(rgb,np.array([[1,0,1],[0,1,0]],np.float32),(256,256));m.step(moved,1/30,lambda _:[])
    assert len(counts)==2 and counts[0]==counts[1]==9
    p=m.query_particles[0];assert p.observed and np.linalg.norm(p.position-[.5+1/256,.5])*256<.5
    assert m.gray.shape==(256,256)


def test_v4_failed_particle_is_not_retried_from_prediction(monkeypatch):
    m=Tracker(10,revision='v4');rgb=image();m.step(rgb,0,lambda _:[observation()]);m.bind_query_point(0,0,np.array([.5,.5]))
    original=cv2.calcOpticalFlowPyrLK
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lambda *args,**kwargs:(None,None,None))
    m.step(rgb,1/30,lambda _:[]);assert not m.query_particles[0].observed and not m.query_particles[0].seed_valid
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',original)
    for k in range(2,10):m.step(rgb,k/30,lambda _:[])
    assert not m.query_particles[0].observed
    m.step(rgb,10/30,lambda _:[observation()]);assert m.query_particles[0].source=='detector_reseed' and not m.query_particles[0].observed
    m.step(rgb,11/30,lambda _:[]);assert m.query_particles[0].observed


def test_v4_small_carrier_cannot_match_giant_even_via_rescue():
    m=Tracker(1,revision='v4');small=observation((.48,.48,.52,.52));giant=observation((0,0,1,1));rgb=image()
    assert m.area_hijack(small,giant)
    m.step(rgb,0,lambda _:[small]);m.step(rgb,1/30,lambda _:[giant])
    assert m.entities[0].observation is small and m.entities[0].evidence!='detector'
    assert m.entities[1].observation is giant


def test_v4_subpart_updates_physical_shape_separately():
    m=Tracker(1,revision='v4');parent=observation();child=observation((.4,.4,.6,.6));rgb=image()
    m.step(rgb,0,lambda _:[parent]);m.step(rgb,1/30,lambda _:[child])
    assert len(m.last_adoptions)==1
    expected=.25*parent.cloud.shape_scatter+.75*child.cloud.shape_scatter
    assert np.allclose(m.entities[0].belief.shape_scatter,expected)
    assert m.entities[0].measurement is not None and np.allclose(m.entities[0].measurement,child.cloud.centroid)


def test_v4_replay_diffusion_matches_independent_scalar():
    from track1_fmr.calibration_replay import replay_v4,replay_v4_scalar
    values=[]
    for frame in range(8):
        row=np.zeros(32);row[:4]=[frame/30,0,.3+frame*.001,.4];row[4:8]=[1e-5,0,0,1e-5];row[28]=1 if frame%4==0 else 2;row[29:31]=[.001,0];row[31]=1/30;values.append(row)
    archives=[('ep',np.asarray(values))];a=replay_v4(archives,.1,.01,True);b=replay_v4_scalar(archives,.1,.01,True)
    assert a[1]==b[1]==1 and np.isclose(a[0],b[0])
    for key in a[2]:
        assert np.allclose(a[2][key][0],b[2][key][0]) and np.allclose(a[2][key][1],b[2][key][1])


def test_query_particles_do_not_change_carrier_state():
    rgb=image();plain=Tracker(10,revision='v4');hybrid=Tracker(10,revision='v4')
    for model in (plain,hybrid):model.step(rgb,0,lambda _:[observation()])
    hybrid.bind_query_point(0,0,np.array([.5,.5]))
    for k in range(1,5):
        frame=cv2.warpAffine(rgb,np.array([[1,0,k],[0,1,0]],np.float32),(256,256))
        plain.step(frame,k/30,lambda _:[]);hybrid.step(frame,k/30,lambda _:[])
        assert set(plain.entities)==set(hybrid.entities)
        for tid in plain.entities:
            assert np.allclose(plain.entities[tid].belief.mean,hybrid.entities[tid].belief.mean)
            assert np.allclose(plain.entities[tid].belief.covariance,hybrid.entities[tid].belief.covariance)
