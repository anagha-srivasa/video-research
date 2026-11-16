import cv2
import numpy as np
from track1_fmr.belief_filter import Belief, BeliefTracker, advance
from track1_fmr.dorsal_flux import measure, build_local_super_gaussian_prior
from track1_fmr.lebesgue_tubes import FunctionalTubeBuilder
from track1_fmr.pipeline import Pipeline


def cloud_at(left=10):
    rgb = np.zeros((100,100,3),np.uint8)
    mask = np.zeros((100,100),bool); mask[10:20,left:left+10] = True
    return measure(rgb,mask)


def test_inverse_cache_reuses_same_shape_and_invalidates_changed_shape(monkeypatch):
    tracker = BeliefTracker()
    tracker.tracks[0] = Belief(0.,np.zeros(4),np.eye(4),np.eye(2)*.001,'OBSERVED')
    original = np.linalg.inv; calls = []
    def inverse(matrix):
        calls.append(matrix.copy())
        return original(matrix)
    monkeypatch.setattr(np.linalg,'inv',inverse)
    cached = tracker.inverse_shape_scatter(0)
    tracker.tracks[0] = advance(tracker.tracks[0],.1,.001)
    assert tracker.inverse_shape_scatter(0) is cached
    ys,xs,patch = build_local_super_gaussian_prior((.5,.5),tracker.tracks[0].shape_scatter,160,160,inv_shape_scatter=cached)
    assert len(calls) == 1 and not cached.flags.writeable
    _,_,uncached = build_local_super_gaussian_prior((.5,.5),tracker.tracks[0].shape_scatter,160,160)
    np.testing.assert_array_equal(patch,uncached)
    old = tracker.tracks[0]
    tracker.tracks[0] = Belief(old.time,old.mean,old.covariance,old.shape_scatter*2,old.status)
    assert tracker.inverse_shape_scatter(0) is not cached
    assert len(calls) == 3


def test_single_hit_three_miss_guard_frees_slot_and_retains_multi_hit_tracks():
    tracker = BeliefTracker(max_tracks=1)
    a,b = cloud_at(10),cloud_at(80)
    tracker.update(0.,[a]); tracker.update(.1,[]); tracker.update(.2,[])
    result = tracker.update(.3,[b])
    assert result['evicted_working_set'] == [0] and result['rejected_capacity'] == 0
    assert set(tracker.tracks) == {1} and 0 not in tracker._inverse_shapes
    assert tracker.observed_count == {1:1} and tracker.misses == {1:0}
    tracker.update(.4,[b]); tracker.update(.45,[b])
    for t in [.5,.6,.7,.8]:
        tracker.update(t,[])
    assert 1 in tracker.tracks and tracker.observed_count[1] == 3
    assert tracker.misses[1] == 4


def test_crop_quality_skip_and_new_epoch_equal_quality_cache(monkeypatch):
    a = cloud_at()
    state = Belief(0.,np.r_[a.centroid,0.,0.],np.eye(4)*.001,a.shape_scatter,'OBSERVED')
    builder = FunctionalTubeBuilder(0,maximum_duration=.2)
    patch = np.full((96,96,3),10,np.uint8)
    assert builder.wants_patch(state,0.,0.)
    builder.add(state,a,0.,patch,0.)
    state = advance(state,.1,.001,a.centroid,a.obs_cov_c)
    assert not builder.wants_patch(state,0.,0.)
    builder.add(state,a,0.)
    state = advance(state,.2,.001,a.centroid,a.obs_cov_c)
    assert builder.wants_patch(state,0.,0.)
    segment = builder.add(state,a,0.,patch+10,0.)
    assert segment.best_view_time == 0. and np.all(segment.best_view_patch == 10)
    assert builder.best[0] == .2 and np.all(builder.best[1] == 20)
    pipeline = Pipeline(heartbeat_seconds=.01,heartbeat=lambda rgb:[dict(measure=a,box=np.array([.1,.1,.2,.2]),saliency=1.)])
    original = cv2.resize; crops = []
    def resize(image,size,*args,**kwargs):
        if size == (96,96):crops.append(image.shape)
        return original(image,size,*args,**kwargs)
    monkeypatch.setattr(cv2,'resize',resize)
    rgb = np.zeros((100,100,3),np.uint8)
    pipeline.step(rgb,0.); pipeline.step(rgb,.1)
    assert len(crops) == 1


def test_eviction_cleans_live_pipeline_builders_and_evidence():
    a = cloud_at()
    pipeline = Pipeline(heartbeat=lambda rgb:[dict(measure=a,box=np.array([.1,.1,.2,.2]),saliency=1.)])
    rgb = np.zeros((100,100,3),np.uint8)
    for time in [0.,.1,.2,.3]:
        result = pipeline.step(rgb,time)
    assert result['evicted_working_set'] == 1
    assert not pipeline.tracker.tracks and not pipeline.builders and not pipeline.evidence
    assert not pipeline.last_observed_ts and not pipeline.tracker._inverse_shapes
    assert pipeline.tracker.max_tracks == 40


def test_prediction_reuse_is_tied_to_source_time_and_process_noise(monkeypatch):
    import track1_fmr.belief_filter as module
    tracker=BeliefTracker()
    tracker.update(0.,[cloud_at()])
    original=module.advance; calls=[]
    def counted(*args,**kwargs):
        calls.append((args[0],args[1],args[2]))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'advance',counted)
    first=tracker.predict_at(.1,[0])[0]
    assert tracker.predict_at(.1)[0] is first and len(calls)==1
    result=tracker.update(.1,[])
    assert result['predictions'][0] is first and len(calls)==1
    assert not tracker._prediction_cache
    tracker.predict_at(.2)
    assert len(calls)==2
    tracker.noise*=2
    changed=tracker.predict_at(.2)[0]
    assert len(calls)==3
    old=tracker.tracks[0]
    tracker.tracks[0]=Belief(old.time,old.mean,old.covariance*2,old.shape_scatter,old.status)
    replaced=tracker.predict_at(.2)[0]
    assert len(calls)==4
    assert not np.array_equal(changed.covariance,replaced.covariance)
