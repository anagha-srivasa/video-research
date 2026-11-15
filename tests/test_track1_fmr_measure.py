import numpy as np
from track1_fmr.dorsal_flux import measure
from track1_fmr.belief_filter import directional_noise


def test_region_measure_and_occlusion_inflation():
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[5:15, 10:20] = [220, 20, 10]
    mask = np.zeros((20, 30), bool); mask[5:15, 10:20] = True
    cloud = measure(rgb, mask)
    np.testing.assert_allclose(cloud.centroid, [14.5 / 30, 9.5 / 20])
    assert cloud.algebraic_sig.shape == (24,) and cloud.fourier_support.shape == (9,)
    covariance = directional_noise(np.eye(2), [0, np.pi / 2], [0, 1], 2)
    assert covariance[0, 0] > covariance[1, 1]
    assert not cloud.centroid.flags.writeable


def test_functional_builder_records_short_unsupported_epochs():
    from track1_fmr.belief_filter import Belief, advance
    from track1_fmr.lebesgue_tubes import FunctionalTubeBuilder
    rgb = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), bool); mask[5:15, 10:20] = True
    cloud = measure(rgb, mask)
    state = Belief(0., np.array([.5, .5, .1, 0.]), np.eye(4), np.eye(2), 'OBSERVED')
    builder = FunctionalTubeBuilder(1)
    builder.add(state, cloud, 0.)
    state = advance(state, .1, .1)
    assert builder.add(state, None, 0.) is None
    assert builder.unsupported_short_epochs == 1
    state = advance(state, .2, .1)
    builder.add(state, None, 0.)
    assert builder.close() is None
    assert builder.unsupported_geometry_epochs == 1


def test_tracker_coasts_and_recovers_without_changing_shape():
    from track1_fmr.belief_filter import BeliefTracker
    rgb = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), bool); mask[5:15, 10:20] = True
    cloud = measure(rgb, mask)
    tracker = BeliefTracker()
    first = tracker.update(0., [cloud])['tracks'][0]
    hidden = tracker.update(.1, [])['tracks'][0]
    recovered = tracker.update(.2, [cloud])
    assert hidden.status == 'PREDICTED' and recovered['tracks'][0].status == 'OBSERVED'
    np.testing.assert_array_equal(hidden.shape_scatter, first.shape_scatter)
    assert recovered['matched'] == 1 and not recovered['identity_verified']


def test_blind_pipeline_has_no_fabricated_contact():
    from track1_fmr.pipeline import Pipeline
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[25:40, 25:40] = 255
    pipeline = Pipeline(heartbeat=lambda rgb: [])
    first = pipeline.step(frame, 0.)
    second = pipeline.step(frame, .1)
    assert not first['identity_verified'] and not second['identity_verified']
    pipeline.close()
    assert all(x.r_phys_prob is None for x in pipeline.junctions())


def test_tube_retains_only_highest_quality_crop():
    from track1_fmr.belief_filter import Belief, advance
    from track1_fmr.lebesgue_tubes import FunctionalTubeBuilder
    rgb = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), bool); mask[5:15, 10:20] = True
    cloud = measure(rgb, mask)
    state = Belief(0., np.array([.5, .5, .1, 0.]), np.eye(4), np.eye(2), 'OBSERVED')
    builder = FunctionalTubeBuilder(1)
    builder.add(state, cloud, 0., np.full((5, 5, 3), 10, np.uint8), .2)
    state = advance(state, .1, .1, cloud.centroid, cloud.obs_cov_c)
    builder.add(state, cloud, 0., np.full((5, 5, 3), 20, np.uint8), .8)
    assert all(len(sample) == 2 for sample in builder.data)
    tube = builder.close()
    assert tube.best_view_time == .1 and (tube.best_view_patch == 20).all()


def test_opt_in_prior_is_tagged_predicted():
    from track1_fmr.belief_filter import Belief, advance
    from track1_fmr.lebesgue_tubes import FunctionalTubeBuilder
    rgb = np.zeros((20, 30, 3), np.uint8)
    mask = np.zeros((20, 30), bool); mask[5:15, 10:20] = True
    cloud = measure(rgb, mask)
    state = Belief(0., np.array([.5, .5, .1, 0.]), np.eye(4), cloud.shape_scatter, 'OBSERVED')
    builder = FunctionalTubeBuilder(1, carry_prior=True)
    builder.add(state, cloud, 0.)
    state = advance(state, .1, .1); builder.add(state, None, 0.)
    state = advance(state, .2, .1); builder.add(state, None, 0.)
    tube = builder.close()
    assert tube.epistemic_tag.value == 'PREDICTED' and tube.best_view_patch is None
    np.testing.assert_allclose(tube.legendre_h[0], cloud.fourier_support)


def test_dorsal_grid_preserves_native_crop_coordinates(monkeypatch):
    import track1_fmr.pipeline as module
    frame = np.zeros((320, 640, 3), np.uint8)
    frame[80:240, 160:480] = [220, 20, 10]
    calls = []

    def heartbeat(image):
        calls.append(image.shape)
        mask = np.zeros(image.shape[:2], bool)
        h, w = mask.shape
        mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        return [dict(measure=measure(image, mask), saliency=1., box=np.array([.25, .25, .75, .75]))]

    def flux(previous, current, excluded):
        assert previous.shape == current.shape == (160, 160, 3)
        assert excluded.shape == (160, 160)
        return dict(supported=True, clouds=[])

    monkeypatch.setattr(module, 'flux', flux)
    pipeline = module.Pipeline(dorsal_grid=160, heartbeat=heartbeat)
    pipeline.step(frame, 0.)
    state = pipeline.tracker.tracks[0]
    np.testing.assert_allclose(state.mean[:2], [319.5 / 640, 159.5 / 320])
    # Inspect the retained crop: it must come from the native RGB frame.
    builder = pipeline.builders[0]
    centre = state.mean[:2] * [640, 320]
    extent = np.maximum(4, 3 * np.sqrt(np.diag(state.shape_scatter)) * [640, 320])
    left, top = np.maximum(0, (centre - extent).astype(int))
    right, bottom = np.minimum([640, 320], np.ceil(centre + extent).astype(int))
    import cv2
    np.testing.assert_array_equal(builder.best[1], cv2.resize(frame[top:bottom, left:right], (96, 96)))
    pipeline.step(frame, .1)
    assert pipeline.previous.shape == (160, 160, 3)
    assert calls == [frame.shape]
    native = module.Pipeline(heartbeat=heartbeat)
    native.step(frame, 0.)
    assert calls[-1] == frame.shape
    import pytest
    with pytest.raises(ValueError):
        module.Pipeline(dorsal_grid=0)
