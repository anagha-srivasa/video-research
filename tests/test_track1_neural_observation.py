import io
import pickle
import numpy as np
import pytest
from track1_neural_observation.benchmark import ArrayUnpickler, official_metrics, lk_tracks


def test_numeric_dataset_unpickler_rejects_executable_globals():
    import os
    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(os.system))).load()
    array = np.arange(12).reshape(6, 2)
    result = ArrayUnpickler(io.BytesIO(pickle.dumps({'points': array}, protocol=4))).load()
    np.testing.assert_array_equal(result['points'], array)


def test_official_metric_perfect_and_wrong_visibility():
    metric = official_metrics()
    points = np.array([[[[5., 10.], [6., 10.], [7., 10.]]]])
    hidden = np.zeros((1, 1, 3), bool)
    queries = np.array([[[0., 10., 5.]]])
    good = metric(queries, hidden, points, hidden, points, 'first')
    assert good['average_jaccard'][0] == 1.
    bad = metric(queries, hidden, points, ~hidden, points, 'first')
    assert bad['average_jaccard'][0] == 0.
    assert bad['average_pts_within_thresh'][0] == 1.


def test_lk_future_query_is_not_activated_early():
    rng = np.random.default_rng(12)
    rgb = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    video = np.repeat(rgb[None], 5, axis=0)
    q = np.array([[0., 30., 30.], [3., 40., 40.]])
    positions, visible = lk_tracks(video, q)
    assert visible[:, 0].all()
    assert not visible[:3, 1].any()
    assert visible[3:, 1].all()
    np.testing.assert_allclose(positions[:, 0], np.tile([30., 30.], (5, 1)), atol=1e-3)


def test_adapter_forbids_silent_reseeding_and_out_of_order_frames(monkeypatch):
    import track1_neural_observation.tracker as module
    monkeypatch.setattr(module, 'resources', lambda: {})
    class FakeModel:
        def track_frame(self, frame, queries, state, autocast):
            return np.array([[3., 4.]], np.float32), np.array([True]), object()
    tracker = module.Tracker.__new__(module.Tracker)
    tracker.model, tracker.device = FakeModel(), 'cpu'
    tracker.reset()
    rgb = np.zeros((10, 10, 3), np.uint8)
    tracker.step(rgb, 100, np.array([[3., 4.]]))
    with pytest.raises(ValueError):
        tracker.step(rgb, 100)
    with pytest.raises(ValueError):
        tracker.step(rgb, 200, np.array([[3., 4.]]))
