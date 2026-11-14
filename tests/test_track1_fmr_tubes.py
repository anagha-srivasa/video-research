import numpy as np
from track1_fmr.lebesgue_tubes import TubeBuilder


def test_knots_close_before_trigger_and_never_use_future_sample():
    tube = TubeBuilder(threshold=5)
    for i in range(5):
        assert tube.add(i * .1, [i * .1, 0], 0., 'OBSERVED') is None
    segment = tube.add(.5, [10., 0], 20., 'OBSERVED')
    assert segment.end == .4 and segment.reason == 'innovation'
    np.testing.assert_allclose(segment.position(.4), [.4, 0], atol=1e-10)
    assert tube.samples[0][0] == .5
