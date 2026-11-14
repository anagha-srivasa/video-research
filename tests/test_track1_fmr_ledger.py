import numpy as np
import pyarrow.parquet as pq
from track1_fmr.contracts import SpacetimeTubeSegment, EpistemicTag
from track1_fmr.ledger import write_tubes


def test_nullable_patch_roundtrip(tmp_path):
    # Test output remains inside the project root even when pytest is run elsewhere.
    from pathlib import Path
    path = Path('work/track1_fmr/ledger_null_test.parquet')
    segment = SpacetimeTubeSegment(1, 0, 1, np.zeros((4, 2)), np.zeros((2, 9)),
                                  np.eye(2), np.eye(4), np.zeros(24), None, None, EpistemicTag.OBSERVED)
    write_tubes(path, [segment], video_id='v', episode_id='e')
    row = pq.read_table(path).to_pylist()[0]
    assert row['best_view_patch'] is None and row['best_view_patch_shape'] is None
    assert not row['identity_verified']
