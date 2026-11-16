"""Development writer for complete variable-epoch tube records.

This helper is intentionally not injected into the running locked audit.
"""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_tubes(path, tubes, *, video_id, episode_id):
    rows = []
    for segment in tubes:
        patch = segment.best_view_patch
        rows.append(dict(video_id=video_id, episode_id=episode_id, track_id=int(segment.track_id),
                         t_start=float(segment.t_start), t_end=float(segment.t_end),
                         legendre_c=segment.legendre_c.ravel().tolist(),
                         legendre_h=segment.legendre_h.ravel().tolist(),
                         shape_scatter=segment.shape_scatter.ravel().tolist(),
                         belief_cov_end=segment.belief_cov_end.ravel().tolist(),
                         algebraic_sig=segment.algebraic_sig.tolist(),
                         best_view_patch=None if patch is None else patch.tobytes(),
                         best_view_patch_shape=None if patch is None else list(patch.shape),
                         best_view_time=segment.best_view_time, epistemic_tag=segment.epistemic_tag.value,
                         identity_verified=False, physical_calibration_verified=False))
    schema = pa.schema([('video_id', pa.string()), ('episode_id', pa.string()), ('track_id', pa.int64()),
                        ('t_start', pa.float64()), ('t_end', pa.float64()),
                        ('legendre_c', pa.list_(pa.float64(), 8)), ('legendre_h', pa.list_(pa.float64(), 18)),
                        ('shape_scatter', pa.list_(pa.float64(), 4)), ('belief_cov_end', pa.list_(pa.float64(), 16)),
                        ('algebraic_sig', pa.list_(pa.float32(), 24)), ('best_view_patch', pa.binary()),
                        ('best_view_patch_shape', pa.list_(pa.int32())), ('best_view_time', pa.float64()),
                        ('epistemic_tag', pa.string()), ('identity_verified', pa.bool_()),
                        ('physical_calibration_verified', pa.bool_())])
    path = Path(path)
    if not path.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('Tube ledger must stay inside the project root')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression=None)
    tmp.replace(path)
    return len(rows)
