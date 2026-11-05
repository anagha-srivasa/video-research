# Track 1 training ingestion package

`derived/track1_training_ready/status.json` is the readiness entry point. This is a source-backed development package for **retrospective Track 1 reconstruction**, with M16 functional history, separate causal source-centroid kinematics, masked appearance, and optional certificate-bound M32 geometry. Scientific acceptance remains false. Track 2 remains on hold.

## What was prepared

- A consistent SQLite snapshot preserves all 2,096 stored Qwen review rows and their raw response/failure provenance. The selected pass completed 834 new questions, preserved 206, and ended with 82 unresolved questions.
- `review_coverage.parquet` records all 58,800 case/frame opportunities; absent labels remain absent. `machine_observations.parquet` exposes 2,781 case/frame observations after explicit same-cue, same-frame sharing. These are not 2,781 independent VLM calls or verified instance matches. `frame_diagnostics.parquet` contains 6,480 native-frame diagnostic records.
- Only 44 projected-relation observations pass a conservative optional weak-label mask. **No** physical-contact, boundary, identity, event or physical-velocity ground truth is manufactured. Unknowns use label `-1` and false masks. Weak labels are not joined to provisional source tracks: a matching class cue does not establish instance identity. Retrospective neighbor images also make these unsuitable as causal input features.
- Existing fitted source-contour tensors are reused unchanged. Qwen supplied no accurate replacement contours, so it cannot justify changing those fits. Approximate attention boxes are not segmentation masks. Frozen v1/v2 artifacts remain unchanged.
- `states.parquet` retains supported and unsupported opportunities, deduplicating identical video/track/timestamp opportunities with a fixed context-first priority. Selection does not use validation quality. Indices filter numerical support; failures remain in the state ledger. Sequences never cross source, episode, source/correspondence epoch, or a missing quarter-second hop.

| Sequence length | Train | Validation |
|---|---:|---:|
| 1 | 3,377 | 737 |
| 4 | 1,370 | 273 |
| 8 | 762 | 153 |
| 16 | 266 | 46 |

All indexed sequences have appearance support; consumers still receive explicit masks. Of the length-16 sequences, 104 train / 5 validation have optional M32 geometry at every step. Those counts do not require one uninterrupted M32 epoch: optional geometry resets and history durations remain explicit. All counts overlap and do not represent independent episodes. Existing training-only PCA/fit selection remains locked; no new validation tuning was performed.

## Loading a training batch

Use the installed runtime `/opt/anaconda3/bin/python3.13` with `PYTHONPATH=src:work/deps:work/track1_deps`. The shell's default `python3` may lack project dependencies.

```python
import pyarrow.parquet as pq
from track1_retrain.prepare import TrainingDataset

index = pq.read_table(
    'derived/track1_training_ready/indices/train.length16.parquet'
).to_pylist()[0]
with TrainingDataset() as dataset:
    batch = dataset.get_sequence(index)
    assert batch['coefficients'].shape == (16, 4, 35)
    assert batch['xi'].shape == (16, 4, 68)
    assert batch['appearance'].shape == (16, 4, 2, 128)
    # geometry32 is a list of optional views, never a zero-filled missing shape.
    # Use appearance_mask and each kinematics support_valid flag in losses.
    # semantic_ground_truth is None: machine observations are a separate table.
```

Load RGB by an exact indexed timestamp from `frames.parquet`:

```python
frames = pq.read_table('derived/track1_training_ready/frames.parquet').to_pylist()
frame = frames[0]
with TrainingDataset() as dataset:
    rgb = dataset.get_frame(frame['video_id'], frame['timestamp_us'])
    assert rgb.shape == (1080, 1920, 3)
```

The original MP4 is decoded at its exact source PTS; no approximate frame is substituted. Many model endpoints lie between native frame timestamps, so do not assume an endpoint is an RGB key. Select an explicit indexed timestamp and retain its offset if the task needs RGB alignment. The reader never silently picks a nearby frame.

The native NPY partition is incomplete on disk. This package does not depend on it, restore it, duplicate RGB, transcode source media, or introduce new compression. Per-frame isolated decoding favors correctness and bounded memory over peak RGB throughput. Cached functional/appearance training avoids decoding altogether. For RGB-heavy training, benchmark a bounded per-video sequential decoder before increasing worker counts; do not launch dozens of independent seeks. Keep originals and referenced derived tensor folders in place: this is an indexed local package, not a standalone copied archive.

## Checks and reproduction

`reports/track1_training_ready/preprocessing.json` reports all denominators and sequence counts. `verification.json` records checks over all tensor files, PSD covariance checks, analytic collocation reconstruction, geometry certificate bindings, original MP4 hashes, native PTS inventory and sampled actual reader batches. Every indexed PTS is checked against the canonical inventory; actual RGB decoding is sampled, not a full new decode of every source frame. The full test suite passes 97 tests.

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m track1_retrain.prepare --verify
```

Verification checks release/dependency hashes before rechecking ingestion. `--resume` is for an interrupted initial build; it reuses the preserved SQLite snapshot and regenerates the preparation tables. Logs and locks are in `work/track1_retrain`. Existing memory guards require at least 5 GiB available and stop at 20 GiB summed process-tree RSS; the hardware is 48 GiB, regardless of larger disk-budget assumptions.

The package is ready for numerical Track 1 reconstruction ingestion. Independent motion/contact/identity references, calibrated physical uncertainty, independent Qwen accuracy validation, genuinely fresh scientific evaluation, and complete on-disk native RGB materialization are not certified by this readiness flag. No model was retrained or promoted in this preparation step.
