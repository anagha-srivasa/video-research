# Dataset v2 development rebuild

The current dataset product follows the requested 250 GB capacity limit using **uncompressed native RGB**. Selection, rather than image compression, reduces the original 2.43 TB decoded-frame estimate. Publisher MP4s/archives retain their original bytes. No native RGB frames are resized or recompressed in the partition.

- `dataset/partition_v2`: seven flat `.rgb.npy` shards, PTS/UID index, clip index and extraction receipts.
- `canonical/v2`: corrected full development label/context timelines, spatial metadata, bounded corrected geometry fits and appearance pilot features.
- `src/data_v2`: geometry adapter, streaming label builder, partition planner/extractor, reader, evaluation and checks.
- `reports/dataset_v2`: results, integrity/parity checks, visual review and dataset dashboard.
- `work/dataset_v2`: temporary scripts, execution logs and tests.

The partition contains **28,591 native 1920×1080 RGB frames**, occupying **177,858,893,696 bytes**, across **267 contiguous two-second clips**. Overlapping clip frames are stored once. Five training and two exploratory validation participants remain separate. Selection balances manual labels, holes, multiple components, small objects and contact relations. These strata are source annotations, not independent physical-motion or contact references. Exposed old test participants are excluded.

All 480,288 source observations from the seven development videos remain in the label store; paired RGB is available only where `frames.parquet` supplies an exact timestamp. The rebuild preserves 805 source masks with holes, fixes metadata for frozen spatial observations, and records 34,709 partial-geometry candidates separately from 445,514 complete observations. Partial geometry is never silently promoted to complete ground truth. Original raw polygons and source status remain available. Retrospective labels have unknown online availability, and the reader refuses online-label access.

Active lab geometry, appearance, contact and compound consumers now share the canonical `inner`→`hole` adapter, hierarchy validation and relevant hierarchy-aware rendering. Frozen v1/v2 code/artifacts remain historical baselines. Corrected geometry is evaluated on matched manual probes plus an explicit additional hole cohort. Improvements are modest: matched validation IoU 0.979344→0.979360 and BF2 0.884566→0.885150. Additional hole validation BF2 is 0.778795→0.786183. Source topology fixes are necessary but do not alone establish a large performance gain.

A corrected native-crop appearance pilot uses 448 manual observations, including 109 with holes. Frozen MobileNetV3-Small runs on Metal; object and context features stay separate, preserve crop aspect ratio, and each use a 128D PCA fitted on the 320 training observations. The source mask is a privileged Track 1 input. This pilot does not establish independent identity accuracy or replace full-timeline appearance tensors.

Readiness is task-specific. `canonical/v2/status.json` records native partition and retrospective geometry checks. A new full-timeline functional training release, covariance recalibration, independent motion/contact/identity references, and fresh locked evaluation remain pending. Track 1 is open and Track 2 remains on hold.

## Commands

Run from this root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m data_v2.run --stage all
```

Stages are `labels`, `partition`, `evaluate`, and `verify`. Extraction verifies source/index/code fingerprints and shard checksums before reusing a completed video. Differing fingerprints require a new output version rather than silent replacement. CPU math threads and decode concurrency are bounded; the writer closes/reopens its mmap every 128 selected frames to release resident mappings.

The separately reproducible context and appearance pilots use:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 \
  python3 work/dataset_v2/export_context.py
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 \
  python3 work/dataset_v2/appearance.py
PYTHONPATH=work/deps python3 work/dataset_v2/check_rgb.py
```

The software seek/decode parity check compares first/middle/last stored frames per video; all 21 sampled RGB arrays matched exactly. Full-file hashes cover every RGB shard. The alignment sheet is an AI inspection of seven manual examples and does not certify every annotation.

```python
from data_v2.store import Dataset
import pyarrow.parquet as pq

row = pq.read_table('dataset/partition_v2/frames.parquet').to_pylist()[0]
with Dataset() as ds:
    rgb = ds.get_frame(row['video_id'], row['timestamp_us'])
    assert rgb.shape == (1080, 1920, 3)
```

The reader caches at most two mapped video shards and returns a copy of the requested frame. Mask coordinates remain on their declared annotation grid; annotation-to-native transforms are explicit in the context tables. `get_mask(..., allow_partial=True)` opts into incomplete evidence; `online=True` raises. Missing labels are not negative observations, and visible-mask centroids are not physical material anchors.

The completed product measures **207.33 GB** including original sources, partition RGB and canonical/v2. Historical work caches and model releases are outside that product. All 267 selected intervals contain every corresponding native frame, 21 software-decode samples matched exactly, full RGB shard checksums passed, and 59 tests passed. All 741 canonical/v1 and Track 1 v1/v2 manifest entries checked unchanged. The initial memory guard stop is recorded; after periodic unmapping, sampled aggregate extraction/evaluation RSS peaked at 4.95 GiB.

For the supplemental review and consolidated manifest, run `visual_review.py`, `check_baselines.py`, and `finish.py` under `work/dataset_v2` after the preceding commands. These scripts add independent parity, baseline, coverage and capacity evidence to the report. A clean full rebuild of the new release has not yet been performed.
