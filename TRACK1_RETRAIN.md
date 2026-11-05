# Track 1 retraining on dataset v2

This open development release is in `derived/track1_dataset_v2`. Numerical status: `derived/track1_dataset_v2/status.json`. Scientific acceptance remains false; Track 2 remains on hold.

79,270 selected source observations were refitted. There are 19,974 attempted model/window records, 3,485 numerically supported states and 1,631 states with represented source completeness plus a decoder-bound continuous geometry certificate. The last condition concerns retained source geometry, not independently verified physical topology or identity.

The fixed-16 comparison stores `(4,35)` coefficients and `(4,68)` GL4 xi. The compound-32 ablation stores `(4,67)` coefficients and `(4,132)` xi, with ring allocation/roles/parents. Both use one-second histories and 0.25-second hops. Coefficients and stored tensors are float32; fitting/covariance calculations use CPU float64. Appearance stores separate `(4,2,128)` object/context coefficients; PCA and temporal selection use training participants only. The frozen CNN uses Metal. No neural backbone fine-tuning is part of this architecture.

The raw coefficients retain their fitted history derivatives. The separate local-CV kinematic state is authoritative for the declared source-position motion estimator. It is not independently calibrated physical motion. Endpoint-neutral history detail does not change the coarse endpoint or its velocity. Geometry coefficients and certificates form a separate view; no calibrated geometry covariance is transferred after rollback.

```python
from track1_retrain.store import Dataset
import pyarrow.parquet as pq
rows = pq.read_table('derived/track1_dataset_v2/functional/P02_122.m16.parquet').to_pylist()
row = next(r for r in rows if r['valid'])
with Dataset() as ds:
    state = ds.get_state(row['video_id'], row['entity_track_id'], row['window_end_us'], model=16)
    assert state['coefficients'].shape == (4,35)
    assert state['xi'].shape == (4,68)
```

Use `model=32` explicitly for the compound ablation. `require_geometry=True` additionally requires represented source completeness and a bound certificate. `ds.geometry(state, timestamp_us)` uses the certified normalized decoder within the history interval. Reader memory maps at most two video/model shards. Invalid states are rejected by default.

Sequence indices are split by participant and available at lengths 4 and 16. They never bridge missing hops or source/correspondence epochs. Current counts: {"train/length16": 0, "train/length4": 396, "validation/length16": 0, "validation/length4": 63}. Do not assume a sixteen-state sequence exists in short clips.

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python3 -m track1_retrain.run --stage all
```

Stages: spatial, appearance, search, refine-selection, functional, evaluate, verify. CPU spatial chunks and complete appearance videos have hash-checked checkpoints. Temporary scripts/logs are in `work/track1_retrain`; uncompressed data/tensors and reports remain under this project root. Memory guards stop before an OOM; the controlled initial mmap-reader stop and subsequent fix remain in resource logs.

Read `reports/track1_retrain/dashboard.html` for paired reconstruction and support. Validation is exploratory; source-mask interpolation is retrospective, and source centroid motion can include occlusion/deformation/camera effects. Independent motion/contact/identity labels, fresh locked evaluation, full joint uncertainty, and long RGB sequence coverage remain open. Frozen v1/v2 artifacts remain unchanged.

## Readiness remediation and longer contexts

The separate development supplement is `dataset/context_v3` with Track 1 states in `derived/track1_context_v3`. It adds native, uncompressed six-second contexts and references existing central frames rather than duplicating them. It preserves the v2 model selection and train-only appearance PCA. Inspect its own status before ingestion; covariance calibration has not been revalidated on the extended contexts.

Read [readiness evidence](reports/track1_readiness/READINESS.md) and [dashboard](reports/track1_readiness/dashboard.html) for source-timing denominators, model-specific failures, contiguous lengths 4/8/16, and the immutable-shape/CV transport comparison. The original v2 source snapshot is SHA256-verified under `work/track1_readiness/v2_source_snapshot`; the optional reader root argument supports the new store without changing the default release.

Use `Dataset(root='derived/track1_context_v3')` with a valid row from that store's own functional metadata or sequence index.

Reproduce the audit and supplement from the existing raw videos with the same runtime environment above:

```sh
python3 -m track1_retrain.run --stage readiness-audit
python3 -m track1_retrain.run --stage context
python3 -m track1_retrain.run --stage transport
```

Independent reviewer worksheets remain blank and pending. The prospective fresh participant reservation is metadata-only; no fresh acceptance set has been acquired or validated. Track 2 remains on hold and no scientific finalization is claimed.

## Prepared-dataset refinement cycle

The completed follow-up mathematical training candidate is `derived/track1_finetuned`. See [TRACK1_FINETUNE.md](TRACK1_FINETUNE.md) for the refitted PCA, training-only temporal/detail selection, rebuilt tensors, matched validation results, resource measurements and the new reader. This v2 store remains the preserved comparison baseline.
