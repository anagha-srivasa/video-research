# Track 1 training and refinement on the prepared dataset

The completed development candidate is `derived/track1_finetuned`. Its `status.json` reports `training_complete=true` and `numerical_ingestion_ready=true`. The source package and all previous baselines remain unchanged. Track 2 remains on hold; no final freeze or baseline promotion was performed.

Open [the result dashboard](reports/track1_finetune/dashboard.html), [selection evidence](reports/track1_finetune/selection.json), [full evaluation](reports/track1_finetune/evaluation.json), and [verification](reports/track1_finetune/verification.json).

## What was trained

This follows the Track 1 blueprint's mathematical architecture. Spatial spline observations are the already-refined deterministic contour fits. No replacement geometry labels were supplied by Qwen. The frozen MobileNet features are reused, with no backbone weight updates. A neural encoder/generator/forecast loss would begin Track 2 and is outside this run.

- Refit separate object/context 128-dimensional PCA projections on **72,235 unique training observations**, deduplicated across both prepared timelines, with track/reliability weighting. Five training participants enter the fit; validation participants do not. Weighted training variance retained is 92.08% for object features and 90.12% for context features. All source features are projected into the new basis; new projection files are isolated from previous releases.
- Run 14 temporal configurations for M16 and M32 on deterministic, track-balanced training probes, considering regularization, robust loss, variance/provenance weighting and Huber thresholds. A family candidate must respect a 2% MSE regret bound for each training participant and protocol. Shape candidates also receive matched training mask-space confirmation. The search retained the baseline coarse position/shape/scale settings.
- Train the optional endpoint-neutral position detail over nine configurations. Both models selected three atoms with regularization **0.003**, compared with the previous 0.01. The detail changes reconstruction inside the history while remaining exactly zero at endpoint position and velocity.
- Refit approximate covariance multipliers on withheld training-source observations, balanced across participant contributions and protecting endpoint errors. This calibration follows parameter selection on the same training participants; it is empirical representation-error calibration, not an independent coverage guarantee. Detail and physical-velocity uncertainty remain uncalibrated.
- Recompute all functional, appearance, covariance, derivative and collocation tensors for both source timelines and both model capacities. The run processed **31,846 model/window attempts**, yielding **5,971 numerically supported model states** and **5,334 geometry certificate bindings**. These counts include two models and overlapping timelines; they are not independent physical examples.

The prepared index policy retains 15,626 deduplicated M16 opportunities and 4,114 valid M16 states. Unsupported attempts stay visible. Source/correspondence epochs and missing hops still break sequences. Sequence counts remain 3,377 train / 737 validation single states, 1,370 / 273 length-four sequences, 762 / 153 length-eight sequences, and **266 / 46 length-sixteen sequences**. These sequences overlap.

The authoritative local-CV source-centroid kinematics are bitwise unchanged from the original estimator. The refit history derivative is a distinct quantity. Optional geometry uses a certified new M32 history when available; otherwise its explicitly named view preserves a certified baseline history or latest stable source epoch. Missing geometry is `None`; no geometry covariance or physical identity is invented. Preserved geometry has its own epoch and duration and must not be interpreted as a recovered one-second physical trajectory.

## Matched exploratory validation

The same retained observations and withheld targets are used for each method. All parameter selection was locked before this evaluation. The validation participants have been used in earlier research, so this is exploratory confirmation. The full attempt ledger contains 4,032 proposed holdout probes, of which 2,327 are supported. Unsupported probes remain recorded separately; the conditional metrics below do not certify complete coverage.

| Model | Holdout | Baseline position NRMSE | Refined position NRMSE | Reduction |
|---|---|---:|---:|---:|
| M16 | Single frame | 0.06151 | 0.05926 | 3.67% |
| M16 | 100 ms gap | 0.07063 | 0.06860 | 2.87% |
| M16 | Last 100 ms | 0.15638 | 0.15604 | 0.21% |
| M32 | Single frame | 0.02049 | 0.02009 | 1.92% |
| M32 | 100 ms gap | 0.02669 | 0.02623 | 1.74% |
| M32 | Last 100 ms | 0.08721 | 0.08716 | 0.05% |

Shape NRMSE is unchanged. Full-mask IoU improves slightly on these probes, for example M32 internal-gap IoU from 0.87736 to 0.88040. These are withheld source-contour reconstruction measurements, not independent mask prediction or real-video dynamics benchmarks. Nearest and linear interpolation comparisons remain in the evidence and can outperform the compact representation. No public benchmark parity is established.

A separate matched PCA feature-reconstruction diagnostic is in `reports/track1_finetune/appearance_evaluation.json`. Validation error decreases slightly for both object and context features on each validation video. One training participant has a small increase despite improved aggregate training reconstruction; no identity/discrimination improvement is inferred from feature reconstruction.

## Loading and reproduction

```python
import pyarrow.parquet as pq
from track1_finetune.store import FinetunedDataset

index = pq.read_table(
    'derived/track1_finetuned/indices/train.length16.parquet'
).to_pylist()[0]
with FinetunedDataset() as dataset:
    batch = dataset.get_sequence(index)
    assert batch['coefficients'].shape == (16, 4, 35)
    assert batch['coefficient_cov'].shape == (16, 35, 4, 4)
    assert batch['xi'].shape == (16, 4, 68)
    assert batch['appearance'].shape == (16, 4, 2, 128)
    # Honor appearance_mask and kinematics[i]['support_valid'].
    # geometry32[i] is optional and includes explicit provenance, epoch and duration.
    assert batch['semantic_ground_truth'] is None
```

Do not mix old and new appearance coordinates. Keep the referenced source/derived folders in place: spatial data and original observation metadata are referenced rather than duplicated. Exact native RGB remains accessible through the prepared source-backed reader described in [TRACK1_TRAINING_READY.md](TRACK1_TRAINING_READY.md). No new RGB copies, transcoded clips or source data compression were created.

Run or resume from the project root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -u -m track1_finetune.run --stage all --resume
```

The stage controller is `src/track1_finetune/run.py`; data/PCA, fitting, and reader/verification code are in the sibling modules. Configuration is `configs/track1_finetune.json`. Scratch/checkpoints and flushed logs are under `work/track1_finetune`. Completed stages require matching source/config hashes; per-video/model refit receipts also check model/projection/episode fingerprints and output hashes. An explicit stage rerun is required after code changes. Full release and input hashes are in the manifests; `final_audit.json` records the post-run artifact and sequence audit.

Training uses exact small weighted float64 solves, rather than neural epochs. The frozen backbone is already cached, so CPU fitting and bounded parallelism are appropriate here. Eight process workers handle the larger refits; PCA uses bounded native threads. The workflow completed in about 133 seconds, with sampled process-tree RSS peaking at **7.80 GiB** and sampled system available memory staying above **24.28 GiB**. The guard limit is 20 GiB process-tree RSS with a 5 GiB system reserve. Shared mmap pages can be counted in multiple process RSS values.

## Acceptance

All **99 tests** pass. Checks cover withheld-target independence, candidate-selection safety, source/participant boundaries and existing numerical contracts. All newly generated tensors were checked for finite values; supported covariance matrices passed PSD checks; collocation tensors and endpoint-neutral detail were reconstructed analytically; certified curves retain coefficient/decoder binding. Every one of the 6,984 sequence indices was audited, and real reader batches passed, including length 16. All baseline/input and release hashes match.

The Qwen ledger remains machine-only evidence and was not used to fabricate contour/contact/identity/motion targets. Independent physical motion, contact, identity, event preservation, joint uncertainty, and fresh scientific evaluation remain open. Scientific acceptance is false. The training cycle is complete as an open Track 1 development candidate.
