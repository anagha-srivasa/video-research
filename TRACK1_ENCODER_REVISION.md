# Track 1 research-report implementation

The development implementation is `src/track1_encoder_revision`, configured by `configs/track1_encoder_revision.json`. Artifacts are in `derived/track1_encoder_revision`; measurements, all attempts and gates are in `reports/track1_encoder_revision`. Open [the dashboard](reports/track1_encoder_revision/dashboard.html). Temporary requests, scripts, checkpoints, logs and empty-root rebuilds remain under `work/track1_encoder_revision`.

The existing `track1_finetuned` M16 carrier, causal kinematic estimator, source observations, frozen backbone/PCA, old training releases and human reference submissions are unchanged. The continuous certificate implementation was not edited: changing that source would invalidate existing decoder-bound certificates. This revision is an explicit optional development view. It does not finalize Track 1 or start Track 2.

## Proposal-to-implementation map

| Report recommendation | Implementation | Result / remaining boundary |
|---|---|---|
| Retained-only continuous/circular correspondence | `phase.py`, four matched variants in `experiments.py` | Implemented with strict static-mask guards; baseline A remains selected by training-only guards |
| Endpoint error decomposition | Target-exposed centroid, phase and source-spatial oracles in the per-probe ledger | Centroid/pose error dominates fractional phase on this bounded endpoint cohort; oracles never select parameters |
| Separate causal innovation/event channel | `events.py`, synthetic training-only selection in `event_training.py`, real source diagnostic streams | Improves event recall over the unchanged switching bank, but does not dominate CV; unpromoted |
| Local certificate diagnostics and repair | `geometry.py` wraps the unchanged certifier | 16 local selections / 84 probes, better average reconstruction than global rollback, no additional certified states recovered |
| Full covariance propagation | `covariance.py`, real refitted-contour diagnostic in `diagnostics.py` | Full conditional Jacobian propagation and cross-channel terms implemented; unestimated upstream/gauge/identity uncertainty remains explicit |
| Cause-aware stable-context expansion | `contexts.py`, streaming `appearance.py` / `source_stream.py` | Three new six-second contexts, 960 native frames streamed; cached features/tensors and sequence reader available without RGB copies |
| Independent-reference ingestion | Strict read-only `references.py` audit reuses existing blinded A/B/adjudication protocol | Zero accepted independent references; no conversion of Qwen outputs into human ground truth |
| Expensive offline work, cheap inference | Certificate-bound `CompiledGeometry` in `runtime.py` | Same default 512-point normalized shapes evaluate in approximately 0.011–0.028 ms in sampled warm tests |
| Clean hash-verified reproduction | Empty-root bounded rebuild with canonical polygons and original MP4 context extraction | All 75 data artifacts matched exactly; full raw-dataset canonicalization/backbone rebuild remains unverified |
| Contact-weighted allocation and calibrated association rejection | Explicit reference gates | Deferred as the report requires: independent contact geometry / identity labels are absent |
| Final lock and fresh evaluation | Machine-readable scientific readiness ledger | Not authorized or satisfied; no final freeze or fresh-label exposure |

## Phase alignment and endpoint diagnosis

The A/B/C/D ablation uses exactly the same retained observations and targets:

- A: current discrete cyclic correspondence.
- B: fractional circular alignment with a continuity penalty, constrained within the retained topology. Each retained frame must preserve both source-mask IoU and BF2. Rejected changes revert to the original frame encoding. A sampled reparameterization-displacement guard is additional engineering evidence, not a continuous geometry proof.
- C: B plus a fixed weak shape-mode regularization increment. It does not increase polynomial degree or control count.
- D: C plus the existing endpoint-neutral position detail, preserving its endpoint position/velocity invariants.

All hidden targets are removed before correspondence estimation, matching and fitting. Target-informed oracles are isolated evaluation fields. The strict training guard requires no loss of training-participant/protocol mask quality or certificate fraction. It retains A. B/C/D remain experiments; there is no automatic default replacement.

BF2 in these source-mask experiments uses **two annotation-grid pixels**, as in the existing renderer; it is not automatically a two-native-pixel contact-boundary score.

There are **504 timing-screened proposed holdout probes and 151 supported probes**. Support is identical across the four variants. `attempts.parquet` retains every failure and reason. These denominators are sampled source-derived opportunities, not independently preregistered dataset coverage. `phase_strata.parquet` reports entity/track, hand/object type, ring count and target encoding compatibility. Encoding compatibility includes topology/allocation and source-fit validity; it does not establish physical topology.

On the eleven supported exploratory validation endpoint probes, mean-video source-mask IoU is 0.76285 for A, 0.76270 for B, 0.76330 for C and 0.76329 for D. The report's proposed +0.03–0.08 endpoint improvement was not achieved. Do not compare these numbers directly with the previous finetuning report's endpoint IoU: the selected cohorts differ.

Using the same baseline endpoint probes, the deliberately target-exposed diagnostics give:

| Diagnostic | Mean-video IoU | Interpretation |
|---|---:|---|
| Actual discrete baseline | 0.76285 | No target correction |
| Target-informed phase only | 0.76364 | Fractional phase alone explains little on this cohort |
| Correct only centroid using the target | 0.89487 | Large source-centroid/pose contribution |
| Target-frame spatial encoding | 0.98756 | Source-spatial oracle; changes shape and topology together |

These are source-label diagnostics, not deployable forecasts or physical-motion ground truth. The spatial oracle does not isolate topology causally. Their purpose is to avoid spending more training on the wrong bottleneck.

## Local geometry repair

Failure records include the unresolved reason, normalized time interval, ring/piece locations, relevant hull pairs and a margin proxy. An unresolved hull test is not proof of an actual intersection. The repair modifies only implicated control-point deformation degrees of freedom, recertifies serialized float32 coefficients using the original proof code and budget, and compares retained-observation error with the existing global rollback. It falls back when local repair does not produce a better certified view.

Of 84 paired probes, 52 need no change, 16 select local projection, 11 retain global rollback and five remain unresolved. Both policies certify 79 probes: **zero new certified states** were recovered. Across those paired certified probes, local-or-fallback views improve held-out source-mask IoU by 0.00883 and BF2 by 0.00963 on average. These are pooled bounded development results, not independent acceptance.

Raw history coefficients and repaired geometry coefficients remain separate. No raw coefficient covariance is transferred through repair. The certificate concerns the normalized canonical curve; it does not certify raster topology, missing source components or physical lineage. Historical mask metrics use the existing spatial renderer; a continuous-curve certificate is not silently transferred to that renderer.

## Causal events

The innovation channel uses only past native-image source positions/variance. It preserves sequential residual/CUSUM evidence, uses a robust local trend with noise-aware significance, resets on gaps and latches events until the next consumer update. Insufficient recent support produces an explicit invalid measurement rather than NaN evidence. Onset/stop/reversal outputs are separate from the fixed switching position estimator.

A finite threshold search used only synthetic training seeds 0–59. It selected significance 5, an 80 ms trend window and one qualifying onset sample under a development false-alarm budget. Seeds 60–89 remain exploratory synthetic validation; earlier diagnostics had already been inspected. These thresholds are not final real-world thresholds.

| Hop | Method | Synthetic onset recall | False onsets / quiet minute |
|---|---|---:|---:|
| 125 ms | Switching bank | 0.294 | 0.000 |
| 125 ms | Innovation channel | 0.529 | 0.000 |
| 125 ms | CV | 0.647 | 3.557 |
| 250 ms | Switching bank | 0.235 | 0.000 |
| 250 ms | Innovation channel | 0.471 | 1.186 |
| 250 ms | CV | 0.588 | 2.371 |

The event channel improves the switching trade-off but misses the report's initial recall target and does not dominate CV. It remains unpromoted. Velocity error and 250/500 ms continuation are measured on the same episodes; adding the event channel leaves switching outputs identical. Existing source-centroid streams are also recorded, explicitly as source diagnostics. Synthetic onset definitions and quiet-minute rates differ from some historical fraction-based reports; do not mix their scores. Independent contact timing, reversal accuracy and physical motion remain unverified.

## Conditional uncertainty

The covariance module propagates a supplied joint control/centre/scale covariance through centering, scale normalization, the db2 transform and log scale using an explicit Jacobian. Full control correlations and shared centre/scale cross-channel effects survive. Supplied joint input cross-covariances are supported by the API; the real-data diagnostic assumes independent input centre/scale/control blocks because their cross-covariances have not been estimated.

The 56-mask diagnostic refits actual source contours and stores full conditional observation covariance matrices. Position covariance uses a numerical Jacobian of distortion-corrected intrinsic-normalized coordinates, matching the existing coordinate units. PSD, analytical Jacobian, finite-difference, normalized-unit and Monte Carlo checks pass. The helper's returned mean for a supplied position Jacobian is a linear map; existing nonlinear feature means remain authoritative. Stored diagnostic matrices are auxiliary observation uncertainty, not replacements for temporal state covariance.

Phase estimation, topology/identity alternatives, source annotation error, cross-window correlation, repaired geometry uncertainty and independent human calibration remain outside this conditional covariance. Their absence is recorded rather than represented as zero uncertainty.

## Stable contexts and ingestion

The planner searches raw canonical source tracks outside existing RGB intervals, checks full retained source ring graphs, completeness, native timing and support, and refuses to bridge graph changes. Only three contexts passed the bounded search: P04_109, P22_117 and P18_03. No additional eligible context was found for validation participant P08 in this bounded search; improved continuity across multiple validation participants was not demonstrated.

960 exact native frames were streamed from the original MP4s through a bounded pipe. The frozen MobileNet and locked training-only PCA produced cached object/context features. No new RGB shard, transcoding or compression was introduced. The extension contains 120 supported model states across both capacities, 100 complete certificate-bound geometry views and 120 valid appearance states. M16 remains a carrier even where its dominant-ring geometry is incomplete.

There are **10 training and five validation length-16 M16 sequences** in the added contexts. They overlap within only three episodes. Optional M32 geometry has its own explicit epoch, duration, roles and parent hierarchy in the M32 rows of each `*.context_states.parquet`; consult those rows when combining views. The source chain is provisional identity, not independently verified identity. The old default training release is not replaced.

```python
import pyarrow.parquet as pq
from track1_encoder_revision.runtime import RevisionDataset

index = pq.read_table(
    'derived/track1_encoder_revision/train.context_length16.parquet'
).to_pylist()[0]
with RevisionDataset() as dataset:
    batch = dataset.get_context_sequence(index)
    assert batch['coefficients'].shape == (16, 4, 35)
    assert batch['xi'].shape == (16, 4, 68)
    assert batch['appearance'].shape == (16, 4, 2, 128)
    # Respect appearance and kinematic support masks.
    # Optional geometry is a certificate-bound normalized-shape decoder.
    geometry = batch['geometry32'][-1]
    if geometry is not None:
        normalized_curves = geometry.evaluate(index['end_times_us'][-1])
```

For the phase/repair pilot, `RevisionDataset.geometry(...)` requires the source, track, timestamp, holdout protocol and variant explicitly. It rejects uncertified or incomplete geometry. It never silently substitutes a state from a different holdout or fills missing geometry with zeros.

## Inference cost

The new `CompiledGeometry` validates the coefficient/contract/decoder binding and precomputes time-independent shape evaluation. Its output matches the existing canonical decoder. Default resolution is still 512 points per ring; float32 state storage and float64 curve evaluation remain unchanged.

| Rings | Existing decoder median | Precompiled warm median | One-time compile |
|---|---:|---:|---:|
| 1, allocation `[32]` | 13.587 ms | 0.011 ms | 16.558 ms |
| 2, allocation `[24,8]` | 9.206 ms | 0.020 ms | 10.149 ms |
| 3, allocation `[16,8,8]` | 6.441 ms | 0.028 ms | 8.117 ms |

These are 100 paired repetitions for one sampled state per ring-count cohort on this machine. They cover **normalized-shape decoding only**. RGB decoding, segmentation/contours, appearance extraction, fitting, certification, pose projection and cold loading are excluded. No end-to-end RGB FPS or deployment latency budget is claimed. The optimization moves repeated decoder work into compilation without reducing output resolution or changing the mathematical curve.

## Validation, reproduction and remaining gates

All **108 tests** pass. The final audit checks 382 pilot certificate bindings/decoder comparisons and 100 complete context geometry bindings, finite tensors, PSD covariance, GL4 reconstruction, context identity/gap/epoch rules and actual sequence-reader batches. Original raw-video, camera, baseline and reference hashes match. The old decoder source is unchanged.

The empty-root bounded rebuild regenerated all source-backed context features and refitted the selected context polygons; **all 75 data artifacts matched byte for byte**. This is not a clean rebuild of the entire raw pilot's canonicalization, complete spatial stores and full backbone extraction. That larger gate stays false. The first-pass results and prior rebuild evidence are preserved under `work`.

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -u work/track1_encoder_revision/verify_release.py --resume
```

The guarded resume wrapper checks artifact, source, raw-input and earlier-baseline hashes before invoking the controller. The controller also checks source/config hashes at resume, uses a process lock and bounded spawned workers, and records failed stages explicitly. Raw videos stay unchanged. Memory guards retain the existing 20 GiB process-tree limit and 5 GiB system reserve. Resource samples, logs and the two rebuild roots provide engineering evidence, not unique physical-memory or thermal-limit certification.

To repeat a clean bounded rebuild, give the helper a **new empty** directory under `work`:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 work/track1_encoder_revision/clean_rebuild_check.py \
 --root work/track1_encoder_revision/another_clean_rebuild
```

To audit independent submissions without creating any labels:

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
 /opt/anaconda3/bin/python3.13 -m track1_encoder_revision.references \
 --output reports/track1_encoder_revision/reference_recheck.json
```

There are zero accepted independent references. Contact-aware control allocation, calibrated physical identity rejection, final event thresholds, independent uncertainty acceptance, fresh participant evaluation and a final scientific freeze therefore remain unavailable. Qwen outputs do not satisfy those gates. The implementation is complete as a bounded development revision, with failed hypotheses and conditional evidence retained.
