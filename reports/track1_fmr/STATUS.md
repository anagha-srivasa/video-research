# FMR Track 1 development status

This is an experimental parallel implementation. No baseline is replaced or scientifically accepted. Numerical support or an emitted prediction is not proof of correct object identity, physical contact, depth, or calibration.

## Existing asset triage

| Existing component | Proposed role | Current status |
| --- | --- | --- |
| `runtime.CompiledGeometry` and geometry certificates | Optional on-demand lens | Wrapped behind an EVI decision; requires a valid existing certificate and an explicit outcome model. |
| `events.InnovationEvidence` | Causal event evidence | Called by the development pipeline on observed clouds; independent real-event accuracy unverified. |
| `covariance` and historical calibration | Shape/covariance propagation | Historical calibration is conditional on a different source-position representation and has not been transferred to flux clouds. |
| `phase` and geometry fitting | Optional local geometry | Frozen baseline; not called by blind ingestion. |
| `appearance` and `contexts` | Baseline appearance and context | Preserved; optional RGB crops are cached in development tubes. |
| `references`, `diagnostics`, `experiments`, `run` | Baseline evaluation and orchestration | Preserved. Independent identity/contact references remain needed. |
| `source_stream` | Native exact-PTS decoder | Reused by the pilot; RGB is not duplicated. |
| `track1_dataset_v2` | Frozen source-comparison artifacts | Read-only. The current pilot uses the historical six-second native frame ledger and privileged seed for its seeded comparison. |

## Implemented components and boundaries

The current package has typed cloud and tube contracts, background-aligned flux proposals, a static edge heartbeat, a provisional constant-velocity belief tracker, event-triggered functional segments, sampled 2-D proximity, an explicit-outcome EVI calculation, optional certificate-bound geometry, structured candidate filtering, a lazy YOLOE adapter and variable-epoch collocation/JVP helpers. Detection regions are not certified objects. The visibility field and depth ordering are not independently known; the current region quality uses an unknown marker rather than an invented depth estimate. Covariance and re-identification scores are not calibrated.

The 2-D junction screen uses finite samples, so it is not a continuous-time collision proof. Physical-contact probability is left unknown when reference evidence is absent. A dimensionless contact probability cannot by itself be treated as a physical impulse. The full plan's blind discovery, calibrated EVI, EdgeTAM/depth, semantic grounding evaluation, complete Track 2 contract and all four falsification studies are not complete.

## Pilot results

`native_pilot.json` covers 30 frames of one P02_122 episode using a privileged seed derived from the original foreground points. The candidate was observed on 2 frames and coasted on 28; the old global-fit tracker had 1 observed frame. Identity was not independently verified and the predicted frames are not counted as successful object coverage.

`native_blind_pilot.json` covers 10 frames of that source episode with static edge and flux proposals. The initial configuration reached 100 provisional tracks and rejected 48 proposals. A changed initial covariance floor reduced rejections to 24 in this exploratory replay, but there is no evidence it preserved object identity or improved true coverage. The pilot cannot substantiate the requested 85% coverage gate. It is a training-participant diagnostic, not a locked participant test.

An experimental coasting candidate retains the last observed shape as a **predicted prior**. On a matched 30-frame training clip it produced 196 tube records versus 28, 168 of them predicted-only. An oracle allowed to select a different track at each frame had a tube within 50 pixels of the source centroid at 29/30 timestamps versus 6/30 before. A stricter fixed-ID check found the initial closest track already 135 pixels away and that same ID within 50 pixels at only 6/30 timestamps. Neither is an independent physical-reference test. This reveals why an oracle-nearest score and predicted support cannot establish correct tracking. The candidate remains unpromoted.

The full 20-episode blind engineering audit is checkpointed in `batch_status.json`: it processed all 6,078 post-seed frames, emitted 767 tubes and recorded 183,418 proposal-capacity rejections. Rejections are repeated per-frame proposals, not distinct objects. Peak sampled process-tree RSS was 4.18 GiB. Tube ledgers are uncompressed and carry explicit epistemic tags. On the retrospective source-centroid proxy with a 50-pixel gate at the first source annotation and one fixed ID thereafter, zero of the 20 target tracks could be initialized from an observed tube. Source centroids are not independently certified physical identities, but this result does not support the requested 85% tracking gate. The earlier stopped runs and source snapshots are preserved under `work/track1_fmr/audit_v1`, `audit_v2` and `audit_v3`.

An exploratory TAP-Vid DAVIS point diagnostic evaluated three public videos with labeled first-visible points. The FMR prototype assigns each labeled point to an observed region and keeps a fixed region ID with its initial point-to-centroid offset. With a 50-native-pixel initialization gate it initialized 2/55 points and mean average Jaccard was 0.000; a subsequent exploratory three-standard-deviation shape-ellipse gate initialized 1/55 and also scored 0.000. Uninitialized points count as failures. On the same three video/query sets, the previously run LK and TAPNext++ point trackers averaged 0.433 and 0.696, respectively. Their direct point initialization and tracking contracts differ from this region-to-point translation diagnostic, so the result is an end-to-end comparison, not a controlled model-only ablation. It shows the prototype is not yet a viable point-tracking system; it does not test physical contact.

The shape-ellipse protocol was then applied to all 30 public videos: FMR mean average Jaccard 0.0014 versus 0.3813 for the matched LK point baseline, with 147/650 points initialized and the 100-track cap reached in all videos. The first three videos were already inspected during protocol adjustment; on the other 27 videos the scores were 0.0016 versus 0.3755. All FMR and LK point predictions and visibility decisions are saved in `work/track1_fmr/tapvid_full_probe.*.npz`. Public point tracking is not the same task as independent project contact or object-identity evaluation.

The official YOLOE-26 small segmentation checkpoint and MobileCLIP2 encoder were downloaded. On one exploratory source image, the `hand` text prompt returned score zero. A separate 96×96 patch test used a synthetic 10,000-record ledger with one qualifying patch: the median of five warm end-to-end calls was 15.51 ms, including 1.17 ms median ledger filtering, with a zero detection score. Cold loading and text-prompt encoding were excluded. This limited timing probe is not a grounding benchmark and does not establish either the 15 ms or 95% reduction gate. See the [official YOLOE model documentation](https://docs.ultralytics.com/models/yoloe).

The synthetic matched historical reconstruction diagnostic (`tube_benchmark.json`) compared 60 validation episodes: 255 adaptive segments versus 1,258 fixed-window records, with position RMSE 16.58 versus 16.64 pixels on 13,840 common timestamps. These are retrospective fits to an existing synthetic state stream; the comparison does not measure serialized bytes, image-level accuracy, independent events, or ringing. It does not establish the requested real-data compression gate.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track1_fmr.evaluate --frames 30
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track1_fmr.evaluate --frames 10 --mode blind
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m pytest tests/test_track1_fmr_belief.py tests/test_track1_fmr_tubes.py tests/test_track1_fmr_contracts.py tests/test_track1_fmr_measure.py -q --basetemp=work/track1_fmr_validation
```

## 2026-11-14 continuation: bounded dorsal grid

Added opt-in `Pipeline(dorsal_grid=160)` (or 128); `None` remains the native-resolution default. Ego-motion, exclusion masks, flux and heartbeat operate on the selected grid; normalized clouds still select best-view crops from native RGB. The frozen encoder baseline is untouched. Area thresholds and morphology are currently measured in extraction-grid pixels and are not resolution invariant.

The reproducible `scripts/exp_track1_fmr/01_bench_dorsal_grid.py` comparison decodes 30 identical native RGB frames once, alternates variant execution order, checks the source-ledger hash, records source hashes and RGB digest, and samples combined process-tree RSS against the 18 GiB cap. This is a short engineering diagnostic, not any of the six controlled qualification suites. Sampling is not a continuous peak-memory proof; both variants coexist and decoding is excluded from ingestion timings.

Final diagnostic: native median 233.08 ms/frame versus grid median 9.75 ms/frame. Native ended with 100 provisional tracks and 231 capacity rejections; grid ended with 0 tracks. Sampled combined process-tree RSS peaked at 1.17 GiB. Retained previous-frame RGB is 6,220,800 versus 76,800 bytes; this is not total ledger or process memory. See `dorsal_grid_diagnostic.json` for per-frame evidence.

**Grid candidate failed observation yield and remains opt-in, unpromoted.** Zero proposals must not be reported as successful rejection reduction or tracking coverage. Next remediation is resolution-aware proposal area/morphology and small-region preservation, followed by matched identity/reference evaluation before any downstream complexity. Existing full-cohort results above belong to their historical implementations; they are not rerun results for this candidate.

Validation: all 15 FMR tests passed, including grid exclusion dimensions, normalized centroid coordinates, exact native RGB crop selection, native default and invalid-grid rejection. Existing TorchScript deprecation warnings remain.

Reproduce with `PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_dorsal_grid.py`. Process enumeration must be permitted for the memory guard.
