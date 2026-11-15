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

The current package has typed cloud and tube contracts, background-aligned flux proposals, a native-frame prompt-free YOLOE heartbeat, a provisional constant-velocity belief tracker, event-triggered functional segments, sampled 2-D proximity, an explicit-outcome EVI calculation, optional certificate-bound geometry, structured candidate filtering, a lazy YOLOE adapter and variable-epoch collocation/JVP helpers. Detection regions are not certified objects. The visibility field and depth ordering are not independently known; the current region quality uses an unknown marker rather than an invented depth estimate. Covariance and re-identification scores are not calibrated.

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
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track1_fmr.evaluate --frames 30
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track1_fmr.evaluate --frames 10 --mode blind
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m pytest tests/test_track1_fmr_belief.py tests/test_track1_fmr_tubes.py tests/test_track1_fmr_contracts.py tests/test_track1_fmr_measure.py -q --basetemp=work/track1_fmr_validation
```

## 2026-11-14 continuation: bounded dorsal grid

Added opt-in `Pipeline(dorsal_grid=160)` (or 128); `None` remains the native-resolution default. Ego-motion, exclusion masks, flux and heartbeat operate on the selected grid; normalized clouds still select best-view crops from native RGB. The frozen encoder baseline is untouched. Area thresholds and morphology are currently measured in extraction-grid pixels and are not resolution invariant.

The reproducible `scripts/exp_track1_fmr/01_bench_dorsal_grid.py` comparison decodes 30 identical native RGB frames once, alternates variant execution order, checks the source-ledger hash, records source hashes and RGB digest, and samples combined process-tree RSS against the 18 GiB cap. This is a short engineering diagnostic, not any of the six controlled qualification suites. Sampling is not a continuous peak-memory proof; both variants coexist and decoding is excluded from ingestion timings.

Final diagnostic: native median 233.08 ms/frame versus grid median 9.75 ms/frame. Native ended with 100 provisional tracks and 231 capacity rejections; grid ended with 0 tracks. Sampled combined process-tree RSS peaked at 1.17 GiB. Retained previous-frame RGB is 6,220,800 versus 76,800 bytes; this is not total ledger or process memory. See `dorsal_grid_diagnostic.json` for per-frame evidence.

**Grid candidate failed observation yield and remains opt-in, unpromoted.** Zero proposals must not be reported as successful rejection reduction or tracking coverage. Next remediation is resolution-aware proposal area/morphology and small-region preservation, followed by matched identity/reference evaluation before any downstream complexity. Existing full-cohort results above belong to their historical implementations; they are not rerun results for this candidate.

Validation: all 15 FMR tests passed, including grid exclusion dimensions, normalized centroid coordinates, exact native RGB crop selection, native default and invalid-grid rejection. Existing TorchScript deprecation warnings remain.

Reproduce with `PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_dorsal_grid.py`. Process enumeration must be permitted for the memory guard.

## 2026-11-14 revised remediation: Stage 1 only

The revised five-stage plan is sequential: Stage 2 and later remediation have **not** started because Stage 1 failed its real-video gate. Historical results above describe their saved source versions, not this candidate. The earlier grid candidate's edge heartbeat has now been removed from active FMR ingestion.

Implemented normalized flux area bounds (`min_area_frac=0.0005`, `max_area_frac=0.40`), morphology radius (`radius_frac=0.012`), scale-relative feature spacing/LK window/RANSAC tolerance, mean-flux × sqrt(area fraction) × compactness ranking, bounding-box NMS at IoU > 0.5 and a combined maximum of 15 emitted proposals per frame. Compactness/saliency are engineering proxies, not measured physical saliency. Pixel sampling floors remain where required by discrete images.

`PromptFreeHeartbeat` uses the official local `yoloe-26s-seg-pf.pt` checkpoint on native RGB every two seconds, discards labels, rasterizes native-normalized segmentation polygons on a common 160×160 measurement grid and produces provisional clouds. Using polygons avoids stretching model letterbox padding. It does not use a text encoder or an edge fallback. The adapter lazily loads weights; benchmark cold load/warmup is separately recorded. API/checkpoint reference: https://docs.ultralytics.com/models/yoloe/ . No 4 ms amortized inference assumption is asserted.

Native remains the default flux resolution; 160 and 128 grids remain opt-in. The heartbeat is now semantic model inference, so the pipeline is no longer entirely blind/class-free in its discovery mechanism. A caller may inject an explicit proposal provider for tests or controlled comparisons. The model is local and automatic dependency installation is disabled.

`01_bench_grid_invariance.py` locks episode IDs, parameters, code hashes, checkpoint hash and source-ledger hash in `protocol.json` before execution, refuses to overwrite a run directory, decodes each native frame once, alternates variant execution order, and records RGB hashes, decoded pixels, per-frame proposal counts and ingestion timings. A 50 ms process-tree RSS monitor checks the 18 GiB limit and writes a failure report on errors. Sampled peaks are not a continuous OS-enforced cap. Mean warm ingestion includes the scheduled heartbeat, proposal extraction and current tracker/tube work; cold model warmup and decoding are separately excluded. The same resident model is shared, but inference is executed separately for each variant. Both pipelines coexist in RSS measurements.

Training smoke: one 30-frame P02_122 episode, ratio 1.0, grid mean 13.75 ms and 15 final tracks. All proposals came from the native heartbeat; neither flux path produced proposals. This is not a flux-continuity success.

The predeclared five validation episodes (150 native RGB frames) produced:

| Episode | Video | Grid/native emitted proposals | Grid mean ms/frame | Grid final provisional tracks | Local gate |
| --- | --- | --- | --- | --- | --- |
| `02b25bb6d59d53d1fbbcaa90` | P08_09 | 0.833 | 13.16 | 10 | FAIL |
| `1a21d64b3bbbd12c58c4c76f` | P08_09 | 3.385 | 14.20 | 31 | FAIL |
| `a8008818663cba7b980a57e9` | P08_09 | 1.000 | 14.23 | 15 | PASS |
| `5d7739d543d8db9ffede8e0f` | P18_03 | 1.150 | 14.98 | 19 | PASS |
| `5ff0a3abb4efec01a9311dd1` | P18_03 | 0.588 | 16.23 | 40 | FAIL |

Sampled combined process-tree peak RSS: **1.89 GiB**. All five episodes had nonzero tracks; only two passed all local numeric checks. Ratios include heartbeat proposals, so per-source flux/heartbeat counts are also recorded to expose parity driven solely by shared semantic seeds. Confirmed/tentative counts are explicitly null: lifecycle confirmation and eviction belong to Stage 2 and are not implemented here. Provisional coasting is not observed target coverage.

**Stage 1 remains unpromoted.** Normalizing thresholds alone did not establish scale-invariant region extraction. Next work is to test the prescribed smaller area fraction (`0.0002`) or SLIC grouping on training data, then freeze a new protocol before evaluation. These validation participants already appeared in earlier exploratory audits and have now been inspected again; they must not be represented as a fresh locked holdout for tuning subsequent variants. A fresh evaluation split would be needed for independent qualification. No downstream identity, point transport, covariance calibration, query-binding or contact result is implied by proposal counts. Innovation likelihood fitting alone cannot establish NEES without independent state truth.

Verification: **17 FMR tests passed** (18 existing TorchScript deprecation warnings), including a two-resolution synthetic flux check, ranked NMS/cap behavior, native RGB heartbeat/crop coordinates and the existing belief/tube/ledger checks. Frozen `track1_encoder_revision` sources were untouched.

Artifacts: `stage1_train_v1/` and `stage1_validation_v1/` contain pre-run protocols, progress and final results. Reproduce into a new tag (existing tags are immutable):

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split validation --device cpu --tag stage1_validation_replay
```
