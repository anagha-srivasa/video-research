# FMR Track 1 development status

This is an experimental parallel implementation. No baseline is replaced or scientifically accepted. Numerical support or an emitted prediction is not proof of correct object identity, physical contact, depth, or calibration.

**Current checkpoint (2026-11-18, observation/identity v4):** The single prompt-free stride-ten/native-capped detector, three-inlier batched flow, direct query particles, anti-hijack area guard, subpart adaptation and additional silver subset diagnostics are implemented and evaluated. **307 tests pass** and both integrity verifiers pass. **Stage 2 qualification fails; Stage 3 remains gated.** Mesh strict/lineage coverage is **74.41% / 77.00%**, agreement ≤8 px **38.55% / 38.78%**, and recovery **71 / 66 of 471**. Query-seeded hybrid agreement is **36.48% / 36.63%**, with **23 / 19 of 471** recoveries, below the targets. Joint DAVIS runtime is **24.51 ms/frame**; query-free native runtime is **26.71 ms/frame**, above 13.5 ms. Sampled RSS is **4.52 GiB** and detector VISOR recall **7/8 (87.5%)**. Lineage later P08 support reaches **93.37%**, but P21 support is only **0.57%**. Silver binding is 2/2 with **290/300** evaluable states; only **50/300 (16.67%)** satisfy the in-support proxy. Full scored-sequence NEES is **6996.49**; even the reference-conditioned subset is **181.28**. Model labels remain silver; no live calibration or physical identity/contact promotion. Frozen baselines and qualified Stage 1 remain unchanged. Final report: [observation_identity_v4/RESULTS.md](observation_identity_v4/RESULTS.md); calibration: [stage2_silver_v6/result.json](stage2_silver_v6/result.json). Earlier sections are historical snapshots.

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

## 2026-11-15 continuation: DIS flow, exclusion and spatial agreement

Implemented the revised Stage 1 path in `dorsal_flux.py` and `pipeline.py`. No Stage 2–5 module or frozen encoder source was changed.

- Flow inputs are Gaussian-prefiltered **before** decimation, with sigma = 0.5 × native/grid span on each axis. The Gaussian runs on grayscale used by DIS; a separate area-resampled RGB grid supplies color/moment observations. Native resolution remains the default; small grids are opt-in.
- OpenCV DIS ULTRAFAST supplies dense displacement. RANSAC fits a full affine field from a bounded background lattice outside known regions. Subtraction yields normalized **image displacement per frame**, not physical velocity or depth. Affine fitting does not remove arbitrary parallax, and optical flow cannot guarantee observability of textureless motion. Official API reference: https://docs.opencv.org/4.9.0/de/d4f/classcv_1_1DISOpticalFlow.html .
- Predicted track shape ellipses provide a binary known-density approximation. Novel energy is residual magnitude times (1 − known density), with motion threshold 0.008 normalized units/frame. Known pixels are excluded again after morphology. New flux proposals are saliency-ranked, box-NMS filtered and capped at **10 per frame**. The combined heartbeat/flux fusion retains its existing 15-proposal cap; pre-fusion and selected counts are recorded separately.
- The bounded `last_flux` result exposes residual flow, the pre-exclusion motion mask and the emitted flux union. No advection, confirmed lifecycle, covariance calibration or surface-point transport is claimed. Exclusion prevents new discovery within declared known support; a confirmed tracked region can correctly have no **novel** flux. Therefore residual-motion activity and novel-discovery activity are separately recorded. A nonzero novel-flux rule for every active clip is conservative and can fail even when all motion is already explained.

### Split and metric changes

`stage1_validation_v1` is now explicitly dev/tuning evidence; its files were preserved. `stage1_holdout_v2_split.json` was created exclusively, before DIS training trials or holdout RGB decoding. It locks eight deterministic 30-frame action-centered intervals (four P06_108, four P21_01), all exact frame PTS, source-video/frame/action hashes, and participant exclusions. Episodes are separated by at least five seconds. Selection uses only publisher action metadata and a fixed hash ordering, never model outcomes.

P06 and P21 are participant-disjoint from all recorded FMR/belief development and historical baseline training participants. They were historical baseline **test** participants, with pre-existing ingestion/QA artifacts; these data are fresh to this FMR protocol, **not globally untouched data**. EPIC action labels indicate expected activity but are not independent frame-level motion ground truth. After this evaluation, these participants must not be reused as unseen validation for a tuned successor. A strictly globally unseen holdout would need additional participants.

The benchmark now compares emitted flux-region union masks on a common 160×160 grid, using area occupancy >= 0.5 when projecting native masks. Per-episode `IoU_flux` is total framewise intersection divided by total framewise union (spatiotemporal micro-IoU); empty/empty is undefined, not a perfect score. No heartbeat pixels enter this metric. Compressed boolean mask arrays and their hashes are saved alongside per-frame source-separated counts, raw-motion activity, novel activity, exact RGB stream hash, decoded pixels, preprocessing/discovery/heartbeat latency and full pipeline warm ingestion latency. Proposal ratios no longer authorize promotion.

Each run locks code/checkpoint/split hashes and gate definitions in `protocol.json` before execution, refuses to overwrite its output directory, shares each RGB decode, and alternates execution order. Cold model loading/warmup is separately recorded. Warm per-frame timing includes native-frame preprocessing, scheduled heartbeat and the existing tracker/tube pipeline. The variants share a resident model but execute inference separately. Both pipelines coexist in RSS measurements.

The RSS monitor samples cached root/decoder PIDs every 50 ms; process discovery occurs before timing each decoded frame because macOS whole-system PID scans materially inflated earlier timings. This remains a sampled application guard, not an OS-enforced continuous peak limit. Discovery and decode are outside ingestion timings. Exceeding the cap or failing memory inspection aborts the run with a failure report.

### Training diagnostics before the locked evaluation

The original 160/.008 DIS trial yielded no novel proposals on the training P02_122 clip and 36.60 ms mean ingestion. Grayscale prefiltering and moving PID discovery outside timing reduced the later trial to 28.24 ms; no spatial IoU could be computed because both emitted unions remained empty. The prescribed 128/.005 fallback was also tried **only on training**: native emitted one novel flux proposal and grid emitted zero, with grid mean 31.84 ms. Its mean flow-discovery cost was 0.73 ms but preprocessing cost was 26.51 ms. The original 160/.008 configuration was therefore retained for the predeclared holdout, without tuning to holdout outcomes. All intermediate protocols/results and the training profile are preserved in `stage1_dis_train_*`.

### Eight-episode holdout result: FAILED

Every episode used 30 identical native RGB frames across variants (240 decoded frames total). Activity denominators below exclude the first frame, where no previous frame exists.

| Episode | Video | IoU_flux | Grid nonempty novel-flux frames | Grid mean ingestion ms/frame |
| --- | --- | --- | --- | --- |
| `dis_e56c2c85b7a2500492f8d63f7c916816` | P06_108 | 0.0000 | 0/29 | 29.35 |
| `dis_b687e690851e200f1038dda65c6cd693` | P06_108 | 0.0364 | 19/29 | 32.72 |
| `dis_8c45e74c68ae0633ee585c2bb901f39d` | P06_108 | 0.1376 | 3/29 | 29.84 |
| `dis_e5ffcb291dc874d2a6220cc7cc48ad72` | P06_108 | 0.0000 | 0/29 | 29.56 |
| `dis_4d18a7447afaca7440111aac9f3e0558` | P21_01 | 0.0423 | 15/29 | 33.30 |
| `dis_4ff5f990c9d79a8245c850ab8263a020` | P21_01 | 0.0060 | 1/29 | 29.59 |
| `dis_2d54d66bfd929751b1376e8e26f28578` | P21_01 | 0.0000 | 0/29 | 29.98 |
| `dis_9b6c19ce677048c0ea2f82790faada67` | P21_01 | 0.1238 | 13/29 | 34.41 |

**0/8 episodes passed** the joint local gate. Grid spatial IoU ranged from 0.0000 to 0.1376 (required >= 0.65); three grid episodes had no emitted novel flux (native had nonempty novel flux in all eight). Grid mean warm ingestion ranged from 29.35 to 34.41 ms/frame (required <= 15). Grid preprocessing alone averaged 23.45–24.68 ms/frame; flow-discovery averaged 0.96–1.84 ms/frame. Peak sampled combined process-tree RSS was **1.90 GiB**, within the 18 GiB limit. Counts and timing establish neither object identity nor observed target coverage.

The supplied low-contrast/parallax diagnoses remain hypotheses about the earlier real episodes: synthetic low-contrast motion is detected, but the real-video qualification result does not establish that those failures are fixed. The empirical compute envelope is **not** qualified under the revised Gaussian preprocessing requirement.

**Stage 1 remains unpromoted; Stage 2–5 remediation was not started.** Next engineering work belongs on training/dev data: reduce measured prefilter/resampling cost while preserving its declared anti-alias transfer, and test whether flow is lost before exclusion, within the known-region approximation, or at morphology/area filtering. Do not disable exclusion merely to increase discovery counts. Freeze a successor before using additional unseen participants; the evaluated `stage1_holdout_v2` is now consumed.

### Verification and reproduction

All **19 FMR tests passed** (18 existing TorchScript deprecation warnings). Checks include low-contrast translated texture whose RGB residual stays below the old threshold, dominant camera translation subtraction, known-density discovery suppression, proposal bounds, anti-aliasing and the existing contracts/belief/tube/ledger checks. All protocol/source/mask hashes were verified after the holdout run, and all eight reported IoUs were recomputed from the saved `(30, 2, 160, 160)` boolean mask arrays. `git diff --check` passed. No model training, independent physical-reference evaluation or downstream promotion occurred.

Locked artifacts: `stage1_holdout_v2_split.json` and `stage1_holdout_v2/{protocol,progress,result}.json`, plus one flux-mask NPZ per episode. Replays must use a new tag and are replays of consumed data, not fresh validation:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split holdout --device cpu --tag stage1_holdout_v2_replay
```

## 2026-11-15 continuation: Stage 1 v3

Implemented the four v3 changes on consumed development data before selecting fresh holdout episodes. `belief_filter.py`, tube fitting, semantic querying, contact modules and `track1_encoder_revision` remain unchanged by this continuation.

### Extraction and preprocessing changes

- Grayscale flow inputs use native-to-2×target `INTER_AREA`, a 5×5 sigma=1 Gaussian at that intermediate resolution, then `INTER_LINEAR` to the target. This replaces the native-resolution Gaussian. It is a **different filter response**, not an exact transfer-function or Nyquist-equivalence certificate. RGB for region color moments uses separate fast linear sampling; native RGB still supplies cached crops. Color signatures under this sampling choice remain uncalibrated.
- DIS uses finestScale=0 at 160 (or 128), finestScale=1 at 320, patch size=8, stride=3. Thus 160@0 and 320@1 share nominal normalized patch footprint; this does not guarantee equal flow fields. Normalized motion floor is 0.0035/frame and minimum area fraction is 0.00025. Existing maximum area, morphology and proposal-ranking rules remain. Novel extraction is capped at ten, followed by combined heartbeat/flux NMS capped at fifteen.
- Known ellipses exclude background-fit samples but no longer erase all discovery inside them. Discovery suppression is max(density × velocity agreement), with sigma_flow=0.005. State velocities are **image-coordinate normalized units/second**; each is multiplied by the exact PTS interval to obtain displacement/frame. Its affine-background displacement is subtracted before comparison to residual flow, so coordinate frames and units agree. Weights are evaluated only at occupied density pixels to reduce cost without changing the formula.
- Spatial overlap alone no longer suppresses onset. However, the proposed parameters do **not** guarantee every subtle onset passes: for zero camera flow, a static seed and residual magnitude 0.004/frame, the formula produces explained weight about 0.726, so remaining novelty is only about 0.0011, below 0.0035. A regression check records this mathematical limitation. Weight agreement is an engineering heuristic, not a calibrated motion likelihood.
- Pipeline retains the actual selected observed proposal union for reference evaluation; predicted/coasting support is not included in reference coverage. Residual image motion is not called physical velocity. No lifecycle, flow advection or independent uncertainty-calibration acceptance is implied.

### Consumed dev pool and manual-reference metric

`stage1_dev_pool_v3.json` merges the one P02 smoke clip, five consumed P08/P18 clips and eight consumed P06/P21 clips, plus five deterministic exact-manual-reference clips from those same participants: **19 episodes** total. Reference clips begin at the externally annotated timestamp, which is also the first scheduled heartbeat. This reference-phase sampling is explicit. The extractor receives RGB only; reference masks are loaded by the evaluator after split locking, never used to initialize or modify proposals.

The scale metric is framewise flux-union agreement between 160 and 320, projected onto a common 160 evaluation grid by area occupancy >=0.5, aggregated as intersection/union over space and time. Heartbeat masks do not enter scale IoU. Empty/empty stays undefined. Saved NPZ arrays explicitly order the comparison as `[grid320, grid160]`.

The new reference metric is **labeled-pixel spatial coverage recall** of the actual <=15 selected heartbeat+novel proposals at exact annotation timestamps. It is not object-instance recall, tracking recall, IoU, precision or identity accuracy. It uses unions of VISOR `manual_sparse` source polygons; unlabeled objects do not become false negatives, and no exhaustive scene annotation or physical-identity truth is inferred. The reference diagnostic has no newly invented promotion threshold.

Final dev extraction candidate (`stage1_v3_dev_v2`): novel flux in 19/19 episodes, scale IoU 0.2091–0.8934 and mean grid ingestion 19.15–38.10 ms/frame. **44 manual-mask instances** across the reference timestamps yielded **51.00%** aggregate labeled-pixel coverage. Sampled RSS peaked at 4.01 GiB. The second dev run restricted velocity-weight computation to occupied pixels; its 19 saved IoUs were unchanged from the preceding extraction candidate (all per-episode IoUs match), while timing improved. Subsequent changes added only the fresh-source decoder and protocol input checks before the final freeze.

### Genuinely new local participants and freeze order

No strictly unused participant remained in the original local dataset. P03/P04/P11/P22/P27/P30 were historical training participants, so they were not relabeled as fresh evaluation. Instead, two new short source videos, **P14_02** and **P26_15**, were acquired (122,064,683 and 135,474,262 bytes) using a pinned public mirror. Both full-file publisher MD5 and SHA256 were checked; publisher URLs, checksums and exact transport revision are in `stage1_v3_acquisition.json`.

The original `dataset/videos` is read-only and remained untouched. Fresh sources reside at `work/track1_fmr/P14_02.mirror.MP4` and `P26_15.mirror.MP4`. A project-scoped isolated FMR decoder supports these paths without changing the frozen baseline decoder. It enforces exact source PTS, native dimensions, sequential RGB pipe records, and errors rather than substituting nearby frames.

An initial code freeze was preserved, followed by `stage1_v3_code_freeze_v2.json` after adding fresh-source I/O. **The final decoder-capable code was frozen before episode selection or fresh RGB decoding.** `00_lock_stage1_holdout_v3.py` then demuxed compressed packet PTS metadata only and chose four deterministic action-centered 30-frame episodes per participant, with >=5 second center separation. `stage1_holdout_v3_split.json` locks all eight IDs, exact PTS, input hashes and the preselection code hashes. No fresh frames were used for tuning. These participants are new to recorded project training/evaluation; exposure in external pretrained checkpoints is unknown. Publisher action labels indicate expected activity, not independent frame-level motion adjudication.

Each run writes its protocol and a complete relevant Python source snapshot before decoding, refuses to overwrite a tag, alternates paired order, shares exact decoded RGB, and records code/checkpoint/split/RGB/mask hashes. The new holdout rejects participant overlap and changed code. Input/reference hashes are checked on dev as well. RSS remains a cached-PID 50 ms sampled guard, with per-frame discovery outside timing; it is not an OS-enforced hard maximum. Cold model warmup and decoding are excluded from warm ingestion; scheduled heartbeat, preprocessing and current tracker/tube work remain included. Protocol reports OpenCV 5.0.0, **16 OpenCV workers** and four Torch workers; `OMP_NUM_THREADS=1` did not establish a single-threaded OpenCV measurement.

### Locked stage1_holdout_v3 result: FAILED

240 exact native RGB frames were decoded once across the eight episodes, then supplied identically to both variants. Original source cadence is about 59.94 FPS; the fixed per-frame threshold is not equivalent to a 30-FPS speed threshold. State-to-displacement conversion uses actual timestamps. Each short episode includes its initial heartbeat; these measurements are not a long-running steady-state amortization study.

| Episode | 160/320 flux IoU | Grid novel-flux frames | Grid mean ingestion ms/frame |
| --- | --- | --- | --- |
| `v3_P14_02_2` | 0.4449 | 11/29 | 21.53 |
| `v3_P14_02_0` | 0.3414 | 14/29 | 25.56 |
| `v3_P14_02_7` | 0.5848 | 14/29 | 26.99 |
| `v3_P14_02_6` | 0.1382 | 15/29 | 24.19 |
| `v3_P26_15_2` | 0.4436 | 15/29 | 22.58 |
| `v3_P26_15_6` | 0.5835 | 14/29 | 23.11 |
| `v3_P26_15_0` | 0.0880 | 12/29 | 23.50 |
| `v3_P26_15_1` | 0.8279 | 15/29 | 20.73 |

**Novel-flux episode activity: 8/8 (100%)**, above the 85% gate, with no silent episode in this sample. **Scale consistency: 1/8 >=0.65**, range 0.0880–0.8279. **Latency: 0/8 <=15 ms/frame**, range 20.73–26.99. Mean preprocessing was 7.88–7.98 ms/frame; mean discovery (DIS, affine fit, velocity weighting and proposal/moment extraction combined) was 6.73–9.78 ms/frame. The requested ~0.5 ms preprocessing and 6–9 ms full-ingestion estimates were not achieved. Peak sampled combined process-tree RSS was **1.46 GiB**, within the 18 GiB gate. Fresh participants have no local canonical VISOR masks, so reference recall is reported only on dev/reference data, not invented for this holdout.

**0/8 episodes passed the joint Stage 1 v3 gate. Stage 1 remains unpromoted; Stages 2–5 were not started.** Nonzero flux is an observation-yield improvement, not proof of useful or correctly identified objects. V3 resolves the earlier all-zero training discovery case but does not qualify scale consistency or ingestion speed. Future work must use consumed dev data to isolate the remaining resampling, fine-scale flow/weighting and provisional-track costs; a tuned successor needs additional untouched participants. P14/P26 and this v3 holdout are now consumed.

### Verification and reproduction

All **22 FMR tests passed** (18 existing TorchScript deprecation warnings), including velocity-consistent duplicate suppression versus onset, exact dt/camera-frame displacement units, explicit DIS scales, anti-aliasing, and a real H.264 codec round-trip that verifies exact nonadjacent PTS selection and missing-PTS rejection. Existing belief/contracts/tube/ledger tests pass. Frozen encoder sources were not modified. All final protocol/code/checkpoint/split/input/mask hashes were checked, and all eight spatial IoUs were recomputed from saved mask arrays; `git diff --check` passed. No model training or downstream promotion occurred.

Artifacts: `stage1_dev_pool_v3.json`, `stage1_v3_dev_v2/`, `stage1_v3_code_freeze_v2.json`, `stage1_holdout_v3_split.json`, `stage1_v3_acquisition.json` and `stage1_holdout_v3/`. Fresh sources remain in work and are not embedded in the report. Replays require a new tag and cannot be called unseen validation:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split holdout --device cpu --tag stage1_holdout_v3_replay
```

## 2026-11-15 continuation: provided stateful drop-in integration

Integrated the supplied `DorsalFluxExtractor`, `TrackPriorMask`, `FluxProposal` and `FluxStepResult` contract into `dorsal_flux.py`. The current pipeline now uses the stateful extractor directly rather than recreating DIS per frame. Its reset/shape behavior, cached DIS configuration, component-level velocity telemetry, raw/novel/explained fields, intermediate-resolution Gaussian, RGB `INTER_AREA` path, 24×24 background lattice, 0.004 normalized RANSAC threshold, 0.5×motion-floor novelty threshold, morphological **closing**, compactness floor and post-closing explained-density gate (>0.65) follow the supplied variant. The existing moment/CloudMeasure, local prompt-free heartbeat and functional `flux`/`anti_alias` entry points remain available; functional callers share the same analysis kernel.

### Integration corrections and declared limits

The pipeline rasterizes **soft Gaussian densities from shape scatter S**, separately from belief covariance B, on the extractor's current grid. Predicted image velocities are converted from units/second to displacement/frame using exact source PTS intervals. Active density arrays are finite/range/shape checked and immutable. Native uint8 RGB and extractor parameters are checked at entry; no upsampling, mismatched prior grids or invalid timestamps are silently accepted.

State centroids live in **image coordinates**, while `FluxProposal.mean_vel_norm` is camera-residual displacement. Comparing those directly or initializing the image-state velocity from residual/dt alone would mix coordinate frames. The integrated exclusion removes the same fitted camera displacement from predicted and measured motion (equivalently comparing raw image flow to image displacement). Each proposal therefore also exposes `mean_image_vel_norm`. For **new unmatched flux births only**, `BeliefTracker.update(..., birth_kinematics=...)` consumes mean image displacement / exact dt and velocity covariance `(sigma_flow/dt)^2 I`; heartbeat births keep their existing unknown/zero-velocity initialization. Mean flow accumulation uses float64. A test with camera displacement 0.01, object residual 0.02 and dt=0.02 verifies image-state birth velocity 1.5/s rather than the erroneous residual-only 1.0/s.

This is the explicitly requested narrow birth-initialization change to `belief_filter.py`, **not** Stage 2 lifecycle, advection, identity or calibration promotion. Existing association/coasting rules remain experimental, and supplied velocity covariance is a heuristic rather than calibrated evidence.

Affine evaluation uses actual sampled normalized pixel centers consistently, avoiding the provided unsnapped-lattice/full-grid-center mismatch. Background fallback modes are explicit (`all_pixels_fallback` / `median_fallback`); their flow may be retained diagnostically but their proposals are not accepted observations. Out-of-image flow support is rejected. These safeguards are deliberate corrections to a literal paste of the supplied code.

Strong onset versus matched motion is regression-tested. The soft formula is not a universal onset guarantee: low speeds near the configured floor can still be attenuated below the half-threshold, and Gaussian density is below 0.65 over much of a region's outer footprint even for matching motion. Unit success must not be generalized to perfect duplicate elimination or arbitrary low-contrast motion recovery.

### Pre-holdout check on consumed data

The code was evaluated on the full existing **19-episode `stage1_dev_pool_v3`**: P02 smoke, consumed P08/P18 and P06/P21 action clips, plus their exact-manual reference clips. The pair remains 160@DIS scale0 versus 320@scale1, with shared exact RGB decode, alternating order, <=15 selected proposals/frame, source-separated counts, saved masks and spatiotemporal micro-IoU. No P14/P26 or other fresh evaluation data were used in this successor run.

Before decoding, `stage1_dropin_dev_v1/protocol.json` locked parameters, source/checkpoint/split hashes and a full relevant source snapshot. `stage1_dropin_unit_checks.json` records the tested source/test hashes. The RSS guard sampled cached root/decoder PIDs every 50 ms, with process discovery outside timed ingestion. Model warmup and decoding are excluded from warm ingestion; prior rendering, cloud moments, heartbeat, existing tracker/tube work and initialization-frame scheduled heartbeats remain included. No steady-state amortization claim is made.

| Requested pre-holdout check | Measured result | Outcome |
| --- | --- | --- |
| Pooled preprocessing median <=1.0 ms/frame | **15.98 ms/frame**; per-episode medians 15.92–16.08 | FAIL |
| Full warm ingestion <=12.0 ms/frame | Pooled mean **69.70 ms/frame**; episode means 47.98–99.86 | FAIL |
| Three previously silent clips recover novelty without cap flooding | All three recover; one clip reaches ten proposals on one frame | Recovery passes; conservative no-saturation check fails |
| Unit checks: stationary onset / moving duplicate suppression | **25 FMR tests pass**, including exact-dt/camera-frame birth units and cached DIS/reset | PASS |

The focus clips were:

| Previously silent episode | Nonempty novel-flux frame pairs | Frames reaching ten-proposal cap |
| --- | --- | --- |
| `dis_2d54d66bfd929751b1376e8e26f28578` | 14/29 | 1 |
| `dis_e56c2c85b7a2500492f8d63f7c916816` | 23/29 | 0 |
| `dis_e5ffcb291dc874d2a6220cc7cc48ad72` | 28/29 | 0 |

One capped frame is not independently sufficient evidence of a sustained flood. The executable precheck uses a **conservative zero-cap-saturation interpretation**, explicitly recorded in `preholdout_checks.json`; this interpretation is not a new object-quality metric and is not the reason timing fails.

Novel flux appeared in **19/19 episodes**. Scale micro-IoU ranged 0.2759–0.7867; it is not a semantic or identity score. Four episodes reached the existing 100-provisional-track limit by their last frame; the general Stage 2 confirmation/eviction lifecycle was not added. Peak sampled combined process-tree RSS was **4.07 GiB**, below 18 GiB. Existing TorchScript deprecation warnings remain.

An isolated synthetic-1080p microprobe (20 warm repetitions after three warmups) further separated preprocessing from tracker cost. Median timings were: native grayscale 0.14 ms, grayscale `INTER_AREA` to 320 3.85 ms, small Gaussian 0.060 ms, final grayscale linear sampling 0.004 ms and RGB `INTER_AREA` to 160 **8.08 ms**. This synthetic timing diagnostic is not real-data throughput or scientific acceptance. It does show that the supplied resampling path cannot be assumed to cost 0.5–1 ms on this setup. Exact records and worker count are in `preprocessing_microprobe.json`.

### Gate disposition and artifacts

**The four pre-holdout checks did not all pass. No new freeze, split selection, participant acquisition or holdout RGB evaluation was performed for this successor.** The old P14/P26 split remains consumed, immutable evidence for the preceding candidate. The benchmark explicitly rejects holdout mode for this dev-only drop-in variant; do not overwrite or reuse `stage1_holdout_v3_split.json` as fresh data. Stage 1 remains unpromoted and general Stage 2–5 remediation remains closed. Further work should stay on consumed dev data and address measured resampling, soft-prior and existing tracker costs before any new holdout is selected.

All protocol/source/mask hashes were verified, and all 19 spatial IoUs were recomputed from saved boolean masks. `git diff --check` passed. The tests include the existing belief/contracts/tube/ledger/real-codec checks, plus stateful DIS cache/reset, prior/config rejection, Gaussian-prior integration and flow-informed birth velocity/covariance. No independent identity/calibration/contact claim is made.

Artifacts: `stage1_dropin_unit_checks.json` and `stage1_dropin_dev_v1/{protocol,progress,result,preholdout_checks,preprocessing_microprobe}.json`, with per-episode mask NPZs and the exact evaluated source snapshot. A new tag is required for a dev replay:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split dev --tag stage1_dropin_dev_replay
```

## 2026-11-15 continuation: local super-Gaussian and lazy color dev_v2

Implemented the three requested changes in the active FMR ingestion path, then ran the exact requested `--split dev --tag stage1_dropin_dev_v2` on all **19 consumed dev/reference episodes**. No new participant, fresh RGB evaluation, code freeze or `stage1_holdout_v4_split.json` was created. General Stage 2 lifecycle/eviction remains untouched by this continuation; existing flow-informed births remain as previously implemented.

### Code changes and limits

1. `preprocess_gray_pyramid` uses repeated `cv2.pyrDown` while **both** axes can halve without going below their target, then one final linear resize. It allocates no RGB grid. For 1080p ->160 this means 1920×1080 ->960×540 ->480×270 ->160×160. For the 320 comparator it stops at 960×540 before resizing. The final horizontal resize is 3:1 in both cases, so this is the supplied stopping rule, **not an exact Nyquist/transfer-equivalence certificate**. No ARM instruction-level implementation assumption is used as measured evidence.
2. `build_local_super_gaussian_prior` renders `exp(-(d_M²)²/32)` only inside a clipped 2.5σ bbox. `TrackPriorMask` now carries the local density patch, global grid size and offset; background-mask composition and motion-consistent exclusion use those slices directly. There is no per-track full-grid density materialization or full-grid exponential. SPD/finite shape and ROI bounds are checked; off-screen priors safely return empty. A dense-reference regression verifies the same formula inside the ROI. S remains the shape carrier, not belief uncertainty B.
3. Full-frame color resampling was removed from both active extraction and heartbeat moment construction. Geometric moments use the proposal grid, while `extract_proposal_lab_moments` samples only actual foreground positions in the native bounding crop, at most **1,024 positions**. The sampled Lab conversion is packed into a tile of at most 32×32; padding is excluded from statistics. This enforces the sampling limit that the pasted helper did not enforce and avoids its background-color fallback when a thin mask disappears after resizing. Heartbeat moments are computed after NMS, and flux moments only for the <=10 kept proposals. The explicit compatibility-only RGB `anti_alias` helper remains for old callers, but active per-frame ingestion does not use it. Color moments are approximate sparse samples in OpenCV uint8 Lab encoding, not newly calibrated semantic features.

A 2σ shape-moment contour is not a verified physical boundary. The plateau is a changed suppression heuristic, and its cutoff still has finite boundary effects. The prior's >0.25 background-exclusion footprint is wider than the old Gaussian's, independently of the >0.65 discovery gate. Its effect on available background support must therefore be measured, not assumed harmless.

### Executed dev gates

`stage1_dropin_dev_v2/protocol.json` locked the new filter, local prior/color rules, code/checkpoint/split hashes and exact source snapshot before shared RGB decoding. The 160/320 paired order, source-separated telemetry, saved masks, manual references and 50 ms process-tree RSS guard were retained. Prior-render latency and actual local/full-grid pixel counts are now reported separately. Fresh holdout mode remains blocked for this dev-only variant.

The new anti-flood criterion is implemented as requested for each focus clip: **cap-hit fraction <=10% over 29 frame pairs** (at most two capped pairs) and median **<=5 novel proposals/frame**. The first initialization frame is excluded from this denominator. All 19 episodes also report their cap fractions and medians. The full-ingestion gate now uses the requested **pooled mean <=12 ms**, not the preceding stricter every-episode mean interpretation; per-episode means remain visible.

| Dev gate | Measured result | Outcome |
| --- | --- | --- |
| Pooled median preprocessing <=1.0 ms/frame | **0.415 ms/frame** | PASS |
| Pooled mean full warm ingestion <=12.0 ms/frame | **16.50 ms/frame** | FAIL |
| Three focus clips recover novelty, cap fraction <=10%, median <=5 | Two silent; third capped 10/29 pairs | FAIL |
| Unit suite + local-bbox super-Gaussian regression | **27 tests passed** | PASS |

| Focus episode | Accepted novel-flux pairs | Pairs at ten-proposal cap | Median novel proposals |
| --- | --- | --- | --- |
| `dis_2d54d66bfd929751b1376e8e26f28578` | 14/29 | 10/29 (34.5%) | 0.0 |
| `dis_e56c2c85b7a2500492f8d63f7c916816` | 0/29 | 0/29 (0.0%) | 0.0 |
| `dis_e5ffcb291dc874d2a6220cc7cc48ad72` | 0/29 | 0/29 (0.0%) | 0.0 |

The two silent focus clips (`dis_e56c2c...` and `dis_e5ffcb...`) report **`all_pixels_fallback` on all 29 frame pairs**. Their broader composed prior footprint leaves fewer than twelve eligible background lattice samples. The extractor can compute fallback diagnostics, but those fits are not accepted as background-supported observations; their proposals are therefore rejected. Zero accepted flux here must not be presented as zero true motion or successful duplicate suppression. The third focus clip has 14 nonempty pairs but cap saturation of 34.5%, failing the new sustained-flood threshold despite its low median.

Across the pool, accepted novel flux appears in **11/19 episodes (57.9%)**, down from 19/19 in dev_v1. There are 264 insufficient-background fallback pairs out of 551 post-initialization pairs. Eight scale comparisons have empty/empty unions and are **undefined**, not scored as IoU 1.0. Defined scale micro-IoUs range 0.0168–0.4764. Per-episode full ingestion means range 10.84–28.10 ms/frame. Labeled-pixel reference coverage is **50.92% over 44 manual-mask instances**, a source-segmentation coverage diagnostic rather than identity, precision or observed target-tracking coverage.

The active prior patches account for 36,392,369 evaluated density pixels versus 515,481,600 hypothetical full-grid track pixels across the pool, a **14.16× geometric pixel-count reduction**. This is not a claim of 25–50× measured runtime speedup. Mean prior rendering, including its prediction/validation work, is about 4.03 ms/frame. Full ingestion still includes the current tracker, cloud moments, events, crops and the scheduled initial heartbeat per short episode; timings do not establish long-running steady-state amortization. Peak sampled combined process-tree RSS was **3.95 GiB**, within 18 GiB.

### Disposition and reproducibility

**Two of four pre-holdout checks pass. All four do not pass, so no new code freeze, new holdout split or Stage 2 opening occurred.** The speed change is real on this dev protocol, but novelty/background support and anti-flood behavior regress. Further remediation should stay on consumed data and separate background-fit support from the broader discovery plateau without accepting unsupported foreground-inclusive fits as reliable background. The result does not establish that remaining failures are purely computational or geometric.

All **27 FMR tests passed** (18 existing TorchScript deprecation warnings), including dense/local prior equality with correlated scatter, bounded/off-screen ROI behavior, local suppression coordinates, the expanded plateau values, <=32×32 foreground-only Lab conversion and empty-mask rejection, plus all earlier stateful/camera/PTS/birth/belief/tube/ledger/real-codec checks. Current source/protocol/mask hashes were checked, all 19 IoUs were recomputed from saved `(30,2,160,160)` masks, and `git diff --check` passed. The frozen encoder baseline was not modified.

Artifacts: `stage1_dropin_dev_v2_unit_checks.json` and `stage1_dropin_dev_v2/{protocol,progress,result,preholdout_checks}.json`, per-episode mask NPZs and exact evaluated source snapshots. All runs remain consumed development evidence. The requested tag now exists and is immutable; use a new tag for replay:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split dev --tag stage1_dropin_dev_v2_replay
```

## 2026-11-15 continuation: decoupled background core and recent-prior budget (dev_v3)

Applied the three requested changes and ran the exact consumed-dev command `--split dev --tag stage1_dropin_dev_v3` across the same 19 episodes and exact source PTS. This continuation did not change tracker lifecycle, confirmation, coasting horizon, calibrated covariance or downstream stages.

### Implemented rules

- Tightened local super-Gaussian bbox cutoff from 2.5σ to **1.8σ**, retaining `exp(-(d_M²)²/32)` and ROI-only evaluation. The theoretical unrounded, unclipped bbox area ratio is (1.8/2.5)²=0.5184; actual rounded/clipped ROI counts remain measured rather than assumed to have that exact reduction.
- Background RANSAC now samples a **32×32 lattice**, over normalized margins 0.03–0.97, and excludes only composed prior density **>0.92**. It uses actual sampled pixel centers consistently in fit and evaluation. If fewer than 16 lattice samples survive, it tries at most 256 non-core image pixels; foreground-inclusive and median fallbacks remain explicitly unsupported diagnostic modes. `bg_lattice` and `bg_sparse_pixels` are accepted sampling modes, with candidate, sample and inlier counts logged.
- Kept the background flow in the return contract so image/residual velocity conversions stay consistent; the pasted three-field return was not substituted for the existing four-field camera-aware contract.
- Raised novelty binarization from 0.5× to **0.85×v_min**, before morphological closing, and added **saliency >=1.5e-5**. Existing raw-motion, support-boundary and post-closing explained-density gates remain. This threshold is not asserted to eliminate arbitrary parallax or preserve all true object motion.
- `Pipeline.last_observed_ts` updates only for actual associated cloud measurements, including births, and is not refreshed by prediction/coasting. Prior candidates are ranked by most recent association timestamp, with observed-status and stable track-ID tie breaks; **at most 25** are rendered. This bounds only prior computation. All provisional tracks remain in the existing tracker; no unrequested eviction or identity validation was introduced.

### Mathematical qualifications

The literal threshold **0.92** corresponds to d_M≈1.278 under the super-Gaussian, whereas exactly 1.25σ has density≈0.9265. The code follows the requested numerical threshold and does not call it an exact 1.25σ boundary. At radial d_M=1.8, density≈0.720; this lower bound does not hold at every rectangular ROI corner, especially with correlated scatter. Neither contour is a certified physical boundary.

The previous broad-density mask did remove background lattice support in the two focus clips, but not every pixel in a rectangular 2.5σ bbox universally has density>0.25. These distinctions are covered in the background/core and dense-local regressions.

The pre-existing raw-motion gate is already `speed > v_min`; therefore changing the subsequent novelty floor from 0.5 to 0.85 mainly affects motion attenuated by partial explanation. It cannot, by itself, prove that all previously reported untracked 0.3–0.5 pixel residuals were the cause of proposal flooding. The saliency gate also rejects some low-energy/small true regions in principle; no independent object-recall guarantee is claimed.

### Dev_v3 outcomes

Before decoding, the protocol locked code/checkpoint/split hashes, thresholds, ranking budget and exact source snapshots. The source-separated 160/320 paired run retained shared RGB decode, alternating order, native-frame heartbeat, actual proposal unions, manual references and the sampled 18 GiB process-tree guard. The new accepted background modes are distinguished from unsupported fallbacks in benchmark counters.

| Requested dev check | Result | Outcome |
| --- | --- | --- |
| Pooled median preprocessing <=1 ms/frame | **0.410 ms/frame | PASS |
| Pooled mean full warm ingestion <=12 ms/frame | **16.05 ms/frame | FAIL |
| Each focus clip >=5/29 novel pairs, <=10% at cap, median <=5 proposals | All recover; one caps 3/29 pairs (10.34%) | FAIL |
| Unit suite + tight-background-core regression | **31 tests pass** | PASS |

| Focus episode | Accepted novel-flux pairs | Pairs at ten-proposal cap | Median novel proposals |
| --- | --- | --- | --- |
| `dis_2d54d66bfd929751b1376e8e26f28578` | 14/29 | 3/29 (10.34%) | 0.0 |
| `dis_e56c2c85b7a2500492f8d63f7c916816` | 27/29 | 0/29 (0.00%) | 2.0 |
| `dis_e5ffcb291dc874d2a6220cc7cc48ad72` | 29/29 | 2/29 (6.90%) | 4.0 |

All **551/551 post-initialization pairs** used `bg_lattice`; no `all_pixels_fallback`, `median_fallback` or sparse-pixel fallback was needed in this run. Every pair had at least **300 eligible non-core lattice samples**. These are model-selected non-core samples, **not independently verified clean background points**. Novel flux appears in 19/19 episodes, recovering the dev_v2 observation-yield loss. No more than 25 priors were rendered on any frame.

The sustained cap limit is at most two pairs out of 29: **3/29=10.3448% fails**, even though it is close to 10%. It is not rounded down for promotion. The two other focus clips satisfy both cap and median limits. Background support recovery does not establish that extracted regions are correctly identified physical objects.

Mean prior-render phase time was **2.58 ms/frame**, including prediction and validation, versus 4.03 ms in dev_v2; the proposed 0.6–0.9 ms estimate was not achieved. Pooled full ingestion improved only from 16.50 to **16.05 ms/frame**, while recovering observations that were previously rejected; this is not an equal-work speed ablation. Per-episode ingestion means range 11.99–21.55 ms/frame. Six episodes still reached the existing 100-provisional-track cap; top-25 prior selection does not reduce all tracker/tube work.

Scale micro-IoU is defined in all 19 episodes and ranges 0.1151–0.6403; no scale-qualification claim is made. Labeled-pixel reference coverage is **51.06%** over 44 exact manual-mask instances, a segmentation-coverage diagnostic rather than identity/target-tracking accuracy. Peak sampled combined process-tree RSS was **3.79 GiB**, below the 18 GiB gate. Initial heartbeat bursts remain included in these short-episode warm timings; they are not a long-running amortization measurement.

### Disposition and verification

**Two of four checks pass. No new code freeze, participant acquisition, fresh holdout selection, `stage1_holdout_v4_split.json` or general Stage 2 work occurred.** Background support and focus-clip yield recover, but latency and the strict sustained-cap criterion remain unmet. Further remediation stays on consumed data; existing P14/P26 results remain historical consumed evidence.

All **31 FMR tests passed** (18 existing TorchScript deprecation warnings). New checks verify that the suppression annulus can remain eligible for background fitting, that sparse sampling stays outside the core with bounded sample count, and that recent-observation ranking caps rendering at 25 without deleting tracks or refreshing timestamps during coasting. Existing local-prior/color, cached-DIS, image/camera/dt birth, belief, contract, tube, ledger and real-codec tests remain passing. Current source/protocol/checkpoint/mask hashes were verified and all 19 reported spatial IoUs were recomputed from saved masks. `git diff --check` passed; frozen encoder sources were untouched.

Artifacts: `stage1_dropin_dev_v3_unit_checks.json` and `stage1_dropin_dev_v3/{protocol,progress,result,preholdout_checks}.json`, plus per-episode masks and exact evaluated source snapshots. The requested tag is now immutable. For consumed-data replay use a new tag:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split dev --tag stage1_dropin_dev_v3_replay
```

## 2026-11-15 continuation: cached inverses and bounded working set (dev_v4)

Implemented all four requested changes and ran `01_bench_grid_invariance.py --split dev --tag stage1_dropin_dev_v4` on the same consumed 19-episode `stage1_dev_pool_v3`. No fresh participants or P14/P26 holdout data were opened for this successor.

### Code changes

- Extractor defaults now use **min_area_frac=0.00045**, **min_saliency=2.5e-5**, and **max_proposals=8**, retaining the 32×32 background lattice, 0.92 core threshold and 1.8σ suppression footprint. At 160×160 the integer area requirement is ceil(11.52)=**12 pixels**; at 320 it is 47. These filters are not claimed to preserve all real hand/object motion without independent recall evidence.
- `BeliefTracker.inverse_shape_scatter(track_id)` caches the regularized float32 inverse keyed to the immutable shape values. Births initialize the cache; changed shape values invalidate it before use. Prior rendering accepts that cached inverse, avoiding per-frame matrix inversion. The prior budget is now **15**, still ranked by actual last-associated timestamp with deterministic tie breaks. Shape validation remains; no calibrated covariance inference is implied by an inverse cache.
- The default live tracker cap is **40**. Observation and consecutive-miss counters are stored per track. A single-observation track is evicted after its third consecutive missed frame; a subsequent association resets misses. Multi-hit tracks are not expired by this guard. Expired single-hit tracks free slots before new births in the same update, retain monotonic non-reused IDs, and release signature/count/inverse-cache state. Pipeline closes their active tube builders and removes live event/prior-timestamp state. Existing emitted segments retain their epistemic tags; no coasting frame becomes observed coverage. This is the explicitly requested working-set safeguard, **not full Stage 2 confirmation, re-ID or coasting qualification**.
- Best-view selection now checks the cloud's scalar visibility quality **before native slicing/resizing**. Pipeline and tube builder share the same causal epoch-boundary decision, so the first crop in a new visibility/innovation/duration epoch is retained even if its quality equals the previous epoch's best. Within an epoch equal/worse quality skips pixel work. Current real-frame quality remains the existing unknown marker, so these timings do not establish independent visibility-quality estimation.

The forward-facing crop gate and clock now share `boundary_reason`; without this correction, a literal old-best comparison would skip valid equal-quality crops at a newly opened epoch. Inverse-cache reuse/invalidation and pipeline cleanup are independently regression-tested.

### Locked dev_v4 results

Before decoding, the run locked the new source/checkpoint/split hashes and exact code snapshot. Paired source RGB, alternating 160/320 order, source-separated telemetry, saved masks, manual reference checks and the 50 ms sampled 18 GiB process-tree guard were retained. The benchmark records evictions, crop counts, actual live/ROI budgets and the **configured proposal cap**.

**Cap saturation is measured at eight proposals, not the historical ten.** Reusing a ten-proposal saturation test after reducing the maximum to eight would make the metric trivially zero and is not done. Counts at different caps are not directly interchangeable measures of speckle quality.

| Requested pre-holdout check | Result | Outcome |
| --- | --- | --- |
| Pooled preprocessing median <=1 ms/frame | **0.411 ms/frame | PASS |
| Pooled full warm ingestion mean <=12 ms/frame | **11.97 ms/frame | PASS |
| Every focus clip >=5/29 recovered pairs, <=10% at configured cap, median <=5 proposals | All recover, but two exceed cap-fraction limit | FAIL |
| Unit suite + inverse/prior/pruning/crop regressions | **35 tests pass** | PASS |

| Focus episode | Accepted novel-flux pairs | Pairs at eight-proposal cap | Median novel proposals |
| --- | --- | --- | --- |
| `dis_2d54d66bfd929751b1376e8e26f28578` | 14/29 | 5/29 (17.24%) | 0.0 |
| `dis_e56c2c85b7a2500492f8d63f7c916816` | 25/29 | 0/29 (0.00%) | 3.0 |
| `dis_e5ffcb291dc874d2a6220cc7cc48ad72` | 29/29 | 4/29 (13.79%) | 4.0 |

All focus clips meet the recovery and median requirements. The latter two capped clips exceed the strict at-most-two-pairs requirement: **4/29=13.79%** and **5/29=17.24%**. Lowering the proposal maximum and increasing the area/saliency floors did not make sustained saturation pass; no frame was dropped or cap rule weakened to qualify.

Across the pool, novel flux appears in **19/19 episodes** and all **551/551** post-initialization pairs use `bg_lattice`, with no unsupported fallback accepted. Scale micro-IoU ranges 0.1817–0.6598; this diagnostic does not authorize promotion under the user-revised qualification protocol. Source manual-mask pixel coverage is **51.18% over 44 instances**, not independent object identity, fixed-ID observed coverage or NEES calibration.

The live tracker never exceeded **40 tracks**, and no frame rendered more than **15 priors**. The guard evicted 1474 single-hit working-set tracks across the 19 episodes; these are provisional IDs, not 1474 distinct physical objects. There were 2594 native best-view crop operations. Mean prior-rendering time was **1.46 ms/frame**, down from 2.58 in dev_v3 but above the projected 0.85 estimate. Per-episode full ingestion means range 9.02–16.12 ms/frame. The pooled 11.97 result passes this single locked dev measurement narrowly; it does not establish long-running tail-latency guarantees. Initial scheduled heartbeat bursts are included, while model warmup and shared source decode remain excluded. Peak sampled combined process-tree RSS was **3.99 GiB**, below 18 GiB.

### Conditional execution and gate disposition

**Three of four dev checks pass.** The failed sustained-cap criterion prevents the requested transition to candidate freeze, fresh `stage1_holdout_v4` acquisition/selection/evaluation, and general Stage 2. No new freeze, data acquisition or v4 split was created. A holdout-mode guard check was executed and correctly rejected entry **before** creating a report or opening fresh data with `Fresh holdout remains closed until all dev_v4 checks pass`.

The benchmark now contains the user-requested conditional v4 holdout path: it requires an all-pass dev_v4 report, unchanged source hashes, a fresh participant-disjoint eight-episode split locked after candidate freeze, and the fixed parameters. If that stage becomes authorized by actual passing dev evidence, its primary gate is pooled warm ingestion <=15 ms, nonempty novel flux in >=85% of action episodes and sampled process-tree RSS <=18 GiB. Scale IoU remains diagnostic under the latest user plan. No such qualification is claimed in this run.

Further Stage 2 work remains gated; the limited working-set counters/pruning above were explicitly requested for dev_v4 and must not be labeled as identity confirmation, flow-advection acceptance or independent covariance calibration. Further remediation must use consumed data; reducing a cap alone cannot demonstrate reduced false proposals or preserved true-object recall.

### Verification and reproduction

All **35 FMR tests passed** (18 existing TorchScript deprecation warnings). New checks verify cached and uncached priors agree, inverses reuse unchanged shapes and invalidate changed ones, three misses prune only single-hit tracks and free slots, multi-hit tracks survive this limited guard, inverse/event/builder/timestamp state is cleaned on eviction, and native crops skip equal quality while new epochs still retain their first patch. Existing core/ROI/Lab, exact-time/camera/birth, tracker, tube, ledger and real-codec checks remain passing.

Current protocol/source/mask hashes, all 19 saved-mask IoUs, and the observed 40-track/15-prior/eight-proposal bounds were verified; `git diff --check` passed. The encoder baseline was untouched. No independent identity, contact, covariance-calibration or downstream promotion result is implied.

Artifacts: `stage1_dropin_dev_v4_unit_checks.json` and `stage1_dropin_dev_v4/{protocol,progress,result,preholdout_checks}.json`, with exact evaluated source snapshots and per-episode masks. The requested tag is immutable; use a new tag for consumed-data replay:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/01_bench_grid_invariance.py --split dev --tag stage1_dropin_dev_v4_replay
```


## Stage 1 v5 qualification and Stage 2 checkpoint (2026-11-15)

Stage 1 v5 adds the ego-motion speed floor, outer-rim rejection and adjacent co-moving fragment merge before NMS; the proposal cap is restored to ten. The original `stage1_dropin_dev_v5` run passed three checks but missed pooled latency at 12.382 ms/frame. Reusing the identical current-frame belief predictions removed duplicate computation. `stage1_dropin_dev_v5_cached` passed all four checks: median preprocessing 0.429 ms, pooled ingestion 11.630 ms, focus-clip discovery 24/29, 28/29 and 14/29, zero cap hits, and proposal medians 1, 3 and 0. All 19 saved proposal-mask arrays were bit-identical between these two runs.

`stage1_v5_code_freeze.json` precedes fresh frame selection and decoding. `stage1_holdout_v4_split.json` locks eight episodes from previously unused participants P07/P15, excluding all thirteen previously consumed participants. Acquisitions use a pinned public mirror revision and publisher MD5 plus SHA-256 checks, recorded in the split. Original dataset videos and the frozen encoder baseline were preserved. P07_18 publisher MD5 is `d2d3c1372623a12e6909177a0a26968c`; P15_01 is `1fee4c8a49abe2038a5ee8b9bfd82c59`.

The fresh holdout passed the requested engineering qualification: pooled ingestion **10.396 ms/frame**, **8/8** episodes with novel flux and **1.591 GiB** sampled process-tree peak RSS. The 50 ms sampler is not an OS-enforced memory cap. Scale IoU remains diagnostic under the revised gate (only one episode reaches 0.65); historical `local_gate_passed` fields retain the older IoU condition. See `stage1_holdout_v4/qualification_assessment.json`. Frozen source snapshots were reverified after Stage 2 changed current sources. P07/P15 are now consumed evaluation participants.

Stage 2 uses Hungarian assignment with Mahalanobis, approximate ellipse IoU and Lab appearance costs. DIS image-flow advection supplies predicted position/velocity with propagated uncertainty. Flow-only states remain PREDICTED; only actual assigned clouds receive OBSERVED status. Joseph-form conditioning reuses the computed prediction. Measurement-floor and process-noise controls are exposed but remain uncalibrated (defaults 0 and 0.001). Ellipse overlap and appearance costs are engineering approximations, not certified geometry or calibrated identity likelihoods. Confirmed lifecycle/re-identification remains incomplete. All **42 FMR tests pass**, with 18 existing TorchScript deprecation warnings.

`stage2_cohort_v1/result.json` covers all 20 historical cohort episodes: **6,078 decoded frames**, including 20 initialization frames and **6,058 strictly post-seed frames**. Of those, 5,985 have eligible source-centroid annotations. The diagnostic initializes at most one nearby OBSERVED provisional ID per episode and never switches IDs; it initializes in 12/20 episodes. Observed-only success within 50 native pixels is **24/5,985 (0.401%)**: 18 and 6 successes in two episodes, zero in the other eighteen. Predicted support earns no observed coverage. Source centroids are a proxy and do not establish physical identity. Sampled peak RSS is **1.742 GiB**; elapsed audit time is 171.6 seconds. This result does not support Stage 2 promotion.

`stage2_cohort_v1/verification.json` records recomputed coverage, matching current/frozen source and output hashes, 22,603 finite observation records and verified tags for 1,630 emitted OBSERVED segments. Per-frame diagnostics, observation/posterior arrays and tube ledgers are retained per episode. Segment counts are not frame coverage.

The reference search found four 3,000-row anchor templates under `reports/track1_remediation/independent_review/` and `reports/track1_readiness/independent_review/`, each with **zero** completed position/identity/reviewer rows; `references/track1/adjudicated/` has no usable adjudicated records. Existing independent-evaluation reports also indicate no usable labels. See `stage2_cohort_v1/reference_readiness.json` for exact paths and hashes. Consequently NEES, 95% joint coverage and independent IDF1 remain null. Innovation likelihood/NIS and source centroids cannot substitute for independently adjudicated same-target position/identity truth. The next calibration step requires those labels and an explicit binding to model state/track definitions; no covariance tuning or scientific acceptance is claimed.

Reproduce the current Stage 2 diagnostic into a new immutable tag:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02_eval_cohort_continuity.py --tag stage2_cohort_replay
```

Stage 1 result reproduction requires its frozen source snapshot, since current Stage 2 assignment/advection differs. Replaying consumed episodes does not create fresh qualification evidence.


## Stage 2 stationary lock, archive, visual re-ID and model review (2026-11-15)

The revision adds a 15×15 grayscale NCC verifier with a 1.25-sigma shape-core residual-speed check. Only confirmed tracks can receive a template-verified observation; flat patches, low NCC, high residual flow and border-truncated patches reject. The template remains anchored to a proposal observation, rather than refreshing itself on every lock. Template-verified measurements are recorded separately from proposal and re-ID measurements. Unverified confirmed tracks coast with damped velocity and expanding covariance. Tentative tracks require three observed hits and expire after three misses; confirmed tracks leave the active set after one second of lost support or positional uncertainty above the prescribed bound. A dormant archive holds at most 64 entities for ten seconds since their last observation. Active capacity remains 40.

Dormant re-ID is a second Hungarian pass after active association, using the time-expanded radius, Fourier shape magnitudes and Lab moments. The supplied snippet's signature indices did not match this repository: implemented Fourier indices are `0:4`, Lab moments `4:10`. Spatial reach is measured from the last observed cloud centroid, not a potentially drifted coasting position. Re-ID resurrects the original provisional ID. Competing candidates use actual local `work/yoloe-26s-seg.pt` SAVPE embeddings and cosine separation; unresolved comparisons abstain. The encoder runs on CPU only when competition occurs and caches archived reference embeddings. Reference crops/embeddings are bounded by active plus dormant entity capacity. This is an uncalibrated matching heuristic, not verified physical identity.

The reference-only query binder accepts either the 50-pixel centroid gate or a 2-sigma shape ellipse, breaking ties with shape Mahalanobis distance. An ellipse is approximate support, not the original segmentation mask; a broad region can admit an incorrect semantic binding. Inference never receives source centroids or external mask polygons.

| Cohort diagnostic | Initialization | Strict observed-only centroid proxy | Mean ingestion | Sampled tree RSS |
| --- | --- | --- | --- | --- |
| `stage2_cohort_v1` historical baseline | 12/20 | 24/5,985 (0.401%) | not recorded | 1.742 GiB |
| `stage2_cohort_v2` NCC 0.72, archive | 20/20 | 26/5,985 (0.434%) | 17.360 ms | 1.956 GiB |
| `stage2_cohort_v3_ncc65` NCC 0.65, corrected re-ID spatial origin | 20/20 | 244/5,985 (4.077%) | 21.559 ms | 1.704 GiB |
| `stage2_cohort_v4_savpe` final visual re-ID, NCC 0.65 | 20/20 | 244/5,985 (4.077%) | 27.407 ms | 1.825 GiB |

All cohorts contain 6,078 decoded frames, of which 6,058 are strictly post-initialization; 5,985 have eligible source annotations. Predictions earn zero observed credit. The final run's 244 successes comprise 57 proposal, 185 template-verified and 2 re-ID observations. It produced 83,401 finite observation records, 64,884 template-verification updates across all tracks, 310 resurrections and 1,121 real SAVPE encoder calls. Extra observations therefore do not imply correct target continuity. **The 65% coverage and 13.5 ms ingestion gates fail.** The last two timing runs overlapped the external LM Studio reviewer; these are recorded engineering measurements, not isolated performance qualification. RSS monitors cover the ingestion process tree, excluding the separately running LM Studio server. The 18 GiB guard is sampled, not OS-enforced.

The final 6,434 emitted OBSERVED tube segments cover 910 distinct provisional IDs. All-track segments average 321.7 per episode; fixed selected IDs account for 53 segments (2.65 per episode), but low coverage makes that low number uninformative as a success metric. Event/fit/status boundaries also split tubes. The plan's original inference that 1,630 segments implied 1,630 ID deaths was not established: the historical run's segments contain 459 distinct emitted IDs. Neither count supplies adjudicated physical fragmentation truth.

`stage2_cohort_v4_savpe/verification.json` recomputes scores and validates all frame, observation, tube, source snapshot and checkpoint hashes. Current final ingestion sources match its frozen snapshot; the Stage 1 extractor hash is unchanged. The frozen encoder baseline has no source diff. All 46 FMR tests pass, with 18 existing TorchScript deprecation warnings. Tests cover flat/changed-template rejection, stationary locks, tentative expiry, coasting/dormancy, original-ID resurrection, ambiguous-candidate abstention, visual-similarity resolution, containment binding and finite conditional calibration replay.

`stage2_visor_v2_savpe` evaluates two multi-timestamp externally annotated `manual_sparse` right-hand sequences, with P02 development and P08 validation participants disjoint. Both participants were historically consumed, so this is not a fresh holdout. The fixed-ID ellipse has zero observed support at both subsequent manual timestamps; P02 fails initial binding and P08 initially covers 95.14% of the manual mask then loses observed support. Centroid error at P08 initialization is 4.33 pixels. No polygons enter inference. This small sparse diagnostic is not a scene segmentation/identity benchmark. Source: https://epic-kitchens.github.io/VISOR/site and https://github.com/epic-kitchens/VISOR-VIS .

Per the user's explicit replacement of manual review with local model review, `02b_review_and_calibrate.py` prepares and executes LM Studio Qwen3.8 review on 300 hash-locked native timestamps: 150 each from P08/P18 episodes. Images are resized only for the reviewer; source RGB and datasets remain unchanged. The model sees images and target descriptions, without tracker predictions, point ledgers or mask polygons. Four-image requests use explicit indices, native `/api/v1/chat`, reasoning off, deterministic temperature and no persistent server chat. Valid prior responses are resumed; invalid responses are retained and retried. Raw output, model identifier, image/manifest hashes, request settings and reviewer code hashes are retained per frame. API source: https://github.com/lmstudio-ai/docs/blob/main/1_developer/2_rest/chat.md .

A four-frame external check (`stage2_vlm_visor_check_v1/external_accuracy.json`) found centroid differences of 6.24, 34.00, 119.50 and 447.55 native pixels. The model prompt requests an anatomical hand centroid excluding forearm; some external hand masks include a different visible region, with gloves also separately labelled. This reveals a coordinate/region-definition mismatch as well as possible model error; confidence alone does not verify reference accuracy. Model labels are exported under `references/track1/model_reviewed/`, preserving `human_adjudicated=false`, rather than being inserted into the existing incompatible physical-landmark adjudication workflow.

The covariance script `02c_calibrate_and_eval_nees.py` fits process noise and a measurement floor by innovation likelihood on development participants excluding P08/P18, with recorded association IDs held fixed. It then computes a model-reference discrepancy diagnostic on the locked review timestamps. This is an offline centroid filter without DIS replay; it does not replace live tracker parameters. Because reference and measurement support definitions are not demonstrably equal, any silver NEES is labelled a mismatch diagnostic. Genuine adjudicated NEES, scene IDF1 and scientific promotion remain unavailable. Participant separation in this fit does not erase earlier development exposure to these cohort participants.

Reproduce into new tags (runs refuse overwrite):

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02_eval_cohort_continuity.py --tag stage2_continuity_replay --ncc-threshold .65
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02d_verify_stage2_artifacts.py stage2_cohort_v4_savpe
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02b_review_and_calibrate.py
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02e_export_model_references.py
```


### Completed model-review and calibration checkpoint

`stage2_vlm_review_v1/result.json` records **300/300** completed, schema-valid Qwen reviews; **298** have confidence at least 0.8. All 300 report the target as visible. Confidence is a model assertion, not adjudicated accuracy. `references/track1/model_reviewed/stage2_qwen38_v1/labels.json` and its manifest retain all rows, model attribution and per-response SHA-256 provenance; low-confidence rows are not silently removed. `stage2_vlm_review_v1/verification.json` verifies all review images, manifest, response and export hashes. The earlier truncated/invalid responses remain under `invalid_attempts/`.

Development-only fixed-association fitting converged after 51 objective evaluations: acceleration spectral density **0.5** (the configured upper bound) and normalized observation-floor standard deviation **0.0039012522**. A boundary optimum is not independent calibration evidence. Live defaults remain unchanged. `stage2_silver_calibration_v1` preserves the centroid-only evaluation, which has zero bound/evaluable target frames. `stage2_silver_calibration_v2_support` reuses the same dev fit only after verifying identical observation hashes and participant exclusion, adds the requested 2-sigma birth-shape containment gate, and still cannot bind either target at its first reviewed timestamp. Both bound IDs are null; **0/298** high-confidence frames are evaluable under the fixed-ID policy. No later favorable ID is substituted to manufacture a score. The support shapes are read from immutable per-ID tube scatter, checked for equality across segments and hash-bound in the protocol.

Accordingly model-reference NEES and joint coverage remain null, as do independent identity IDF1 and human-adjudicated metrics. This is now an explicit reference-to-track binding/support-definition failure, rather than an assertion that no automated labels exist. The model-reviewed points do not by themselves establish that a proposal-region centroid is the same physical state. The final implementation, reviewed labels and failed gates are retained for the next diagnostic; no scientific promotion or downstream Stage 3 opening occurred.

Reproduce the support-bound offline diagnostic without refitting on model references:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py --cohort reports/track1_fmr/stage2_cohort_v4_savpe --tag stage2_silver_support_replay --reuse-fit reports/track1_fmr/stage2_silver_calibration_v1/dev_fit.json
```


## Stage 2 v5 masked verification, containment and isolated validation (2026-11-16)

The implementation follows the plan's stricter prose where the drop-in snippet differed: foreground intersections require **at least nine pixels**, with **no fallback to an all-ones mask**, and a child must have **less than 60%** of its parent's area. Adaptive patch half-widths are 3–6 grid pixels (7×7 to 13×13 patches). Genuine proposal masks intersect the 1.25-sigma shape core when reference patches are captured. Current correlation and residual-speed checks use only the retained foreground intersection. NCC must reach 0.78; residual core speed must stay at or below 1.20 times the effective discovery floor. Templates refresh on genuine proposal/re-ID observations and remain frozen across template-verification updates. A shape core is still an approximation to physical support; a successful NCC is not independently verified identity.

Confirmed active tracks and dormant entities can associate with a smaller contained proposal using the 2.25-sigma gate, area test and Lab-moment threshold 0.45. When the containment branch wins an assignment, the same provisional ID is retained and position/shape adapt halfway toward the child. The prescribed widened gate (6.25) is a separately retained development variant. Normal association and covariance remain distinct from the shape-containment gate.

Only entities with at least five observed updates enter the dormant archive. Short confirmed entities that exceed their coasting limit are evicted rather than archived. SAVPE is considered only for the two lowest algebraic candidate costs when they differ by at most 0.12. Missing reference embeddings and the current proposal are encoded in **one batched network forward**, and a shared ten-native-frame cooldown applies across all candidate comparisons. The final run made 69 physical forwards, versus 1,121 in the previous candidate; `verification.json` checks that every forward is separated by at least ten frames and that no frame contains multiple forwards. Batch size means that the forward count is not the number of embedded crops; no unmeasured fixed milliseconds-per-call assumption is asserted.

LM Studio's model was unloaded, the local server stopped and no review/calibration worker remained before timing. The cohort driver forces the heartbeat to CPU and checks that the server port is closed before execution and every hundred frames, outside the measured ingestion interval. All timing candidates ran sequentially. The original source videos, frozen encoder implementation and Stage 1 v5 extractor were preserved.

| Consumed 20-episode run | Containment gate | Observed-only proxy | Mean ingestion | Sampled peak tree RSS |
| --- | --- | --- | --- | --- |
| `stage2_cohort_v5` | 5.06 | 97/5,985 (1.621%) | 17.174 ms | 2.998 GiB |
| `stage2_cohort_v5_wide` | 6.25 | 103/5,985 (1.721%) | 16.694 ms | 3.048 GiB |
| `stage2_cohort_v5_cached` | 6.25 | 103/5,985 (1.721%) | 16.153 ms | 3.124 GiB |
| `stage2_cohort_v5_fast` | 6.25 | 103/5,985 (1.721%) | 15.013 ms | 3.086 GiB |

All runs initialize **20/20** episodes. Coverage remains below 65%, and the final **15.013 ms/frame** exceeds 13.5 ms/frame. RSS passes the sampled 18 GiB process-tree check. The final timing optimizations batch position solves and Lab distances, cache a parent inverse per association row, retain an equivalent elementwise covariance symmetry test and skip polygon intersection only for disjoint polygon bounds. **All observation times, IDs, means, covariances, support arrays and counts are bit-identical across the 20 episodes of the widened, cached and final fast runs.** These optimizations do not establish improved tracking accuracy.

The final 103 proxy successes comprise 49 proposal observations and 54 template-verified observations. Its 63,898 finite observation records emit 7,831 OBSERVED tube segments across 791 provisional IDs; tube segmentation is not physical identity fragmentation. `stage2_cohort_v5_fast/verification.json` validates source/output hashes, frozen extractor equality, observed-only scoring and the actual SAVPE budget. `equivalence.json` records bit-identical outputs, and `initial_id_lifecycle.json` retains per-episode lifecycle evidence.

**A remaining mechanical limitation:** 17/20 initial fixed IDs never reach three observed updates, so they cannot invoke the confirmed-only stationary verifier or enter the five-hit archive. The extractor suppresses velocity-consistent explained motion from novel proposals (`novel_binary[explained > 0.65] = 0`); flow advection alone remains PREDICTED, and stationary verification rejects high residual speed. This combination can leave an existing moving region without an observation-producing continuation path. The telemetry establishes the confirmation shortfall; it does not independently identify the physical target or prove one exclusive cause. The proposed background-lock/part-splitting diagnoses remain hypotheses rather than adjudicated root-cause measurements.

### Expanded external-mask diagnostic

`stage2_visor_v3` locks P02 development, P08 validation and additional P06/P21 multi-timestamp sequences. The additional sequences are reserved for this diagnostic before execution; their participants were historically consumed elsewhere, so they are not fresh participant holdouts. Initialization covers 95.14%, 94.02% and 93.31% of the P08, P06 and P21 masks, respectively; P02 does not bind. **Every later annotated timestamp has zero observed support.** The prescribed adaptation-rate retry at 0.75 (`stage2_visor_v3_adapt75`) has the same outcome. Both fail the 50% support gate. Human mask polygons are reference-only and never enter inference; ellipse support is not a predicted segmentation mask.

### Support-consistent model review and confirmation-window calibration

After all timing and VISOR runs finished, local `qwen/qwen3.8-27b` was restored for a new 300-frame review, reusing the hash-locked P08/P18 timestamps and source images. The new prompt requests the visible foreground hand/manipulator, including attached visible forearm and any worn glove, plus a tight normalized bounding box and estimated area centroid. It explicitly excludes the anatomical wrist-cutout definition. These are model estimates, not actual pixel masks or human adjudication; every output retains `human_adjudicated=false`.

The parser validates both JSON arrays and wrapper objects, finite normalized geometry, nonempty boxes, centroid containment, unique frame indices and complete batch membership. Valid previously saved raw arrays were recovered without another model call; original rejected records remain preserved. Export verification checks normalized values against the raw model response, along with image, task, manifest and response hashes. Review outputs use `stage2_vlm_review_v2_support`; old labels are preserved.

The v5 cohort NPZs add causal per-observation scatter, ellipse bounding boxes and observed-hit counts, replacing the invalid assumption that shape remains constant after part adaptation. Calibration allows the first five visible reviewed timestamps, requires a confirmed observation, and binds through 2.5-sigma containment or box IoU at least 0.20. The first eligible ID remains fixed; there is no favorable post-window reinitialization. The stored boxes bound the current 2.5-sigma ellipse, rather than a retained segmentation polygon.

Development fitting excludes P08/P18 and uses fixed recorded associations. The widened process-noise interval is `[0.0001, 5.0]`, with a numerical tolerance in the interior-bound check. The development-only fit converged after 72 evaluations at **q_acc = 2.4050868**, strictly interior, and observation-floor standard deviation **0.00979414**. `stage2_silver_calibration_v3_devfit` retains that fit; the final evaluator reuses it only after verifying identical observation hashes and participant exclusion. Evaluation waits for all 300 schema-valid reviews before hashing the full reference set. No live tracker parameter is replaced and no holdout discrepancy is used to refit the dev parameters.

The support-consistent review and final silver metrics are still in progress at this checkpoint. Stage 2 remains unpromoted; no downstream stage is opened.


## Observation–Identity–Carrier repair and audit reconciliation (2026-11-16)

The supplied comprehensive audit matches the frozen spline and manually initialized `track1_belief` experiments. Their intentional point-death and origin-relative similarity assumptions remain preserved. Local commit `df755e39766427847c4c88f10acd0e80ae3f113a` has 16 commits in its history and already contains experimental FMR detector births, cloud measures, coasting/dormancy and re-ID. Their existence does not establish production readiness. `observation_identity_v1/audit_reconciliation.json` pins the supplied audit and distinguishes these systems. The remote GitHub HEAD was not independently retrieved.

The concrete remaining FMR mechanism is that velocity-consistent tracked motion is suppressed from novel discovery, while flow advection alone produces no OBSERVED update and the stationary verifier requires prior confirmation. Seventeen of twenty initial fixed IDs in the v5 cohort never reached three observed updates. The new path provides direct detector evidence every frame, independent of the novelty gate, to test this observation/confirmation bottleneck rather than add richer geometry.

`src/track1_fmr/observation_identity.py` separates detector observations from association and carrier state. A valid detector box can create a carrier when segmentation is missing or invalid. Box moments are analytic; mask moments describe current visible pixels. State uncertainty and shape scatter remain separate, and observed shape is refreshed instead of enforcing global coherence with initialization. No fine spline or boundary certificate is needed for an entity to exist. Detector confidence is not treated as calibrated identity confidence. Partial visibility is asserted only through explicit supplied evidence, rather than guessed from mask size.

Association reuses the bounded belief tracker with explicit motion/shape, appearance/re-ID and containment switches. Every detector observation is genuine current-frame evidence; optional short-term LK uses fresh features from the previous detector support and forward/backward verification, supplies prediction only, and does not become the identity authority. A >100 ms gap does not permanently kill the entity or block detector recovery. Coasting and dormant memory remain bounded. Re-detection can restore the same provisional ID, but re-ID scores are not calibrated physical-identity probabilities. The old FMR pipeline remains separately available and is included as a baseline.

### Locked evaluation and metrics

`03_eval_observation_identity.py` locks code snapshots, both local checkpoints, the publisher-checksummed public point dataset, the exact external-mask selection and native decoder dependencies before evaluation. No query point or manual mask enters detector/region inference. At a query's first visible timestamp, the evaluator binds the smallest observed support containing that point, fixes the ID and initial point offset, and never reinitializes to a later favorable ID. All eligible unmatched queries remain failures. A fixed point-to-region offset is a **translation proxy**: articulation within a correctly tracked region can still produce point error. It is not a complete object-MOT identity benchmark or a controlled model-only comparison against a query-seeded point tracker.

The independent public point run covers **30 videos, 1,999 frames and 650 queries**, with **28,174 visible point-times strictly after the query timestamps**. Observed coverage and 8-pixel agreement use that same visible denominator. Carrier coverage uses all post-query point-times, including hidden timestamps; its denominator therefore differs from observed coverage. Predictions/coasting never receive observed-point credit. The common-observed subset contains only **312 point-times**, which demonstrates why conditioning all comparisons on common support would hide most robustness differences.

Detector-only box and mask support contain **96.29%** and **89.29%** of visible reference points, respectively. These are point-support recall numbers, not verified object recall: a broad actor box can contain a point without identifying the correct semantic object. Detector-only outputs have frame-local IDs and receive no persistent-tracking score.

| Method | Observed visible coverage | Observed point agreement ≤8 px | Carrier coverage | Conditional observed RMSE, 256-px coordinates | Frame-weighted component ms |
| --- | --- | --- | --- | --- | --- |
| Motion + box moments | 60.00% | 24.92% | 87.59% | 33.92 | 71.35 |
| Motion + mask moments | 57.47% | 26.29% | 82.23% | 29.22 | 71.12 |
| + appearance/re-ID, box | 60.20% | 24.88% | 88.57% | 32.79 | 71.12 |
| + appearance/re-ID, mask | 57.60% | 26.31% | 82.36% | 29.52 | 71.12 |
| + short-term flow, box | 69.19% | 29.46% | 89.12% | 36.66 | 75.99 |
| + short-term flow, mask | 63.95% | 30.74% | 81.70% | 26.28 | 75.97 |
| Existing FMR v5 | 3.27% | 0.70% | 12.88% | 37.18 | 13.14 |
| Query-seeded LK (different initialization) | 49.68% | 47.32% | 38.61% | 5.31 | not measured |

The mask-plus-flow path has mean per-video observed-only TAP-Vid Jaccard **0.1650**, existing FMR **0.0060**, and query-seeded LK **0.3813**. The new path recovers correct observed point agreement within five frames on **58/471** reference occlusion returns, versus **1/471** for existing FMR and **6/471** for LK. Appearance alone adds little over motion in this trial; flow improves point agreement but does not establish physical identity. Query-seeded LK retains substantially better point accuracy and a different initialization contract. The experiments are historically consumed diagnostics, not a fresh locked holdout or evidence of a novel tracker beating the published state of the art.

Timing performs one shared detector pass per input frame and charges its measured cost to each ablation, then adds that method's synchronous processing. It includes detector segmentation and both moment summaries in the common frontend; it is **not** a bbox-only detector versus segmentation-model cost comparison. These are component estimates, not isolated deployed end-to-end latency. The combined experiment peak sampled process-tree RSS is **6.518 GiB**, including the loaded 2.3 GiB public dataset and all ablations, rather than per-method production memory. The 18 GiB guard is sampled, not an OS-enforced cap. The prior 13.5 ms real-time gate is not met.

### Independent masks and remaining limits

The four VISOR clips retain P02/P08 and the predeclared P06/P21 sequences; participants were historically consumed. At their later manual timestamps, the best current detector mask IoUs are **0.030 (P02)**, **0.657 (P08)**, **0.891 (P06)** and **0.434 (P21)**. On all eight annotated masks, mask detector recall at IoU ≥0.5 is **62.5%**, versus **37.5%** for boxes. Current-region detection therefore remains inadequate on some targets, even when point-support recall is high.

Fixed-ID mask-plus-flow support recovers **99.02%** of the later P08 manual mask, while later P02/P06/P21 support remains zero. Appearance-only mask tracking instead retains **96.45%** on P21 but not P08/P06. In particular, P06 has a strong later detector mask but no retained fixed-ID observation, exposing association/initial-binding limitations separately from detector recall. These sparse results do not satisfy reliable physical identity coverage. Moment centroid/scatter errors are retained with their observed-reference counts in `assessment.json`; unsupported references are not hidden from coverage reporting.

There are no complete persistent object-instance labels for scene IDF1/ID switches in this protocol, no independently validated contact truth, and no unprivileged frozen-spline outputs on the same public point dataset. Those fields remain null. The frozen spline baseline is not silently replaced by LK. Directional-lobe, contact-lens and Track 2/3 event-certificate ablations are deferred in the audit's prescribed order; no contact-F1, geometry superiority, calibrated covariance or scientific promotion is claimed. Coarse detector segmentation still runs every frame; zero fine-boundary lens invocations does not mean segmentation was free or on-demand.

`04_verify_observation_identity.py` verifies frozen inputs/checkpoints/source snapshots and output hashes, reloads the independent references and saved predictions, and recomputes denominators, observed counts, fixed-query localization, carrier counts and common support. All 30 videos and all unmatched queries are retained. `verification.json` and `assessment.json` contain the resulting checks and pooled metrics. The frozen `src/track1_encoder_revision`, `src/track1_belief` and Stage 1 extractor have no source diff.

Validation: **282 repository tests pass**, with 18 existing TorchScript deprecation warnings. The suite requires a project-contained temporary directory because several provenance contracts reject paths outside the repository; the initial default-temp run's three path failures disappear with that fixture configuration. New tests cover box-only detector output, explicit direct evidence, current deforming shape, gaps, coasting, re-detection, appearance ablation, non-authoritative flow and common-denominator accounting.

Reproduce with a new immutable tag:

```sh
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/03_eval_observation_identity.py --tag observation_identity_replay --videos 30
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps /opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/04_verify_observation_identity.py --tag observation_identity_v1
PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m pytest -q tests --basetemp work/pytest_observation_identity_v1
```

### Closing the preceding interrupted v5 checkpoint

The support-consistent Qwen review completed **300/300 valid records**, all meeting the 0.8 confidence threshold, and exported boxes/centroid estimates under `references/track1/model_reviewed/stage2_qwen38_v2_support/` with `human_adjudicated=false`. The five-timestamp calibration replay still bound **0/2** episodes and evaluated **0/300** frames; silver NEES and joint coverage remain null. Development-only NIS is **1.748** at interior q_acc **2.405**, with live tracker defaults unchanged. The isolated resident-model v5 run measured **13.723 ms/frame** after separately reported **2.796 s** warmup, **2.548 GiB** sampled peak RSS and unchanged **103/5,985** observed-only source-centroid successes. It remains unqualified. The optional 3× stationary-noise parameter and its unit check were present at the incoming audit's base commit; a native 3× fallback was not reported as completed. The new audit prioritizes observation/identity repair over further noise-threshold iteration.


## 2026-11-18 keyframe observation/identity revision

Implementation and evaluation are complete; qualification fails. All **292 repository tests pass**. Corrected 30-video DAVIS evaluation (1,999 frames, 650 queries, 28,174 visible point-times) yields strict/lineage coverage **64.50% / 67.91%**, point agreement ≤8 px **32.49% / 33.23%**, and reacquisition **36 / 40 of 471**, below the accuracy/recovery targets. Mean ingestion is **36.40 ms/frame DAVIS / 38.02 native**, above 13.5 ms; sampled RSS is **5.31 GiB**. Detector-only VISOR recall is **7/8 (87.5%)**, but later P21 tracking still fails. New native trajectories cover 20 episodes and 6,078 frames. Silver calibration binds 2/2 episodes but evaluates only **207/300** frames; mean NEES **468.52** and joint 95% coverage **1.45%** fail. Labels remain model-reviewed, not human-adjudicated. No Stage 2, identity/contact or geometry promotion. Frozen baselines and qualified Stage 1 are unchanged. Final report: [observation_identity_v2_keyframe_evalfix/RESULTS.md](observation_identity_v2_keyframe_evalfix/RESULTS.md); calibration: [stage2_silver_v4_keyframe/result.json](stage2_silver_v4_keyframe/result.json). Earlier sections are historical snapshots.

The original `observation_identity_v2_keyframe` point evaluation is superseded by `observation_identity_v2_keyframe_evalfix` after a delayed-query support-coordinate fix; its native measurements are unchanged and reused with source/checkpoint/input hash checks. Both integrity verifiers pass, including scalar/batched calibration equivalence and reference-free transport replay. These passes do not override failed accuracy, latency or calibration gates.


## 2026-11-18 observation/identity v3

The requested single-detector stride-eight, eroded 12-anchor mesh, patch-NCC verification, open LK covariance propagation and appearance rescue revision is implemented and fully evaluated. **299 tests pass**; both integrity verifiers pass, including native LK covariance equations and independent scalar calibration replay. **Stage 2 qualification fails; Stage 3 remains gated.** DAVIS strict/lineage coverage is **46.33% / 47.97%**, point agreement ≤8 px **25.55% / 25.34%**, and reacquisition **22 / 20 of 471**. Timing improves to **19.48 ms/frame DAVIS / 23.25 native**, still above 13.5 ms. Sampled RSS is **4.65 GiB**. Detector VISOR recall is **6/8 (75%)**; later P08 support fails. Silver calibration binds 2/2 episodes but evaluates **232/300** frames, below 265, with mean NEES **480.74** and joint 95% coverage **11.64%**. The model labels remain silver. V3 regresses tracking versus corrected v2 and is not promoted. Frozen baselines and qualified Stage 1 remain unchanged. Final report: [observation_identity_v3/RESULTS.md](observation_identity_v3/RESULTS.md); calibration: [stage2_silver_v5/result.json](stage2_silver_v5/result.json). Earlier sections are historical snapshots.

The pasted zero-innovation and duplicate-dorsal diagnoses were not supported by v2 source. V3 nevertheless tests the requested open propagation and smaller single-head frontend. It retains continuous-white-acceleration noise units, rejects unverifiable flat patches, discards failed anchors, and updates calibration replay so LK motion does not masquerade as independent measurements. The measured tracking and calibration failures prevent downstream promotion.


## 2026-11-18 observation/identity v4

The single prompt-free stride-ten/native-capped detector, three-inlier batched flow, direct query particles, anti-hijack area guard, subpart adaptation and additional silver subset diagnostics are implemented and evaluated. **307 tests pass** and both integrity verifiers pass. **Stage 2 qualification fails; Stage 3 remains gated.** Mesh strict/lineage coverage is **74.41% / 77.00%**, agreement ≤8 px **38.55% / 38.78%**, and recovery **71 / 66 of 471**. Query-seeded hybrid agreement is **36.48% / 36.63%**, with **23 / 19 of 471** recoveries, below the targets. Joint DAVIS runtime is **24.51 ms/frame**; query-free native runtime is **26.71 ms/frame**, above 13.5 ms. Sampled RSS is **4.52 GiB** and detector VISOR recall **7/8 (87.5%)**. Lineage later P08 support reaches **93.37%**, but P21 support is only **0.57%**. Silver binding is 2/2 with **290/300** evaluable states; only **50/300 (16.67%)** satisfy the in-support proxy. Full scored-sequence NEES is **6996.49**; even the reference-conditioned subset is **181.28**. Model labels remain silver; no live calibration or physical identity/contact promotion. Frozen baselines and qualified Stage 1 remain unchanged. Final report: [observation_identity_v4/RESULTS.md](observation_identity_v4/RESULTS.md); calibration: [stage2_silver_v6/result.json](stage2_silver_v6/result.json). Earlier sections are historical snapshots.

Mesh and query-seeded hybrid metrics are separated, query particles do not affect carrier association, and geometric re-seed predictions receive no observed credit until later image verification. The new reference-conditioned in-track NEES remains diagnostic and does not replace full-sequence qualification. Original runtime artifacts remain immutable; corrected float32 verifier arithmetic is archived separately under `verification_source/`.
