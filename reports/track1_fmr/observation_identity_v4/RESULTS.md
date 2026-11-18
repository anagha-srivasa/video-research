# Observation–identity v4: evaluated, not qualified

The v4 revision and all six requested commands are complete. **307 repository tests pass** with 18 existing TorchScript deprecation warnings. Both integrity verifiers pass, including independent hybrid-particle acceptance/re-seeding replay, native LK covariance equations, scalar calibration replay and the new in-support subset metrics. **Stage 2 still fails; Stage 3 remains gated.**

The implementation uses the plan's permitted **single prompt-free detector** option, with native-capped detector resolution and fixed stride ten. It does not add a second manipulator text-prompt pass. See `IMPLEMENTATION.md` for exact parameters, departures from unsafe predicted-point retry behavior, and corrections to the pasted audit.

## Point metrics: mesh control versus direct hybrid

All four methods use the same carrier states and initial observed-support binding. `keyframe_*` transports the point using carrier mesh/box geometry. `hybrid_*` additionally tracks the initial query point through the joint FB-LK batch; it is explicitly a **query-seeded hybrid**, not blind object discovery. Only its own image-verified, in-support point updates receive observed credit. Predicted fallback and detector-based re-seed hypotheses do not receive credit until a later FB-verified step.

| Method | Visible observed coverage (target ≥62%) | Agreement ≤8 px (target ≥42%) | Returns recovered (target ≥50/471) | Conditional observed RMSE at 256 px |
| --- | --- | --- | --- | --- |
| keyframe_strict | 74.41% | 38.55% | 71/471 | 46.747 px |
| keyframe_lineage | 77.00% | 38.78% | 66/471 | 48.519 px |
| hybrid_strict | 45.29% | 36.48% | 23/471 | 26.265 px |
| hybrid_lineage | 48.39% | 36.63% | 19/471 | 25.953 px |

The denominator remains **30 DAVIS videos, 1,999 frames, 650 queries and 28,174 visible post-query point-times**. Unmatched queries stay in the denominator. Initialization reaches **567/650** versus v3's 467/650 and corrected v2's 583/650; gold-fish recovers **23/25** initial queries versus v3's 0/25.

Mesh control restores the ≥62% coverage and ≥50-return gates, but its point agreement remains below 42%. Direct tracking lowers conditional RMSE relative to this mesh control, yet loses enough observed coverage and post-occlusion returns that its full-denominator point agreement is lower. It does not reproduce the standalone LK reference's 5.31 px conditional RMSE or 47.32% point agreement. That standalone baseline also initializes every supplied point and has a different initialization contract; it is not a controlled model-only comparison.

These are image-point diagnostics, not verified physical-object IDF1. Lineage is a causal spatial hypothesis, and never uses later reference coordinates to select a branch. Saved traces contain 136 subpart-adoption events; these counts do not prove correct physical identity.

## Compute and discovery

| Criterion | Target | v3 | v4 | Outcome |
| --- | --- | --- | --- | --- |
| DAVIS mean ingestion | ≤13.5 ms/frame | 19.482 | 24.505 | Fail |
| Native mean ingestion | ≤13.5 ms/frame | 23.249 | 26.714 | Fail |
| Sampled process-tree RSS | ≤18 GiB | 4.650 GiB | 4.519 GiB | Pass |
| Detector-only VISOR recall, IoU ≥0.5 | ≥75% | 6/8 = 75% | 7/8 = 87.5% | Pass |

There are **206/1,999 keyframes (10.31%)**, with triggers `{'stride': 206}`. No empty, flux or flow-loss bursts remain. DAVIS uses a 256-pixel detector input and native video uses 480. Native archives were generated afresh: **20 episodes, 6,078 frames and 180,216 observed detector/LK carrier records**.

DAVIS v4 timing includes joint particle LK, gating and re-seeding; initial query binding and scoring are excluded. The same joint runtime is charged to the mesh controls, so these timings do not isolate mesh-only execution. V3 DAVIS timing had no direct particles. Native timing is query-free in both versions and is the clearer ingestion comparison. Cold detector warmup and source decoding are separately excluded; runtime appearance rescue remains included. Fewer keyframes did not establish the proposed 5.5–6.5 ms amortized/frontend or ≤13.5 ms overall budget. No matched component profile isolates the exact cost contribution of detector vocabulary, region count, association and query handling.

## Sparse VISOR tracking

Detector-only probes are evaluation calls at eight locked timestamps, never injected into the causal tracker. Later P21 detector IoU is 0.385 and is the only IoU-0.5 failure. Later foreground support of the initially bound carrier is:

| Clip | Strict ID | Model-only lineage |
| --- | --- | --- |
| P02_122 | 0.78% | 0.78% |
| P08_09 | 0.00% | 93.37% |
| P06_108 | 94.68% | 94.68% |
| P21_01 | 0.57% | 0.57% |

Lineage restores substantial P08 support, while strict P08 support remains zero. The weak nonzero-lineage criterion now passes for P06/P08/P21, but P21 support is only about 0.57%; this is not the proposed 90%+ recovery or proof of identity retention. VISOR is a carrier-mask diagnostic; the `hybrid_*` mask rows share these carrier results, not independent particle-mask predictions.

## Silver association proxy and covariance diagnostics

| Metric | Required / meaning | v4 result |
| --- | --- | --- |
| Target episode binding | 2/2 | 2/2 |
| Evaluable fixed-ID frames | ≥265/300 | 290/300 — pass |
| Unevaluable high-confidence frames | Explicit missingness | 10/300 |
| In-support proxy over all eligible labels | ≤50 native px OR ≤2.5σ of shape scatter | 50/300 = 16.67% |
| In-support among evaluable states | Conditional availability denominator | 50/290 = 17.24% |
| Full scored-sequence mean 2-D NEES | [1.6, 2.6] | 6996.495 — fail |
| In-track mean 2-D NEES | Reference-conditioned in-support subset | 181.278 on 50 states |
| Full scored-sequence joint 95% coverage | ≥88% | 14.83% — fail |

“In-support” is a geometric/error proxy, not ground-truth association accuracy. It uses the silver reference to select frames, so in-track NEES is selection-biased. Broad shape support can pass the geometric test even when semantic identity is wrong. The subset neither replaces the full scored-sequence NEES gate nor removes missing frames from the availability denominator. **Even this selected subset remains far outside the desired NEES range.** Full-sequence NEES is computed on the 290 evaluable fixed-ID states; undefined errors on the ten missing states are not imputed as zero.

The old fit supplies initialization only. A new development-only fit excluding P08/P18 converged in **40 objective evaluations**: continuous-white-acceleration q=0.850969135, localization floor=0.0000987718, mean development detector NIS=2.071356 on 11,372 detector updates. Recorded LK displacement diffuses covariance with 0.012²·dt and is not treated as an independent Kalman position measurement. Independent scalar replay matches the batched objective and all fixed-ID/subset scores.

The fitted floor is very small; it is only one covariance component, not the complete posterior standard deviation. Near-two development NIS does not certify correct target association or agreement between detector-mask centroids and model-estimated reference support. The silver score sets and fitted parameters differ across versions, so pooled v3/v4 NEES is not a paired same-frame error comparison.

All references remain **`human_adjudicated=false`**. Human-reference NEES, physical-object IDF1 and contact truth remain unknown. This offline fixed-association replay does not calibrate the live tracker. Stage 2 is not promoted and the Stage 3 serialization benchmark has not been unlocked.

## Verification and reproduction

The point verifier initially found a near-zero-coordinate mismatch below 0.00002 pixels. The independent replay was retaining float64 arithmetic where runtime LK points/anchors are float32. Replay now preserves those dtypes; all original inference arrays, traces and runtime source snapshots remain unchanged. The corrected verifier and wrapper are frozen under `verification_source/` with their hashes. Its source digest matches `verification.json`; qualification thresholds and scored predictions were not relaxed or rewritten.

Use new tags when repeating inference or calibration. These requested commands were executed:

```sh
export PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/anaconda3/bin/python3.13 -m pytest -q tests --basetemp work/pytest_observation_identity_v4
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/03_eval_observation_identity.py --tag observation_identity_v4 --videos 30 --keyframe-stride 10
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/05_probe_keyframe_visor.py --tag observation_identity_v4
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/04_verify_observation_identity.py --tag observation_identity_v4
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py --cohort reports/track1_fmr/observation_identity_v4 --tag stage2_silver_v6 --reuse-fit reports/track1_fmr/stage2_silver_calibration_v3_devfit/dev_fit.json
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/06_verify_keyframe_calibration.py --tag stage2_silver_v6
```

Inference code/checkpoints/inputs are hashed in `protocol.json` and `source/`. Metrics and evidence are in `assessment.json`, `verification.json`, point archives, particle/entity traces and native observation archives. Calibration and its independent verifier are in `../stage2_silver_v6/`. The qualified Stage 1 extractor and frozen spline/LK source trees are unchanged.
