# Observation–identity v3: evaluated, not qualified

The requested revision and all six validation commands were completed with fresh immutable tags. **299 repository tests pass** (18 existing TorchScript deprecation warnings). Both the observation/identity verifier and the silver-calibration verifier pass. **Stage 2 qualification fails; Stage 3 remains gated.**

This comparison uses the corrected v2 evaluation. Values separated by `/` are **strict ID / model-only lineage**, not repeated runs. These are already-consumed diagnostic references, not fresh holdouts.

| Criterion | Required | v2 corrected | v3 | v3 outcome |
| --- | --- | --- | --- | --- |
| Visible observed point coverage | ≥62% | 64.50% / 67.91% | 46.33% / 47.97% | Fail |
| Observed point agreement ≤8 px at 256 resolution | ≥42% | 32.49% / 33.23% | 25.55% / 25.34% | Fail |
| Occlusion returns recovered within five frames | ≥50/471 | 36 / 40 | 22 / 20 | Fail |
| DAVIS mean ingestion | ≤13.5 ms/frame | 36.401 | 19.482 | Fail |
| Native mean ingestion | ≤13.5 ms/frame | 38.020 | 23.249 | Fail |
| Sampled process-tree RSS | ≤18 GiB | 5.307 GiB | 4.650 GiB | Pass |
| Detector-only VISOR mask recall, IoU ≥0.5 | ≥75% | 7/8 = 87.5% | 6/8 = 75.0% | Pass |
| Nonzero later P06/P08/P21 support | All three | P08 fails in lineage; P21 fails in both | P08 fails in both | Fail |
| Silver target episode binding | 2/2 | 2/2 | 2/2 | Pass |
| Evaluable silver frames | ≥265/300 for v3 | 207/300 | 232/300 | Fail |
| Silver mean 2-D NEES | [1.6, 2.6] | 468.522 | 480.743 | Fail |
| Silver joint 95% coverage | ≥88% | 1.45% | 11.64% | Fail |

The denominator is unchanged: **30 DAVIS videos, 1,999 frames, 650 queries, and 28,174 visible post-query point-times**. Uninitialized queries remain in the denominator. Prediction-only carriers receive no observed credit. V3 freshly processed **20 native episodes, 6,078 frames and 76,566 detector/LK-supported observations**; no old native measurements or timings were reused.

## What the measured changes establish

Single-head stride-eight ingestion is faster, but the proposed ≤13.5 ms budget is not met. There were **320/1999 detector keyframes (16.01%)**, rather than ≤14%. Trigger counts overlap: `{'empty': 93, 'stride': 257}`. There are no flux or flow-loss triggers. Gold-fish alone had 62 empty-track triggers and initialized **0/25 queries**. The supplied prompt bank has no fish category; these observations are consistent with reduced discovery support after removing the prompt-free head, though this combined revision does not isolate the causal contribution of prompts, input size or detector removal.

Initial query binding fell from **583/650** in v2 to **467/650** in v3. V3 trace entity/frame evidence counts are `{'detector': 1373, 'fb_lk': 4961, 'predicted': 7301}`; these are repeated carrier records, not physical-object counts. Stricter photometric rejection and changed discovery support accompany lower observed coverage. No independent ablation identifies their individual effects.

Native verification checks saved FB inliers, FB error <1.2 px and NCC ≥0.65 on every LK-supported row. It also reconstructs covariance propagation from preceding recorded states and confirms that LK adds uncertainty without a Kalman measurement contraction. DAVIS verification separately replays query transport from initial queries and saved anchor geometry without later reference positions. The zero-innovation bug alleged in the pasted diagnosis was not present in v2; see `IMPLEMENTATION.md` for source-level corrections and implementation details.

## VISOR: detection versus tracking

Detector-only probes are separate calls on eight locked frames; their detections never enter the causal tracker and their timing is not part of the ingestion benchmark. P02 initialization has detector IoU 0.0; later P21 detector IoU is 0.385. These are the two IoU-0.5 failures. Detection recall therefore meets 75%, but does not preserve v2's 87.5%.

Later-frame foreground support of the initially bound carrier:

| Clip | Strict ID | Model-only lineage |
| --- | --- | --- |
| P02_122 | 0.00% | 0.00% |
| P08_09 | 0.00% | 0.00% |
| P06_108 | 98.39% | 98.39% |
| P21_01 | 4.61% | 34.11% |

P21 lineage support improves to a partial region, but the proposed 90%+ recovery is not established. P08 retention does not improve. High foreground recall on one sparse frame is not complete physical-identity validation.

## Covariance and silver calibration

Calibration now fits **detector innovations only**, with recorded LK displacement and covariance diffusion between them. The old fit is an initializer; the new development fit converged in 119 objective evaluations to **q=0.556508774, detector localization floor=0.014068084**, and mean development NIS **2.048133** on 6,907 detector updates. The q parameter remains continuous-white-acceleration spectral density, rather than silently adopting the pasted code's different squared-noise convention.

P08 and P18 remain excluded from fitting. An independent scalar implementation reproduces the batched objective and the fixed-ID bindings/NEES. LK propagation does not receive fictitious independent-measurement NIS terms. The near-two development NIS does not establish correct target association: held silver NEES and joint coverage still fail badly.

The v2/v3 silver averages cover different evaluable frame sets and different carrier IDs; they are not a paired same-frame NEES comparison. Model-reviewed references remain explicitly **`human_adjudicated=false`**. Human-reference NEES, complete object IDF1 and physical-contact truth remain unavailable. This is an offline fixed-association diagnostic, not live-pipeline calibration or scientific promotion.

## Reproduction and artifacts

Use fresh tags; existing outputs are not overwritten. All requested commands were run:

```sh
export PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/anaconda3/bin/python3.13 -m pytest -q tests --basetemp work/pytest_observation_identity_v3
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/03_eval_observation_identity.py --tag observation_identity_v3 --videos 30 --keyframe-stride 8
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/05_probe_keyframe_visor.py --tag observation_identity_v3
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/04_verify_observation_identity.py --tag observation_identity_v3
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py --cohort reports/track1_fmr/observation_identity_v3 --tag stage2_silver_v5 --reuse-fit reports/track1_fmr/stage2_silver_calibration_v3_devfit/dev_fit.json
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/06_verify_keyframe_calibration.py --tag stage2_silver_v5
```

`protocol.json` and `source/` preserve inference source/checkpoint/input hashes. `assessment.json`, `verification.json`, point arrays, entity traces and native observation archives retain the measured evidence. Silver calibration and its verifier are in `../stage2_silver_v5/`. Qualified Stage 1 and frozen spline/LK sources are unchanged. The variable-epoch Stage 3 serialization benchmark has not been unlocked.
