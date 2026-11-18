# Observation–identity v2: completed evaluation, failed qualification

The final point/VISOR evaluation is `observation_identity_v2_keyframe_evalfix`. The first `observation_identity_v2_keyframe` run is preserved; its point results are superseded because delayed query binding checked stale mask coordinates. The corrected implementation binds current measured support, without translating fresh detector masks by filter error. Both versions use the same runtime tracker and checkpoint hashes. New native measurements and their isolated timings were generated once in the original run and copied byte-for-byte into the corrected report. `native_calibration_link.json` verifies all 20 files against the calibration inputs.

Software validation: **292 passed**, with 18 existing TorchScript deprecation warnings. Independent verification passes for input/output hashes, frozen Stage 1 source, all point denominators, per-anchor FB evidence on DAVIS, causal lineage edges and replayed point transport using only initial query coordinates. Native source tags/hashes are checked; native per-anchor traces are not separately retained. The calibration verifier independently recomputes binding and silver NEES and checks the batched development objective against scalar filter replay. Software integrity is distinct from passing the scientific gates.

| Criterion | Target | Strict ID | Model-only lineage | Outcome |
| --- | --- | --- | --- | --- |
| Visible observed coverage | ≥62% | 64.50% | 67.91% | Pass |
| Observed point agreement, ≤8 px at 256 resolution | ≥42% | 32.49% | 33.23% | Fail |
| Occlusion returns recovered within five frames | ≥50/471 | 36/471 | 40/471 | Fail |
| DAVIS mean ingestion | ≤13.5 ms/frame | 36.401 | Same runtime | Fail |
| Native mean ingestion | ≤13.5 ms/frame | 38.020 | Same runtime | Fail |
| Sampled process-tree RSS | ≤18 GiB | 5.307 GiB | Shared runtime | Pass |
| Detector-only VISOR recall, mask IoU ≥0.5 | ≥75% | 7/8 = 87.5% | Same detector | Pass |
| Nonzero later P06/P08/P21 support | All three | P06/P08 only | P06 only | Fail |
| Silver episode binding | 2/2 | 2/2 | Fixed ID calibration | Pass |
| Silver evaluable frames | ≥255/300 | 207/300 | Fixed ID calibration | Fail |
| Silver mean 2-D NEES | [1.6, 2.6] | 468.522 | Fixed ID calibration | Fail |
| Silver joint 95% coverage | ≥88% | 1.45% | Fixed ID calibration | Fail |

The point denominator remains **30 videos, 1,999 frames, 650 queries and 28,174 visible point-times**. Unmatched queries stay in it; prediction-only carriers get no observed credit. Native inference covers **20 episodes, 6,078 frames and 179,657 actual detector/LK measurements**. Native timings and RSS include the original native run; reused measurements do not imply a second native timing run.

There were 443 detector keyframes in 1999 DAVIS frames (22.16%). Trigger counts overlap: {'empty': 30, 'stride': 306, 'unexplained_flux': 142, 'flow_loss': 8}. Two detector heads are executed per keyframe. The proposed eightfold frontend reduction was not established. V1 already measured moments on the 160×160 grid; avoiding native mask upsampling was not a verified new optimization.

Later VISOR foreground recall:

| Clip | Strict ID | Lineage |
| --- | --- | --- |
| P02_122 | 0.00% | 0.00% |
| P08_09 | 42.15% | 0.00% |
| P06_108 | 97.19% | 97.39% |
| P21_01 | 0.00% | 0.00% |

Detector-only probes are separate evaluation calls on eight locked timestamps. Their detections never enter the tracker and their timing is excluded from ingestion. On timestamps without a detector keyframe, the scheduled-detector IoU field is zero; that is distinct from detector capability and from FB-LK support.

Development-only calibration refits the new cohort with the old fit as an initializer: q=2.744323386, localization floor=0.018860219, development mean NIS=2.055905, optimizer converged in 19 objective evaluations. P08/P18 are excluded from fitting. The held silver target means remain far from the fixed carriers despite a near-two development NIS; fitting innovation covariance does not fix target association. Mean native-coordinate errors are 416.81 px on the 132 evaluable P08 frames and 261.18 px on the 75 evaluable P18 frames. This is consistent with target-support/association mismatch and drift, but the model labels cannot independently certify physical identities. LK-derived centroid updates are also temporally correlated.

**No Stage 2 promotion.** Labels remain `human_adjudicated=false`; physical-object IDF1, human-reference NEES and contact scores remain unknown. The live tracker is not calibrated by this offline replay. The qualified Stage 1 extractor and frozen spline/LK sources are unchanged. Richer geometry/contact work remains gated.

## Reproduction

Use a fresh tag for each run; existing experiment directories are not overwritten.

```sh
export PYTHONPATH=src:work/deps:work/track1_deps:work/track1_fmr_deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/anaconda3/bin/python3.13 -m pytest -q tests --basetemp work/pytest_observation_identity_v2
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/03_eval_observation_identity.py --tag NEW_TAG --videos 30 --keyframe-stride 6
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/05_probe_keyframe_visor.py --tag NEW_TAG
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/04_verify_observation_identity.py --tag NEW_TAG
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/02c_calibrate_and_eval_nees.py --cohort reports/track1_fmr/NEW_TAG --tag NEW_CALIBRATION_TAG --reuse-fit reports/track1_fmr/stage2_silver_calibration_v3_devfit/dev_fit.json
/opt/anaconda3/bin/python3.13 scripts/exp_track1_fmr/06_verify_keyframe_calibration.py --tag NEW_CALIBRATION_TAG
```

The exact corrected run used `--reuse-native-from observation_identity_v2_keyframe`; its protocol records and verifies the original native provenance. The original full native run and silver calibration used the requested tags.
