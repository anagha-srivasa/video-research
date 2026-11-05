# Bounded Track 1 remediation: results and limits

Track 1 remains open development. Track 2 remains on hold. All four requested candidates are implemented as separate experiments; existing releases/defaults are preserved. No scientific freeze or candidate promotion is claimed.

## A: source epochs and piecewise geometry

Every one of the 1,749 original M32 epoch-failure windows was audited. 1,745 contain observed source ring-graph changes; only four have unchanged topology with unresolved correspondence. A source graph change is not proof of a physical component birth/death: dense interpolation, fragmentation, occlusion and identity ambiguity remain competing explanations. No full-window continuity recovery is demonstrated by the correspondence matcher.

The additional geometry view fits only the latest stable epoch (at least eight observations and 0.3 seconds of support), with persistent ring IDs restricted to that epoch. A known change whose latest spatial fit is invalid cannot resurrect the old epoch. Accepted geometry has complete retained source support and a bound continuous-curve certificate. Full one-second geometry remains available wherever the baseline already passed.

| source | epoch_failures | baseline_full_complete_certified | latest_epoch_segment_complete_certified |
| --- | --- | --- | --- |
| v2 | 1749 | 847 | 1470 |
| context | 864 | 792 | 1094 |

These counts compare different history durations. They do not reclassify failed full one-second histories as valid. [Support intersections and unions](support_tradeoffs.json) explicitly retain baseline-only and segment-only cases.

## C: tiered continuity and source reconstruction

M16 carries the functional history and original separate local-CV kinematics. Optional M32 geometry prefers the full history, otherwise using the latest certified epoch. Missing geometry is `None`. Per-state durations, availability, and adjacent geometry-epoch flags are stored. The original M16 geometry-completeness flag is distinct from optional M32 availability.

| split | M16_core_sequences | all_optional_M32_available | single_compound_epoch |
| --- | --- | --- | --- |
| train | 266 | 104 | 99 |
| validation | 46 | 5 | 5 |

Counts are overlapping sequences, not independent episodes. All five validation sequences with compound geometry throughout are in P08_09; P18_03 has none. The final column retains one compound epoch. The optional-view total may include explicit resets, and is not a continuous compound trajectory across those resets. The earlier 376/51 totals summed separate model representations; the tiered reader uses the 266/46 M16 carrier cohort rather than adding those representations together.

On the same 25 validation probes where a newly available short segment and a certified M16 geometry view can both be decoded:

| method | probes | IoU | BF2 |
| --- | --- | --- | --- |
| core16_canonical | 25 | 0.7507 | 0.2625 |
| tiered32_canonical | 25 | 0.8882 | 0.6105 |

This is in-sample canonical source-mask reconstruction, with the target contour retained as input. It does not establish held-out RGB segmentation quality or the fresh-mask acceptance gates. The full-support paired table is in [geometry_quality.json](geometry_quality.json); tiered geometry exactly retains the full M32 view where available.

## B: causal hypothesis bank and background alignment

The fixed three-hypothesis bank uses stationary, local CV and a regularized transition/acceleration model with decaying acceleration. Past-only prequential scores update hypothesis weights; they are uncalibrated model weights, not independently validated physical regime probabilities. No validation parameter search was run.

Matched validation source-centroid continuation, mean of per-video RMSE over two participants:

| horizon_seconds | queries | nearest_RMSE_px | CV_RMSE_px | bank_RMSE_px |
| --- | --- | --- | --- | --- |
| 0.1 | 185 | 37.06 | 50.44 | 34.71 |
| 0.25 | 175 | 74.48 | 112.61 | 74.16 |
| 0.5 | 158 | 121.83 | 217.34 | 138.12 |

The bank beats CV here, but nearest still wins at 500 ms. These source centroids can move because of deformation, occlusion and camera motion. Camera-supported comparisons use a separate common cohort: background compensation does not consistently beat nearest or the raw bank, so it is not promoted. Background confidence failures reset the camera epoch; no future camera observations are used. The camera-relative view is 2D image alignment, not metric scene stabilization.

Exact synthetic stress results (30 validation episodes, matched cached cutoffs):

| event_recall | hop_seconds | method | position_rmse_px | quiet_false_alarm_fraction | velocity_rmse_px_s |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.125 | nearest | 13.8707 | 0.0 | 185.5209 |
| 0.6667 | 0.125 | cv | 13.3046 | 0.0542 | 172.5146 |
| 0.3333 | 0.125 | switching | 10.3482 | 0.0109 | 152.3927 |
| 0.0 | 0.25 | nearest | 20.9996 | 0.0 | 185.4386 |
| 0.6667 | 0.25 | cv | 21.3573 | 0.072 | 179.4119 |
| 0.3333 | 0.25 | switching | 15.6244 | 0.0176 | 157.4759 |

The bank reduces position/velocity errors and quiet false movement but cuts event recall from 2/3 to 1/3. That fails the intended event-preservation requirement. Switching motion remains a sampled diagnostic candidate; the default reader keeps the original CV estimator.

## D: association and independent contact review

Paired association diagnostics use the same queries, with correctly initialized past source-chain memory. Target source IDs are used only for evaluation, not as matching features. They are provisional annotation chains, not independently verified identities. These tests do not measure complete online tracking or error propagation through real occlusion.

| method | source_chain_agreement | abstention_fraction |
| --- | --- | --- |
| geometry | 0.9864 | 0.0068 |
| geometry_appearance | 0.9898 | 0.0102 |
| geometry_appearance_motion_uncertainty | 0.795 | 0.2005 |

Appearance gives a small source-agreement improvement; adding the current motion-uncertainty cost causes many abstentions. Neither source agreement nor abstention establishes independent IDF1 or consequential contact-swap accuracy.

163 categorical contact proposals have been assembled with two-reviewer boundary worksheets. Source boundary proximity is a diagnostic, not physical signed contact distance. Pre-contact, first contact, sustained contact, release, occluded/high-speed/small-object contact remain independently unlabeled. No annotations were fabricated. Accepted independent references: **0**. Boundary-distance conventions, reviewer reliability, adjudication, identity and event labels remain pending.

## Uncertainty and final acceptance

Position covariance and hypothesis weights are conditional and uncalibrated. Joint position/velocity covariance, geometry covariance, calibrated identity/topology probabilities and contact intervals remain unavailable rather than being filled with false certainty.

For a simultaneous 95% rank-based calibration using five participant units, the required rank is six and the finite threshold does not exist. Exchangeability and an untouched calibration split are also unestablished here, so this is a feasibility diagnostic, not a formal conformal guarantee. See [Tibshirani's conformal notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf), page 2, for the finite-sample rank/infinity convention.

Fresh participants remain reserved and uninspected; an acquired, independently annotated, locked evaluation is absent. [Scientific gate ledger](scientific_readiness.json) records the remaining blockers. Numerical readiness does not pass those gates.

## Verification and reproduction

74 tests passed. Verification checked 2,564 accepted geometry certificates, 672 past-only motion states, 1,200 background frames and 42 reader probes. A same-environment refit of the full P18_03 candidate geometry shard matched its stored hashes; this is not a clean rebuild from all raw data. All prior baseline, retraining and readiness manifest checksums were rechecked and preserved. Dataset payload stays uncompressed and unchanged at 238.715 GB.

[Dashboard](dashboard.html) · [Implementation and reproduction](../../TRACK1_REMEDIATION.md) · [Verification](verification.json) · [Preservation](baseline_integrity.json) · [Deterministic subset](determinism.json)
