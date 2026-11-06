# Probability signal validation

Exploratory source-displacement diagnostics on the same two exposed participants. No physical-motion acceptance or model promotion.

Scale multipliers use held-training-participant predictions only. AUC and reliability bins are descriptive; overlapping queries are not independent samples. Fixed 0.5 is a diagnostic, not a fitted baseline.

| Cohort | Horizon | Arm | Brier (participant mean) | AUC (participant mean) |
|---|---:|---|---:|---:|
| object | 0.25 | energy | 0.28007 | 0.8520 |
| object | 0.25 | position | 0.25968 | 0.8770 |
| object | 0.25 | position_motion | 0.24973 | 0.8762 |
| object | 0.25 | position_metadata | 0.25722 | 0.9008 |
| object | 0.25 | both | 0.25026 | 0.8972 |
| object | 0.25 | relation | 0.25698 | 0.8464 |
| object | 0.25 | functional | 0.24924 | 0.9127 |
| object | 0.25 | memory | 0.24991 | 0.9087 |
| object | 0.5 | energy | 0.26188 | 0.6690 |
| object | 0.5 | position | 0.24673 | 0.7355 |
| object | 0.5 | position_motion | 0.24373 | 0.7604 |
| object | 0.5 | position_metadata | 0.24692 | 0.7659 |
| object | 0.5 | both | 0.24523 | 0.7798 |
| object | 0.5 | relation | 0.25330 | 0.7380 |
| object | 0.5 | functional | 0.24551 | 0.8073 |
| object | 0.5 | memory | 0.24727 | 0.8101 |
| object | 1.0 | energy | 0.31920 | 0.5511 |
| object | 1.0 | position | 0.32403 | 0.5832 |
| object | 1.0 | position_motion | 0.32833 | 0.6508 |
| object | 1.0 | position_metadata | 0.32447 | 0.6625 |
| object | 1.0 | both | 0.32827 | 0.7063 |
| object | 1.0 | relation | 0.33795 | 0.7143 |
| object | 1.0 | functional | 0.32735 | 0.7143 |
| object | 1.0 | memory | 0.32801 | 0.7302 |
| hand | 0.25 | energy | 0.23434 | 0.8004 |
| hand | 0.25 | position | 0.21500 | 0.8116 |
| hand | 0.25 | position_motion | 0.20442 | 0.8224 |
| hand | 0.25 | position_metadata | 0.20913 | 0.8173 |
| hand | 0.25 | both | 0.20263 | 0.8237 |
| hand | 0.25 | relation | 0.20517 | 0.8050 |
| hand | 0.25 | functional | 0.20254 | 0.8212 |
| hand | 0.25 | memory | 0.20269 | 0.8196 |
| hand | 0.5 | energy | 0.23443 | 0.7223 |
| hand | 0.5 | position | 0.22056 | 0.7242 |
| hand | 0.5 | position_motion | 0.21648 | 0.7462 |
| hand | 0.5 | position_metadata | 0.21914 | 0.7233 |
| hand | 0.5 | both | 0.21684 | 0.7378 |
| hand | 0.5 | relation | 0.21610 | 0.7319 |
| hand | 0.5 | functional | 0.21775 | 0.7434 |
| hand | 0.5 | memory | 0.21789 | 0.7411 |
| hand | 1.0 | energy | 0.25977 | 0.6812 |
| hand | 1.0 | position | 0.24946 | 0.6721 |
| hand | 1.0 | position_motion | 0.24378 | 0.7178 |
| hand | 1.0 | position_metadata | 0.24698 | 0.6861 |
| hand | 1.0 | both | 0.24401 | 0.7006 |
| hand | 1.0 | relation | 0.24758 | 0.6722 |
| hand | 1.0 | functional | 0.24390 | 0.7027 |
| hand | 1.0 | memory | 0.24572 | 0.6925 |

## Fixed baselines and conditional-scale calibration

| Cohort | Horizon | Frequency Brier | Fixed 0.5 Brier | Both coverage before / after | Mean 95% radius before / after |
|---|---:|---:|---:|---:|---:|
| object | 0.25 | 0.28426 | 0.25000 | 0.905 / 0.905 | 0.0729 / 0.0818 |
| object | 0.5 | 0.27091 | 0.25000 | 0.888 / 0.938 | 0.1177 / 0.1361 |
| object | 1.0 | 0.31497 | 0.25000 | 0.819 / 0.850 | 0.1655 / 0.2011 |
| hand | 0.25 | 0.24412 | 0.25000 | 0.863 / 0.879 | 0.1265 / 0.1309 |
| hand | 0.5 | 0.24009 | 0.25000 | 0.798 / 0.819 | 0.1782 / 0.1898 |
| hand | 1.0 | 0.26594 | 0.25000 | 0.731 / 0.762 | 0.2059 / 0.2201 |

Position arms retain the v7 per-hop displacement norms. Functional augmentation contains position coefficients only. Every arm uses its own training-only RMS scaling and the same fixed regularization; this is a bounded capacity comparison, not causal attribution.

The initial diagnostic run was archived under work/track2_probability/initial_diagnostic before correcting baseline feature parity and using each past hop own observation for its memory update. No parameters were selected from either validation run.

## Limits
Synthetic controls reproduce feature-level noise mechanisms, not the complete irregular annotation pipeline. Independent-anchor synthetic controls do not remove the shared-anchor confound from the real dataset. Source ancestry and independent physical references remain unresolved. No claim that observed gains survive independent measurement is supported.

Memory is a fixed past-only position-coefficient projection; this tests additional position history, not certified shape or a Koopman generator. Relation gains remain confounded by common scene motion; no matched background-motion feature is added in this round.

Detailed per-participant discrimination, reliability bins, 95% radii/coverage, NLL and Gaussian-implied versus logistic probabilities: scores.json. Per-query evidence: predictions.parquet. Training-only calibration provenance: calibration.json.
