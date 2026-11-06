# Track 2 signal diagnostics

This fresh experiment implements the v6 assessment's diagnostic priorities. Full Track 2 promotion and the existing mean-position gates remain unchanged. All evaluations use source-derived targets on exposed participants; independent physical labels remain absent.

## Measured findings

The original training pipeline learns the predictable controls without an initialization change. Selected loss is about 2–3% of persistence loss on the three positive controls. The null control remains near persistence and real-source selection mostly returns epoch zero. Generator-reset effects remain explicit: controlled fit success is not proof of latent system identification.

| Cohort | Horizon | Frequency Brier | History Brier | Corrected relation Brier | History improvement |
|---|---:|---:|---:|---:|---:|
| object | 0.25 s | 0.28426 | 0.25968 | 0.25698 | 8.65% |
| object | 0.5 s | 0.27091 | 0.24989 | 0.25618 | 7.76% |
| object | 1 s | 0.31497 | 0.32286 | 0.33880 | -2.51% |
| hand | 0.25 s | 0.24412 | 0.21500 | 0.20517 | 11.93% |
| hand | 0.5 s | 0.24009 | 0.22076 | 0.21617 | 8.05% |
| hand | 1 s | 0.26594 | 0.25027 | 0.24779 | 5.89% |


Hand-history Brier improvement is positive for both exposed validation participants at all three horizons. Object-history improvement is positive on both at 250 ms; at 500 ms one participant regresses slightly and at one second aggregate improvement disappears. Corrected hand relations do not consistently improve on history/reliability, and the alternate-pair control can score better, so no specific interaction-association benefit is established. Scale-NLL gains also vary by cohort and horizon; stronger probability scores do not certify full uncertainty calibration.

| Incoming anchor | 250 ms | 500 ms | 1 s |
|---|---:|---:|---:|
| functional_endpoint | 0.11198 | 0.16208 | 0.18928 |
| latest_observation | 0.09040 | 0.15391 | 0.18483 |
| causal_position_estimate | 0.09596 | 0.15678 | 0.18584 |


Latest-observation incoming anchors improve the functional endpoint diagnostic, most at 250 ms. These targets differ from the cached kinematic point-forecast benchmark. The signed-drift comparison still selects persistence for objects and intercept-free ridge for hands on training-participant folds; the new probability task does not retroactively change those forecast decisions.

## Hand classification and measurement timing

Hand selection uses the EPIC noun mapping: hand/left hand/right hand classes, plus explicitly sided glove classes under a declared source-label policy. Generic gloves are excluded because wearing is ambiguous. Names containing “handle” cannot establish a hand. hand_policy.json records names/classes and actual counts. The existing selected source cohort contains left/right hand classes; a predicate defect does not imply handles occurred in it.

Object and hand slopes use actual source timestamps within a fixed 200 ms past window, at least three distinct observations, at least 40 ms span and maximum 100 ms age. Repeated timestamps are collapsed. No fixed 250 ms denominator is used for source slopes. Two-point and fitted slopes, support, age, spans and selected pairs remain in selected_pairs.parquet. Fixed interaction invariants include distance, approach/separation rate and relative speed, with explicit validity. Own-history distance/speed summaries are supplied fairly to applicable controls.

Object and hand target cohorts are evaluated separately. Time shuffling tests temporal order within support-matched past hops. Pair shuffling assigns an alternate observed hand with matching four-hop presence/velocity support and age within 50 ms. Where no alternate is available, the features remain unchanged and effective-shuffle counts are explicit; it does not certify physical association.

[
  {
    "cohort": "object",
    "eligible_pair_shuffle_effective": 396,
    "histories": 600,
    "input_eligible_histories": 600,
    "pair_shuffle_effective": 396,
    "split": "train"
  },
  {
    "cohort": "hand",
    "eligible_pair_shuffle_effective": 0,
    "histories": 447,
    "input_eligible_histories": 447,
    "pair_shuffle_effective": 0,
    "split": "train"
  },
  {
    "cohort": "unknown",
    "eligible_pair_shuffle_effective": 0,
    "histories": 0,
    "input_eligible_histories": 0,
    "pair_shuffle_effective": 0,
    "split": "train"
  },
  {
    "cohort": "object",
    "eligible_pair_shuffle_effective": 57,
    "histories": 64,
    "input_eligible_histories": 61,
    "pair_shuffle_effective": 57,
    "split": "validation"
  },
  {
    "cohort": "hand",
    "eligible_pair_shuffle_effective": 0,
    "histories": 160,
    "input_eligible_histories": 160,
    "pair_shuffle_effective": 0,
    "split": "validation"
  },
  {
    "cohort": "unknown",
    "eligible_pair_shuffle_effective": 0,
    "histories": 0,
    "input_eligible_histories": 0,
    "pair_shuffle_effective": 0,
    "split": "validation"
  }
]

## Actual-protocol optimization controls

| Task | Folds | Median loss / persistence | Selected epochs |
|---|---:|---:|---|
| constant_velocity | 5 | 0.0211 | [198, 199, 194, 200, 200] |
| integrated_acceleration | 5 | 0.0249 | [199, 199, 196, 200, 196] |
| stable_latent | 5 | 0.0254 | [200, 199, 200, 196, 200] |
| shuffled_displacement | 5 | 0.9996 | [0, 0, 13, 15, 27] |
| real | 5 | 1.0000 | [0, 0, 9, 0, 12] |

Twenty synthetic fold fits and five instrumented real-source fold fits use the original v6 model, Data rows, masks, normalization, optimizer, minibatches, weights and epoch-zero stopping. The only synthetic intervention is replacement of position/velocity targets by known functions of input history or context-independent noise. This is a conditional learnability check on the actual input distribution, not a complete physical simulator. Control thresholds were configured before execution. Positive-control follow-up flags: [].

Per-step/epoch logs record un-clipped generator, damping, encoder, acceleration and velocity gradient norms, plus generator change from initialization. A selected-state ablation restores the initial generator and records forecast change. Passing a predictable target does not establish that the true latent generator was recovered; readouts can approximate dynamics. No initialization change, warm-start, weak-form loss or real-data architecture search was introduced in this round.

## Intercepts, probabilities and scale

Four signed-displacement probes compare persistence, mean drift alone, ridge without an intercept, and ridge with an unpenalized intercept. The zero-correction candidate remains in train-participant selection. Fixed regularization is 100; there is no grid search.

A separate question predicts whether source displacement exceeds a training-only 75% quantile. Context-free frequency, object history, reliability, missingness, corrected relations and both shuffle controls are scored with Brier and Bernoulli log loss. This target is source displacement, not physical contact, onset or intent. No probability result replaces the failed point-forecast gate.

The conditional isotropic residual-scale probe uses Gaussian source-displacement NLL around persistence, compared with a horizon-specific constant scale. Its bounded log-variance correction avoids uncontrolled extrapolation. Radial coverage and probability bins are empirical source-target diagnostics, not independently calibrated physical uncertainty. Models use fixed regularization and training-only thresholds/transforms; numerical convergence is recorded in probability_protocol_results.json. Per-participant/episode scores and calibration bins are separate artifacts. No confidence interval treats overlapping queries as independent.

## Complete memory operator and anchor comparison

[
  {
    "complete_endpoint_fill_norm": 2.1751801165382454,
    "horizon_seconds": 0.25,
    "retention_norm": 0.981621762724208
  },
  {
    "complete_endpoint_fill_norm": 2.952493324868143,
    "horizon_seconds": 0.5,
    "retention_norm": 0.9500050270483552
  },
  {
    "complete_endpoint_fill_norm": 4.0,
    "horizon_seconds": 1.0,
    "retention_norm": 0.0
  }
]

The complete persistence-filled operator can amplify errors even when retained history is bounded. All three position anchors use intrinsic-normalized coordinates. Source observation age is retained. Geometry channels and old stores are unchanged. A correlated synthetic covariance identity includes coefficient-anchor cross terms; real joint cross covariance is unavailable, so independent covariance propagation is not asserted.

## Verification and remaining limitations

{
  "covariance_cross_terms_verified": true,
  "frozen_track1_unchanged": true,
  "independent_reference_count": 0,
  "minimum_sampled_available_gib": 21.86138916015625,
  "past_timestamps_verified": true,
  "preserved_v6_files": 497,
  "real_gradient_fits": 5,
  "sampled_peak_tree_rss_gib": 3.3759765625,
  "selected_hand_classes_verified": true,
  "synthetic_fits": 20,
  "thresholds_and_transforms_train_only": true
}

The original v6 source/artifact hashes are checked before and after this run. Stage receipts cover configuration, code dependencies, input artifacts, runtime versions and outputs; partial stages retain their dependency journal. Scratch remains under work/track2_signal and no RGB/source observation payload is duplicated.

The pending blinded physical-motion/contact/identity queue still requires independent reviewers. Automated source diagnostics cannot manufacture those labels. No fresh participant evaluation or scientific finalization has occurred. See metrics.json and per-participant evidence before drawing conclusions from aggregate scores.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=work/track2_signal/matplotlib \
 /opt/anaconda3/bin/python3.13 -m track2_signal.run --stage all
```

Versioned stages: prepare, controls, memory, probes, summarize, verify, report. Resume rejects changed inputs, code or output hashes. Resource guards retain the 20 GiB tree-RSS ceiling and 5 GiB minimum available memory.
