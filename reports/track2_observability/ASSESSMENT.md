# Exactly anchored dynamics and observability

Bounded development only. Original v5 artifacts are preserved. No promotion, scientific freeze, full-scale Track 2 launch or Track 3 work.

## Completed stages

A: Epoch zero is evaluated before training, is eligible for stopping selection, and an explicit zero-epoch refit performs no updates. The four original anchor/loss controls are rerun in this new release. All complete finite observation candidates inside the 10 ms tolerance are audited separately from the nearest candidate; original target labels are unchanged. Every CLI stage checks code, imported local numerical helpers, input artifacts, configuration, runtime versions and output hashes before reuse. Verification and report generation are versioned stages.

B: A source-displacement predictability audit reports direction, magnitude, participant/episode variation and training-derived low/middle/high displacement strata. Physical quiet, steady-motion, onset, reversal, occlusion and interaction strata remain unknown. No blinded reviews were manufactured; the pending v5 queue remains pending. Source strata cannot substitute for independent event labels.

C: Exactly anchored continuous models evolve position, velocity and a small latent acceleration state through an augmented matrix exponential without inverting its generator. Linear and dissipative residual generators are compared with a small polynomial-acceleration decoder, nearest, shrinkage and the original four controls. Known source-anchor position bypasses the latent decoder. Zero-horizon anchoring and position-velocity derivative consistency are tested before training. These properties do not certify physical correctness of the source anchor. Models are small but parameter counts differ; no exact capacity-matching claim is made.

D: A separate uniform-Legendre overlap operator forgets departing history, retains only overlap, and fills the incoming interval with persistence. It is compared with copy, full polynomial extension and a train-only ridge correction to the incoming value. The four-coefficient interface is preserved. This is a diagnostic approximation for the actual irregular weighted store, not its exact update. Future endpoint errors are reported separately from cached kinematic targets.

E: Same-width linear probes compare object history, measurement age/count/quality, hand missingness, nearest-hand relative position/velocity and a deterministic past-time shuffled hand control. No graph network is introduced. All probes include a zero-correction selection candidate. The cohort includes hands as well as objects, so this is an all-entity source-information test. Relations use provisional annotation identities, explicit availability and only timestamps preceding each query. Retrospective annotation availability remains unverified. Shuffling stays within each query's four past samples; it does not borrow future observations.

## Matched source-position results

| Method | 250 ms | 500 ms | 1 s |
|---|---:|---:|---:|
| nearest | 0.09497 | 0.14129 | 0.18135 |
| shrinkage | 0.09412 | 0.14138 | 0.18135 |
| cv_joint | 0.10326 | 0.15219 | 0.21795 |
| cv_position | 0.10331 | 0.15229 | 0.21900 |
| nearest_joint | 0.09497 | 0.14129 | 0.18135 |
| nearest_position | 0.09497 | 0.14129 | 0.18135 |
| anchored_linear | 0.09497 | 0.14129 | 0.18135 |
| anchored_stable | 0.09497 | 0.14129 | 0.18135 |
| anchored_polynomial | 0.09497 | 0.14129 | 0.18135 |

Median seed of mean-participant RMSE, intrinsic-normalized coordinates. Same 221 validation histories; supported targets [201, 183, 146]; common all-horizon histories 146. All selection uses five training participants. The two validation participants are repeatedly exposed and remain exploratory. Check individual seed and participant scores in results.json and per_participant.parquet; overlapping windows are not independent replicates.

Selected final epochs: [['cv_joint', 17, None, 126], ['cv_joint', 43, None, 119], ['cv_joint', 89, None, 135], ['cv_position', 17, None, 126], ['cv_position', 43, None, 119], ['cv_position', 89, None, 126], ['nearest_joint', 17, None, 5], ['nearest_joint', 43, None, 0], ['nearest_joint', 89, None, 0], ['nearest_position', 17, None, 5], ['nearest_position', 43, None, 0], ['nearest_position', 89, None, 0], ['anchored_linear', 17, None, 0], ['anchored_linear', 43, None, 0], ['anchored_linear', 89, None, 7], ['anchored_stable', 17, None, 0], ['anchored_stable', 43, None, 0], ['anchored_stable', 89, None, 7], ['anchored_polynomial', 17, None, 0], ['anchored_polynomial', 43, None, 0], ['anchored_polynomial', 89, None, 7]].

## Findings from this run

Epoch zero wins for two of three seeds in every exactly anchored family. Their median forecasts therefore equal nearest observation; this is a valid no-update outcome, not evidence of learned dynamics. The nonzero seed remains separately visible in results.json. Relational probes select zero correction in every arm, including shuffled and missingness controls, so this experiment does not demonstrate predictive relational information.

Overlap persistence slightly improves short-horizon future functional endpoints compared with copying, and avoids the large polynomial-extension failure. The learned incoming correction still regresses at one second. Neither gain establishes physical motion or source-position forecasting improvement. The expanded any-usable target search produces the same counts on this dataset; its regression fixture now covers incomplete-nearest alternatives.

## Target candidates

[
  {
    "any_usable": 918,
    "horizon_us": 250000,
    "nearest_usable": 918,
    "proposed": 1081,
    "rejected_with_any": 15,
    "rejected_with_nearest": 15,
    "split": "train"
  },
  {
    "any_usable": 840,
    "horizon_us": 500000,
    "nearest_usable": 840,
    "proposed": 1081,
    "rejected_with_any": 18,
    "rejected_with_nearest": 18,
    "split": "train"
  },
  {
    "any_usable": 703,
    "horizon_us": 1000000,
    "nearest_usable": 703,
    "proposed": 1081,
    "rejected_with_any": 22,
    "rejected_with_nearest": 22,
    "split": "train"
  },
  {
    "any_usable": 187,
    "horizon_us": 250000,
    "nearest_usable": 187,
    "proposed": 236,
    "rejected_with_any": 0,
    "rejected_with_nearest": 0,
    "split": "validation"
  },
  {
    "any_usable": 169,
    "horizon_us": 500000,
    "nearest_usable": 169,
    "proposed": 236,
    "rejected_with_any": 0,
    "rejected_with_nearest": 0,
    "split": "validation"
  },
  {
    "any_usable": 135,
    "horizon_us": 1000000,
    "nearest_usable": 135,
    "proposed": 236,
    "rejected_with_any": 0,
    "rejected_with_nearest": 0,
    "split": "validation"
  }
]

Nearest usability and any-usable-within-tolerance are distinct. Alternative positions and complete-candidate disagreement remain diagnostic fields. The original source-state target policy and v5 findings are not overwritten.

## Verification and interpretation

{
  "deterministic_stable_refit": true,
  "epoch_zero_checkpoint_present": true,
  "final_fits": 21,
  "fold_fits": 105,
  "frozen_track1_unchanged": true,
  "hand_inputs_past_query_cutoff": true,
  "held_participant_excluded_from_normalization": true,
  "minimum_sampled_available_gib": 22.275772094726562,
  "preserved_v5_manifest_files": 70,
  "promotion": false,
  "sampled_peak_tree_rss_gib": 5.3453826904296875
}

The completed test log is work/track2_observability/tests_final.log. Tests and integrity receipts provide engineering evidence. They cannot establish independent event preservation, online annotation availability or fresh-participant generalization. Training uses source-derived targets. No full joint uncertainty or calibrated physical velocity is claimed. CPU inference measurements exclude upstream RGB/Track 1 and are not deployment latency.

The retained-history operator is bounded in the ideal uniform coefficient L2 metric; the full polynomial-extension operator can amplify coefficient error. Neither mathematical statement guarantees better video forecasting. See overlap_memory.json for synthetic checks, operator norms, training selection and source endpoint scores.

Weak-form or latent-consistency training has not been added in this cycle. The current experiment first isolates initialization, memory semantics and input information as requested. Independent reviewers are still needed to complete the blinded physical-event review and eventual fresh evaluation.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=work/track2_observability/matplotlib \
 /opt/anaconda3/bin/python3.13 -m track2_observability.run --stage all
```

Stages: prepare, train, evaluate, memory, relations, verify, report. Resume refuses changed dependencies or changed outputs. An interrupted stage validates completed job receipts before reusing them. No source observations or RGB files are duplicated; scratch and resource logs remain under work/track2_observability.
