# Assessment of the new Track 1 / Track 2 research report

The report's conditional next step has already been executed: the bounded Track 2 feasibility experiment completed and failed its mandatory short-horizon gates. The appropriate branch of the supplied plan is **HOLD full Track 2 and return to focused Track 1 work**. Track 1 remains scientifically unfinalized. No candidate is promoted.

This assessment adds a reproducible audit of the report's proposed 5% non-regression and 80% segment criteria using existing locked predictions. It does not retrain a model, repeat validation inference, change the original decision or open fresh participant labels. These additional criteria are **post-hoc**, not retroactive preregistration.

## What was implemented

`src/track2_assessment.py` checks the locked training source, model/adapter and frozen Track 1 hashes. It validates every prediction against the opportunity ledger, recalculates position/velocity errors, requires identical truth across all models, rejects duplicate or dropped records, and preserves unavailable truth as null. It reuses the existing independent-reference validator.

The outputs are [assessment.json](assessment.json), [participant metrics](participant_metrics.parquet), [episode metrics](episode_metrics.parquet), and [current independent-reference audit](independent_references.json). Parquet outputs are uncompressed. Existing source data, checkpoints and scientific reports are not overwritten.

## Results under the report's additional criteria

The margin is at most 5% worse than CV in **each** of position and velocity at **both** 250 and 500 ms. Aggregate errors first compute RMSE per participant, then average participants. The one-second horizon remains diagnostic here; the report does not specify a fourth horizon unambiguously.

| Family | Seeds passing aggregate motion non-regression | Episodes passing both horizons, per seed | Proposed 80% episode criterion |
|---|---:|---:|---|
| Plain Koopman | 0 / 3 | 1 / 5 for each seed | Fail |
| Stable Koopman | 0 / 3 | 1 / 5 for each seed | Fail |
| GRU | 0 / 3 | 0 / 5 for each seed | Fail |
| MLP | 0 / 3 | 1 / 5 for each seed | Fail |

An episode here is a source RGB context identified by participant/video/episode ID, pooling its matched entity histories. This is an explicit post-hoc interpretation of the report's undefined “segment.” It is not a claim about independently annotated physical episodes. All five contexts have some targets at both horizons, but not all histories have targets. Per-horizon coverage below retains those exclusions. Passing episodes are reported against both scored and all proposed episodes; the audit cannot improve the apparent result by dropping an unsupported episode.

The original stable Koopman result remains 29.3% worse than CV at 250 ms, 20.5% better at 500 ms and 45.1% better at one second. None of the twelve runs satisfies even the new report's motion-only non-regression requirement. Event/contact/identity/forecast-uncertainty constraints remain unavailable, so full Pareto dominance cannot be assessed, regardless of proxy improvements.

There are **two exploratory validation participants and five source contexts**, not hundreds of independent validation examples. The 236 proposed histories overlap. No t-test, Wilcoxon or McNemar significance claim is made from those overlapping histories. No regime labels are inferred from target motion to select a winner.

| Participant | Proposed histories | Input-supported | 250 ms targets | 500 ms targets | 1,000 ms targets |
|---|---:|---:|---:|---:|---:|
| P08 | 149 | 144 | 130 | 118 | 94 |
| P18 | 87 | 80 | 73 | 66 | 52 |
| Total | 236 | 224 | 203 | 184 | 146 |

Missing reasons include absent future states, unsupported future kinematics and unsupported input histories. These are source-derived eligibility diagnostics, not a measurement of independently defined 99% dataset coverage.

## Corrections needed before using the supplied report as an implementation plan

| Report statement or proposal | Current evidence / required correction |
|---|---|
| Forecasting is untested; begin minimal feasibility | Completed: 12 primary runs, 60 training-only stopping fits and a deterministic repeat. The original gate decision is NO-GO. Repeating it with new hyperparameters would be a new exploratory experiment. |
| Four coefficients / 16 coefficients / 32 controls | Keep temporal and spatial capacities distinct: four temporal coefficients; M16 has 16 spatial controls and 35 state channels; M32 has 32 allocated spatial controls and 67 state channels. |
| Train on six/eight participants or P02–P22 | Actual training participants are P02, P03, P04, P11 and P22. P08/P18 are repeatedly used exploratory validation. Participant-number ranges must not define a split. |
| Final test e.g. P23/P24 | Existing reservation metadata names P01/P05/P07; acquisition, independent labels and final locked-set status remain false. Metadata reservation does not establish an acquired test set. |
| Track 1 states are independent motion references | They are retrospective source-derived positions and conditional, uncalibrated kinematics. They cannot certify physical velocity/contact/identity. Qwen review remains machine-derived evidence. |
| Reconstruction IoU/BF2 passes fresh segmentation gates | Reconstructing supplied contours is not unseen RGB segmentation or fresh physical-boundary validation. Historical BF2 uses annotation-grid pixels, not automatically native-image pixels. |
| Enlarge topology sampling to guarantee validity | Existing interval/Bernstein certificates bind the canonical decoder. Denser sampling cannot replace a proof. Unresolved certification and source completeness remain explicit separate outcomes. |
| Approximately 17% complete-state yield | 3,485/19,974 was an older mixed-model numerical-support count, not completeness. M16 and M32 have different support; context extension and future-target support have different denominators. |
| Switching event recall is the latest detector result | A separate innovation channel already exists. At 250 ms synthetic validation onset recall is 0.471 versus switching 0.235 and CV 0.588; false onsets per quiet minute are 1.186 / 0 / 2.371 respectively. These are synthetic diagnostics. |
| At most 5% false alarms | Specify a denominator: false alarms per quiet minute differs from quiet-frame false-positive fraction. The report's percentages cannot be applied to the existing rate without a new preregistered event protocol. |
| Extend 95% participant conformal guarantees with more windows | Five independent calibration participant units cannot produce a finite 95% split-conformal rank threshold under the stated construction. More overlapping windows do not increase that count. |
| Exclude missing M32 geometry / occlusion | M16 carries continuity. Optional M32 and appearance require explicit availability masks and matched coverage/quality reporting, not silently changing the primary cohort. |
| Use event/contact consistency loss now | Independent targets are absent. Such loss terms would need explicit weak-label status or reliable references; they are not added to the frozen primary experiment. |
| Early-stop on P08/P18; later merge all train/val | Original stopping uses training-participant LOPO. Do not retroactively tune on the already evaluated participants or combine them before a new protocol and a real locked test set exist. |
| GitHub Actions and full clean rebuild are verified | CPU reload/repeat and a 75-artifact bounded Track 1 rebuild are verified. Full raw-pipeline reproduction and CI execution are not established by those checks. |
| Timeline lists independent labels as done | The current reference audit finds zero accepted independent references. Calendar labels do not substitute for completed evidence. |

## Next scientific milestone

1. **Measurement and reference quality:** retain the existing blinded review protocol, explicit uncertain/unscorable outcomes and provenance. Establish independently reviewed motion anchors, contact boundaries and identity through occlusion. Review disagreement/reliability before choosing physical-motion thresholds. Machine labels can support development but cannot satisfy this independent gate.
2. **Focused Track 1 causal pose diagnosis:** the existing target-informed centroid diagnostic explains a substantial share of endpoint geometry error. Any subsequent estimator experiment should retain M16 continuity, separate history reconstruction from causal motion and event evidence, and measure current pose, velocity, delay, quiet-minute alarms and continuation together. Define its training-only selection and stopping rules before launching it.
3. **Coverage and optional geometry:** preserve all missing-hop/epoch opportunities; report full-epoch, latest-epoch and missing-M32 strata using verified source metadata. Evaluate contact-relevant geometry only where independent reference geometry exists. Missing references must remain unavailable, not fabricated zeros or inferred physical labels.
4. **Fresh evaluation:** obtain the reserved independent participants and preregister their use only after a candidate and references are ready. Reopening exposed validation, changing dimensions or adding appearance/M32 ablations is not the next automatic stage after the failed feasibility gate.

No new model selection, reference acquisition, dataset duplication, final freeze or full Track 2 training is performed by this assessment. It delivers the report reconciliation and computable evaluation changes; independent annotation and a full raw-pipeline rebuild remain outstanding work.

## Reproduce

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m track2_assessment
```

This reads the saved forecast ledger and current development reference submissions. It does not invoke training or model inference. The output records input hashes and the assessment-code hash. Regression tests cover lost/duplicated records, changed truth, zero-imputed missing targets, incorrect error fields, perfect-baseline handling and unavailable reference gates.
