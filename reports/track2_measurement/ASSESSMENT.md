# Track 2 measurement and anchoring experiments

Open development. No candidate promotion, full Track 2 approval, scientific freeze or Track 3 expansion.

The supplied report's first three experiments are implemented and run: target/evidence audit, matched nearest/CV anchors with ridge and train-only velocity shrinkage, and position-only versus joint position/estimated-velocity training. The next phase adds exact polynomial reindexing plus ridge innovation and read-only diagnostics of the existing Koopman models. Region moments replace sampling-dependent boundary moments in a new separately tested helper; frozen geometry features are unchanged.

## Matched forecast results

| Method | 250 ms | 500 ms | 1 s |
|---|---:|---:|---:|
| nearest | 0.09497 | 0.14129 | 0.18135 |
| shrinkage | 0.09412 | 0.14138 | 0.18135 |
| decay | 0.09417 | 0.14165 | 0.18186 |
| ridge | 0.09522 | 0.14184 | 0.18209 |
| nearest_joint | 0.09504 | 0.14138 | 0.18149 |
| nearest_position | 0.09504 | 0.14138 | 0.18149 |
| cv_joint | 0.10326 | 0.15219 | 0.21795 |
| cv_position | 0.10331 | 0.15229 | 0.21900 |
| cached_cv | 0.13518 | 0.26000 | 0.46500 |

Errors are intrinsic-normalized source-position RMSE, averaged over participants, then median across seeds for neural arms. These are retrospective annotation/estimator targets, not independently measured physical motion. All methods use the same 221 validation histories and 201/183/146 supported targets. The original cohort has 224 histories and 203/184/146 targets; three validation histories fail the new recent-observation support requirement. No training histories are excluded. `input_exclusions.json` lists them. The common all-horizon cohort has 146 histories; its separate scores are in `results.json`. Do not compare this table directly with the old 224-history table. `prior_matched_comparison.json` rescores frozen predictions on the same subset.

The nearest anchor reduces the error of the learned CV-anchor model, but neither nearest-residual neural arm improves reliably over nearest. Position-only loss produces nearly the same position results as joint training. Train-only velocity multipliers are [0.06833333493362334, 0.010941329222791892, 0.0]; their improvement at 250 ms is below one percent and does not establish a useful multi-horizon gain. Ridge selected regularization 100; decay selected beta 64. Every declared position gate fails. No independent event labels exist to test preservation of event evidence.

All four neural arms receive identical four-hop inputs: cached position/velocity, recent observation position, age and count. Positions are relative to the most recent observation. Training-only normalization, initialization, minibatch policy, network size and stopping criterion are matched. There are 60 leave-one-training-participant-out fits and 12 final refits. The position-only arms leave the velocity residual head untrained: that output remains anchor velocity and is not a newly learned physical-velocity estimator.

## What the target audit establishes

The full original proposal ledger remains 1,081 training and 236 validation histories. Rejection reasons are retained per horizon. Among rejected validation targets, the audit finds zero complete source-mask centroids within the declared 10 ms target tolerance. It finds 15/18/22 in training. These are annotation-derived observations, not independent reference positions. This result does not prove hard events are retained: physical onset/reversal/occlusion labels remain unknown.

Source-topology change is recorded as an annotation-graph diagnostic, not a physical event. `source_strata.parquet` and `per_episode.parquet` preserve denominators and participant/episode variation. No confidence interval treats overlapping windows as independent. A 48-task blinded review queue contains no predictions or retained/rejected labels; the sampling key stays under work. It has not been reviewed.

Every audited input lacks established complete evidence ancestry/availability. Timestamp filtering alone does not establish online causality. The audit fails closed on online eligibility. A dependency group containing later labels is recorded as a warning about unresolved ancestry, not proof that every earlier annotation used them.

## Polynomial memory and frozen-model diagnostics

The exact cubic Legendre shift has synthetic polynomial and composition tests. It extends the existing polynomial; it is not an exact update of a new irregularly sampled, weighted, regularized window fit. Pure reindexing is worse than coefficient copying on the source windows. Training-only ridge innovation improves decoded overlapping-history position at 250/500 ms but worsens it at one second. `memory_endpoint.json` separately evaluates future functional endpoints: the ridge correction is worse than coefficient copying at all three horizons. The overlapping-history gain therefore does not establish forecast improvement. These functional-position targets differ from authoritative cached kinematic positions; the metrics must not be mixed. Decoded shape errors concern normalized curve coordinates, with same-epoch source targets and no transferred future geometry certificate. The ridge has 59,220 weights, so this does not demonstrate a lower-parameter advantage.

`koopman_diagnostic.json` measures zero-horizon reconstruction, future-encoding consistency, and decoded position derivative versus velocity, on frozen models. Zero-horizon position RMSE is approximately 0.142–0.149. Stable-generator metric condition numbers are approximately 5–6; sampled propagator norms exceed one even while the metric dissipation inequality holds. Energy stability therefore does not imply Euclidean contraction or physical accuracy. No new latent-consistency objective has been trained.

The region-moment helper integrates occupied polygon area and subtracts holes. Tests cover orientation, translation, holes and nonuniform edge subdivision. The same rectangle has different old boundary-point covariance under resampling, while its new region covariance is unchanged. This fixes the demonstrated descriptor sensitivity without claiming it explains the earlier geometry model's failure.

A nonlinear synthetic composition records GL4/8/16 quadrature against 128-node diagnostics and the integration-by-parts identity. This is a numerical check, not a trained weak-form objective ablation.

## Verification and resources

148 tests pass. A repeated final training fit reproduces its weight hash exactly. Fold normalization excludes each held participant. Initial weights match across all four arms. Nearest predictions match the frozen baseline within 2.82e-08. Verified 464 prior source/artifact hashes, plus the frozen Track 1 manifest. Sampled peak training process-tree RSS was 6.22 GiB; minimum sampled available memory was 23.63 GiB. No OOM occurred. Logs are under `work/track2_measurement`; no source observations or RGB files were duplicated.

`latency.json` measures the CPU cached-history path, including the current whole-array feature helper overhead. It excludes RGB processing and Track 1 encoding and is not an optimized deployment benchmark.

A diagnostic-only channel-index bug was corrected before reporting: M16 shape occupies channels 2:34 and log-scale occupies channel 34. The discarded pre-fix diagnostic is retained under work. A regression test ensures position/log-scale cannot contaminate normalized shape error. No neural forecast model or forecast evaluation changed.

## Remaining research scope

The supplied report is an ordered research program. This revision completes its measurement/anchor/loss cycle and the bounded diagnostics above. It does **not** implement or claim acceptance of a structured residual Koopman model, a new neural innovation model, latent-consistency training, weak-form versus derivative training, relational geometry, new physical labels, or a final fresh-participant evaluation. Those remain separate experiments; no broad architecture search or automatic promotion was launched after these negative gates.

Before stronger scientific claims, obtain independent blinded motion/event/identity references and resolve annotation availability. A future model may be tested on source targets, but must retain this restricted interpretation. Machine reviews are not independently adjudicated physical ground truth.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m track2_measurement.run --stage all
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m track2_memory
```

The first command resumes hash-locked training checkpoints and reuses completed evaluation; it needs process-tree inspection for resource guards. The second reuses its completed diagnostic. Final evidence generation and verification scripts are preserved in `work/track2_measurement`.
