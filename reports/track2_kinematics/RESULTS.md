# Controlled kinematics-only prototype: results and report reconciliation

The new report's missing simplest-model comparison is complete. **The learned models beat constant velocity, but neither beats constant position across the required horizons. Full Track 2 remains on hold.** The prior feasibility experiment and Track 1 releases remain unchanged.

Open [the dashboard](dashboard.html), [gate decision](decision.json), [prediction ledger](predictions.parquet), [support audit](pairing_audit.json) and [verification](verification.json).

## Scope and protocol

This is a new, isolated exploratory protocol, motivated by the report's kinematics-only GRU/MLP and constant-position baselines. The earlier GRU/MLP used full functional-state inputs; it did not answer this exact question. The configuration and code hashes were recorded before new training in [protocol_lock.json](protocol_lock.json). P08/P18 validation was already exposed in prior work; prerecording this extension does not make it a fresh evaluation.

Implemented priorities:

| Report priority | Outcome |
|---|---|
| A: CV and constant-position/nearest-state baseline | Evaluated on exactly the same supported future targets; CV forecasts match the original experiment exactly. Constant position holds the latest source-kinematic position with zero velocity. Nearest-state continuation has the same position forecast; it is not counted as another independent baseline. |
| B: small kinematics-only predictors | MLP and GRU trained with three fixed seeds each, plus 30 training-participant stopping fits. |
| C: stable functional latent model | Already implemented and evaluated in the original feasibility experiment. Its failed short-horizon gate is retained; no new latent/ODE search is launched. |
| D/E: geometry/contact and probabilistic variants | Conditional follow-ups are not started: neither simple learned model passes the new proxy gate, and independent scientific references remain absent. |
| Event/identity/contact/calibration evaluation | Existing reference audit rerun: zero accepted independent references. These metrics remain unavailable; no contact truth is inferred from mask overlap or Qwen output. |

The new models receive only four consecutive `(x,y,vx,vy)` states. Their endpoint timestamps span 750 ms at a 250 ms hop. The underlying Track 1 encodings can depend on earlier observations; four stored states should not be described as four independent observations or exactly one second of new evidence.

Positions are expressed relative to the latest input position. The network predicts corrections to CV at 250/500/1,000 ms; its output head is initialized to zero, so its initial prediction equals CV. Translation of all input positions translates the predicted positions equally and leaves velocity unchanged. These choices avoid relearning an absolute image origin and preserve a usable baseline at initialization. They do **not** guarantee that a trained correction improves CV.

The MLP has hidden widths 64/32; the one-layer GRU has hidden size 32. Both have twelve outputs: position and velocity at three horizons. No appearance, M32, functional coefficient, covariance, residual/detail or event feature enters either model. There is no uncertainty prediction or shape forecast.

The objective is training-normalized residual-position MSE plus 0.25 times residual-velocity MSE, with explicit horizon masks and participant-balanced example weights. Input/output RMS scales are learned from each training fold only; zero corrections are not mean-shifted. No training target is filled in as an observed zero. Relative positions, a CV residual parameterization and a different loss accompany the smaller input; **differences from the original full-state models cannot be attributed solely to removing functional coefficients**.

AdamW uses learning rate 0.0003, weight decay 0.0001, batch size 32 and gradient clipping 1. Each seed uses five leave-one-training-participant-out stopping fits, maximum 200 epochs and patience 25. The median best epoch selects the all-five-participant final refit. Seeds are 17, 43 and 89. Final epoch counts are 60/44/58 for MLP and 133/119/120 for GRU. Training losses, stopping curves, initialization/final weight hashes and checkpoints are retained. Training is short because these are small networks operating on cached state arrays; raw-video/CNN processing is not repeated.

## Cohort and coverage

Training participants remain P02/P03/P04/P11/P22; exploratory validation remains P08/P18. P11 cannot become a fresh held-out participant after its earlier training exposure. The pre-existing context-v3 adapter is reused without modifying or duplicating source observations or RGB.

| Split | Input-supported histories | 250 ms targets | 500 ms targets | 1,000 ms targets |
|---|---:|---:|---:|---:|
| Training | 1,047 | 959 | 874 | 722 |
| Validation | 224 | 203 | 184 | 146 |

The source ledger contains 1,081 training and 236 validation M16 opportunities before input-support filtering. All excluded and unavailable opportunities remain in that ledger. Every method predicts all 224 supported validation histories at all three horizons, yielding 5,376 forecast/mask records across eight method/seed combinations. Truth and errors remain null at unsupported horizons. The exact-match audit rejects dropped records, duplicate forecasts, changed targets and zero-imputed missing truth.

Validation histories overlap within only **five RGB contexts from two participants**. They are not 224 independent episodes. Errors are computed per participant before averaging; the table below takes the median across three seeds. There is no window-level significance claim or independent-test confidence interval.

## Matched results

Units are intrinsic-normalized source position and source velocity per second. They are not native-pixel contact errors or independently measured physical trajectories.

| Method | Position 250 ms | Position 500 ms | Position 1 s | Velocity 250 ms | Velocity 500 ms | Velocity 1 s |
|---|---:|---:|---:|---:|---:|---:|
| Constant velocity | 0.13484 | 0.25960 | 0.46500 | 0.58592 | 0.64665 | 0.58258 |
| Constant position | **0.09912** | **0.14448** | **0.18252** | **0.42206** | **0.43463** | **0.45982** |
| Kinematic MLP | 0.10318 | 0.14993 | 0.21020 | 0.43578 | 0.44231 | 0.46874 |
| Kinematic GRU | 0.11118 | 0.16950 | 0.26137 | 0.45505 | 0.46049 | 0.49183 |

The MLP lowers position error versus CV by **23.5%, 42.2% and 54.8%** at the three horizons. Against constant position it is **4.1%, 3.8% and 15.2% worse**. Its velocity errors also remain above the zero-velocity baseline. The GRU has larger regressions against constant position.

Before training, the proxy advancement rule required median position RMSE at least 10% below **each** baseline at 250/500 ms, no more than 5% regression in velocity or one-second position, and no seed with more than 20% short-horizon position regression. Neither model passes. This is a development filter, not a replacement for the report's independent event/contact/identity/calibration gates. No model is promoted and no automatic richer-model training follows.

Source-speed diagnostics use quartile cutoffs learned solely from training input velocities. They are not physical-motion regime labels. In the highest source-speed bin, MLP position RMSE is 0.14561 versus constant-position 0.14208 at 250 ms, and 0.18859 versus 0.18894 at 500 ms. A near tie in one selected diagnostic bin does not change the aggregate decision. The per-participant denominators and errors are retained in [speed_strata.parquet](speed_strata.parquet).

Constant-position superiority shows that beating CV alone is an insufficient bar on this source-derived cohort. It does not establish that the real objects were stationary, identify the cause of every error, or demonstrate that functional representations are universally unhelpful. Measurement noise, source-centroid changes, stale estimates and limited context diversity still require independent checks.

## Necessary corrections to the supplied report

- **M16 dimensions:** the core tensor is `(4,35)`, with four temporal modes and 16 spatial controls. It is not “4–35 coefficients.”
- **M32 availability:** the quoted 436/284/84 training and 50/19/0 validation counts are complete-certified **M16** sequences in the readiness table, not M32 availability. The tiered context audit reports 104 training / five validation length-16 sequences with all optional compound geometry, and 99/five within one compound epoch. These are overlapping sequences, not independent episodes.
- **Position calibration:** the historical ~94% coverage result does not certify this causal kinematic stream or future predictions. Its covariance remains conditional and uncalibrated for this task. No calibration guarantee is transferred.
- **Splits:** P11 is already exposed. Existing fresh-participant reservation metadata names P01/P05/P07; those reservations are not an acquired, independently annotated locked set. No held-out data is added to training after inspecting failures.
- **Missing geometry:** M16 remains the carrier. An unavailable M32 view must stay explicit; replacing it with an unmarked last shape or excluding it from the denominator would alter the task.
- **Stability:** a continuous-time generator requires nonpositive real eigenvalue parts under the declared stability convention; a discrete transition requires spectral radius at most one. “Eigenvalues ≤1” is not a sufficient general complex-matrix statement. Neither criterion certifies decoded geometry or physical dynamics.
- **Acceptance rules:** the report gives conflicting contact timing limits (100/250 ms versus 150/300 ms) and different coverage ranges. Those need one future protocol chosen before independent evaluation. A nominal NLL “within 10%” is undefined without a target distribution, units and scoring convention. Ratios to the number of true identity switches are also ill-defined when that count is zero.
- **Reference integrity:** source-mask overlap is a projected relation, not physical contact ground truth. More correlated windows do not supply more independent calibration participants or repair final-test exposure.

## Runtime, verification and reproduction

The full-precision synthetic training benchmark measured CPU/Metal median step times of 0.43/2.12 ms for MLP and 0.71/3.83 ms for GRU, with synchronized Metal timing. Training therefore used up to 12 single-thread CPU workers. Metal was available and supported these models; CPU was faster for the actual small batch/network sizes.

Cached-state batch-one inference for all three horizons, including input normalization and adding the CV correction, took approximately **0.046–0.047 ms for MLP** and **0.077 ms for GRU**. CV took 0.0051 ms and constant position 0.0028 ms. These measurements exclude upstream RGB, segmentation, appearance and Track 1 construction. They do not certify a mobile-GPU or end-to-end 10 ms budget.

Sampled process-tree RSS peaked at **6.14 GiB**, and available memory stayed above **21.51 GiB**. The existing 20 GiB tree-RSS / 5 GiB reserve guards stayed active; no OOM occurred. RSS includes shared pages.

All **126 tests pass**. Six new tests cover exact initial CV behavior, translation equivariance, constant-position semantics, masked nonfinite targets, training-only normalization and kinematics-only input dimensions. Each checkpoint reload matched exactly on training inputs. A preselected repeat of MLP seed 17 reproduced the complete weight hash bit for bit. Model evaluation occurred once per new final run; verification did not repeat validation inference. Frozen Track 1 and the prior feasibility source/checkpoint/report hashes match.

From the project root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -u -m track2_kinematics.run --resume
```

Completed stages verify hashes and reuse their artifacts. A changed protocol/source or incomplete locked evaluation fails closed. Scratch scripts/logs/fold records live in `work/track2_kinematics`; models in `derived/track2_kinematics`; permanent implementation in `src/track2_kinematics`. Source arrays and RGB stay in their existing stores.

The next scientific dependency is independent measurement/episode quality and a controlled causal-pose diagnosis, with constant position retained alongside CV. Fresh independent motion/contact/identity labels, a properly specified uncertainty evaluation and an untouched final cohort remain outstanding. This bounded extension does not finalize either track.
