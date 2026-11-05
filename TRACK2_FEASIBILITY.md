# Isolated Track 2 feasibility experiment

The bounded experiment is complete. **Full Track 2 remains NO-GO.** Stable Koopman fails the mandatory 250 ms and seed-regret gates. No optional ablation, latent-size increase, production training or Track 1 promotion was started. All frozen Track 1 hashes remain unchanged.

Open [the dashboard](reports/track2_feasibility/dashboard.html), [gate decision](reports/track2_feasibility/decision.json), [all forecasts](reports/track2_feasibility/predictions.parquet) and [final audit](reports/track2_feasibility/final_audit.json). Model checkpoints and the immutable adapter are in `derived/track2_feasibility`. The historical `track2_started=false` fields in frozen Track 1 releases remain unchanged; this separate experiment has its own completion status.

## Approved protocol amendment

The supplied decision report prescribed base-v2 M16 length-four histories, quoting 396 training and 63 validation sequences. Inspection found those totals include M32. Actual M16 counts are **319 / 49**. Under strict episode/epoch/hop and causal-kinematic support checks, base-v2 has only 44 training targets at 250 ms, zero validation future targets, and no 500 ms or 1 s targets. Training on that cohort would not answer the report's question.

The user explicitly approved using the existing frozen six-second `derived/track1_context_v3` cohort after the support audit. The same five training participants and two exploratory validation participants, four model families, seeds, latent size, optimizer and gates were retained. The three newly added revision contexts were **not** used. No fresh reserved participant was accessed.

| Frozen context cohort | All M16 histories | Input-supported histories | 250 ms targets | 500 ms targets | 1,000 ms targets |
|---|---:|---:|---:|---:|---:|
| Training | 1,081 | 1,047 | 959 | 874 | 722 |
| Exploratory validation | 236 | 224 | 203 | 184 | 146 |

All original opportunities and missing reasons remain in `target_ledger.parquet` and `availability_audit.parquet`. Input-supported terminal histories without future targets stay in the adapter. Training excludes batches with no supported target; evaluation still emits forecasts for all input-supported histories and marks unavailable truth explicitly. Invalid-input histories remain in the denominator ledger. No unavailable target is treated as an observed zero.

## Fixed implementation

`src/track2_feasibility` contains data adaptation, models, training, evaluation and the controller. `configs/track2_feasibility.json` is the locked configuration. All scripts, fold records and operational logs remain in `work/track2_feasibility`.

Each of four historical states contains 157 input values: flattened coarse `(4,35)` coefficients (140), authoritative source-kinematic position (2), velocity (2), log-diagonal/lower-Cholesky covariance encoding (10), and three support flags. Substantially non-PSD covariance is rejected; small eigenvalues receive a recorded numerical floor. Only fully supported primary input histories are admitted, so support flags are explicit rather than substitutes for missing observations. No appearance, detail, optional M32 geometry, event, identity or contact label enters the primary model.

Continuous input and target channels are standardized using unique states from the relevant **training participants only**. Each stopping fold fits its own moments excluding the held training participant. Constant-channel floors are declared engineering safeguards. The source covariance is conditional and uncalibrated; it is not a future-uncertainty target.

All learned models output position, velocity and 140 coarse coefficients for 250/500/1,000 ms. Forecast endpoints lie beyond the input cutoff. Future states must remain in the same source epoch/episode with every intervening quarter-second state supported. Future coarse histories naturally overlap previously observed times; their position/velocity endpoints are future targets. Dense annotation availability remains retrospective, so this is a frozen source-state experiment, not an online RGB forecast benchmark.

The fixed architectures are:

- Plain Koopman: four-state observation encoder `628 -> 64 -> 32`, one unrestricted continuous generator and shared state decoder `32 -> 64 -> 144`.
- Stable Koopman: the same encoder/decoder, with `A=(J-R)H`, `J=(K-K.T)/2`, `R=L@L.T+1e-4 I`, `H=B@B.T+1e-4 I`.
- GRU: one layer, 32 hidden channels, direct three-horizon output head.
- MLP: flattened history, hidden widths 64/32, direct three-horizon head.
- CV: `p+h*v`, velocity copied, coarse coefficients copied, no learned parameters.

The encoder/decoder details above are fixed implementation choices, not a validation search. There is no CV residual skip. Koopman uses exact `matrix_exp(h*A)` during training. Inference caches the three propagators after weights are fixed; training invalidates the cache. All models remain float32; metrics aggregate in float64.

The locked objective is `Huber(position) + .25 Huber(velocity) + .25 standardized_MSE(coarse) + .05 latent_L2`. The MLP has no latent penalty. Sequence weights balance training participants; statistics aggregate participant RMSE before averaging. Shared source fitting was already frozen across the five training participants: these LOPO stopping fits concern the dynamics model, not a complete refit of Track 1 per fold.

## Training budget and learning checks

Seeds are **17, 43, 89**. AdamW uses learning rate `3e-4`, weight decay `1e-4`, batch size 32 and gradient clipping 1. Maximum stopping-fold budget is 200 epochs with patience 25. For each model/seed, five leave-one-training-participant-out stopping fits select the median best epoch. The final model then refits on all five training participants for that count. The two exploratory validation participants never select epochs or normalization.

This executes **12 primary final runs plus 60 prescribed stopping-fold fits**. It does not add model variants. Final selected epoch counts are:

| Family | Seed 17 | Seed 43 | Seed 89 |
|---|---:|---:|---:|
| Plain Koopman | 29 | 30 | 27 |
| Stable Koopman | 28 | 30 | 29 |
| GRU | 83 | 63 | 78 |
| MLP | 21 | 24 | 18 |

Training objectives decreased; per-epoch training/stopping losses, participant assignments and initial/final weight hashes are archived. Every final checkpoint reload reproduces predictions exactly on the checked CPU inputs. A fresh repeat of the best observed checkpoint, MLP seed 89 at 18 epochs, reproduced its complete weight hash bit for bit. The repeat is a reproducibility check, not a new model-selection or validation run.

## Matched results and stop decision

The table uses the mean of the two participant RMSEs followed by the median across seeds. Units are intrinsic-normalized two-dimensional source positions, not native pixels or metric physical motion. There is one locked exploratory-validation forward pass per final model/seed.

| Family | 250 ms position RMSE | 500 ms | 1,000 ms |
|---|---:|---:|---:|
| CV | 0.13484 | 0.25960 | 0.46500 |
| Plain Koopman | 0.17713 | 0.20702 | 0.24956 |
| Stable Koopman | 0.17437 | 0.20633 | 0.25509 |
| GRU | 0.24156 | 0.27034 | 0.31256 |
| MLP | 0.16600 | 0.20083 | 0.24761 |

Stable Koopman is **29.3% worse at 250 ms**, **20.5% better at 500 ms**, and **45.1% better at 1 s** than CV. Its median velocity errors are lower than CV at all three horizons. All three stable generators have negative spectral abscissae (approximately -0.048 to -0.057); the plain generators have positive modes. Architectural latent stability does not certify physical forecasts or contour geometry.

The mandatory 250 ms gate fails, including its participant-bootstrap alternative. The worst stable seed is 34.3% worse than CV at 250 ms, failing the 20% seed-regret limit. The 500 ms, bounded 1 s, velocity and stable spectral checks pass. No nonfinite forecast occurred, and the additional predeclared catastrophic pointwise-error check passes. The overall decision is therefore **NO-GO**, not an averaged success across horizons.

There are only two exploratory validation participant units. The participant bootstrap does not turn overlapping histories into independent observations or establish final confidence. All learned model medians lose to CV at 250 ms. Longer-horizon source-state gains show some predictable structure but do not support preferentially selecting Koopman, claiming physical dynamics, increasing latent dimension, or beginning the full Track 2 programme. The next research priority returns to Track 1 causal pose/measurement quality and independent reference coverage.

## Inference and resources

On a synthetic fixed-model fixture, one CPU thread took about 1.31 ms per stable-model training step versus 1.44/1.50 ms for two/four threads. The installed Metal backend rejects `linalg_matrix_exp`; training therefore uses 12 single-thread CPU workers without GPU fallback transfers or reduced precision. This is a small-state workload, not raw-video neural training.

Warm batch-one inference includes normalization, all three horizon predictions and denormalization. It measures cached M16 state input, excluding RGB, segmentation, appearance and Track 1 encoding:

| Model | Typical CPU median |
|---|---:|
| MLP | 0.034 ms |
| Plain/stable Koopman | 0.062–0.063 ms |
| GRU | 0.084–0.085 ms |

No end-to-end camera FPS claim is made. Sampled process-tree RSS peaked at **7.76 GiB**; sampled available system memory stayed above **21.48 GiB**. Guards retain the 20 GiB process-tree ceiling and 5 GiB system reserve. No OOM occurred. RSS includes shared pages and is not unique physical allocation.

## Verification and reproduction

All **113 tests** pass. New checks cover stable-generator dissipativity, cached/uncached parity, missing-target loss gradients, safe covariance encoding, source epoch/hop guards and training-only normalization. The final audit verifies 8,736 forecast/mask records with identical supported target sets for every model, exact checkpoint/adapter hashes and unchanged Track 1 hashes. Reload and best-checkpoint repeat checks pass. Independent-reference, contact, event, identity, forecast uncertainty and fresh-participant gates remain unsatisfied and were not used for fitting.

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -u -m track2_feasibility.run --stage all --resume
```

A completed matching-source run resumes by verifying frozen hashes and reusing checkpoints; it must not reopen validation inference. The evaluation lock refuses a second evaluation. Code/config changes require an explicit new protocol/root, not deleting that lock to tune against validation. Fold artifacts belong to this locked protocol and must not be reused across a changed configuration. No automatic follow-up or optional ablation is authorized by a completed feasibility flag.

The report's quoted public commit is recorded as reported provenance; the actual local commit, environment, dependency lock and source hashes are separately recorded. No unverifiable commit identity is manufactured.
