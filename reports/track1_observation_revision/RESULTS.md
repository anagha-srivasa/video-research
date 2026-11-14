# Track 1 audit-driven corrections

This is an open development experiment. No baseline promotion or universal superiority claim is justified. The full research architecture and independently annotated physical evaluation remain pending. The previously blocked official source/checkpoint acquisition has now succeeded; a separate full-precision Metal tracker evaluation is under way. See [the neural evaluation](../track1_neural_observation/README.md).

## What changed

- Geometry budget search now retains the best certified feasible incumbent. More permitted controls cannot discard that solution. Original coefficients and certificate bindings remain intact.
- A local material-point transport baseline replaces an increasingly implausible global rigid fit with adjacent-frame fits, while preserving point identities. A broken chain can restart only from a valid original-anchor fit. Covariance is a conditional approximation; support is not tracking accuracy.
- The synthetic benchmark crosses all five motion types, six corruptions and two durations in each split. Random orientation, amplitude, onset and noise are independent of that factorial assignment. The 60 training and 60 validation episodes use separate random seeds. These are simulated episodes, not independent human participants.
- Occlusion removes anchor observations, majority outliers reach the estimator, and coherent drift remains a deliberately unidentifiable nuisance. No corruption labels, analytic truth or future data enter estimation. A fixed shared-bias variance assumption is supplied to all filter arms, not the realized true variance.
- Eight dependence/robustness/jump combinations receive the same three process-noise trials. CV receives three local-history trials. The primary candidate `d0_r1_j0` is chosen from the eight arms using training state error only; no validation selection occurs. Fixed-q factorial results are also stored.
- All estimators use the same causal onset/release/reversal detector. The fixed 20 px/s head isolates state differences; a separate three-threshold training selection tests the event-head trade-off. False alarms during motion are counted, as well as false alarms per quiet minute.
- Continuation is evaluated separately from state accuracy. Training selects among exponential velocity-decay times 0.1/0.25/0.5 seconds and ordinary CV. No forecasting neural model is trained.

## Synthetic validation

The table reports square root of the mean episode MSE on matched support. These are new corrected-protocol results, not directly comparable to historical aggregate numbers. Selection minimizes mean episode position RMSE plus 0.1 seconds times velocity RMSE. All proposals and per-method support counts remain in the ledgers.

| method | position_rmse | velocity_rmse | cv_0.25_rmse | train_damped_0.25_rmse | cv_0.5_rmse | train_damped_0.5_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| cv | 16.784 | 82.457 | 25.998 | 23.032 | 39.940 | 31.883 |
| d0_r0_j0 | 16.697 | 77.140 | 26.145 | 22.225 | 41.276 | 32.905 |
| d0_r0_j1 | 16.545 | 77.190 | 28.893 | 22.736 | 47.349 | 33.344 |
| d0_r1_j0 | 16.506 | 75.973 | 23.918 | 21.650 | 36.831 | 30.581 |
| d0_r1_j1 | 16.617 | 81.027 | 28.575 | 22.743 | 46.528 | 33.342 |
| d1_r0_j0 | 18.325 | 77.521 | 31.429 | 24.701 | 49.753 | 34.714 |
| d1_r0_j1 | 18.393 | 80.839 | 39.433 | 26.328 | 67.401 | 36.211 |
| d1_r1_j0 | 16.674 | 78.752 | 24.840 | 22.148 | 38.565 | 31.103 |
| d1_r1_j1 | 18.482 | 90.195 | 48.440 | 28.351 | 85.802 | 37.881 |
| nearest | 16.758 | 93.921 | 23.811 | 23.811 | 36.064 | 36.064 |

`d/r/j` indicate dependence modeling, robust measurement weighting and coherent process-noise jumps. With dependence disabled, the assumed shared variance is added to independent measurement noise; with it enabled, it belongs to persistent reset-group latents. This is an explicit sensitivity model, not calibrated real tracker covariance.

## Mathematical interpretation

The observation model is `y_t = p_t + b_group(t) + epsilon_t`. A coherent tracking drift and actual object movement can produce the same observation sequence. Adding a persistent bias latent does not resolve this identifiability problem without additional observations or defensible priors. The eight-arm comparison tests this modeling assumption explicitly instead of treating dependence correction as automatically beneficial.

The robust arm scales its measurement precision by `min(1, (nu+2)/(nu+NIS))`, with `nu=4`. The jump arm uses the same causal innovation and point-agreement evidence to increase process noise. Agreement across points is not independent proof of physical acceleration: a coherent majority tracking error also agrees. Thus the robust and jump mechanisms can conflict. Their combination must earn its place in held-out evidence.

For damped continuation, `E[v(t+s)|F_t] = exp(-s/tau) v_t` yields `E[p(t+h)|F_t] = p_t + tau*(1-exp(-h/tau))*v_t`. Ordinary CV is the `tau -> infinity` limit. This changes the continuation assumption without rewriting the authoritative instantaneous velocity. It is a bounded baseline with a train-selected timescale, not a newly proven optimal dynamics law.

For geometry, the permissible certified solutions under a larger budget include the smaller-budget incumbent. Selecting the lowest source distortion over the newly certified candidates and that incumbent makes supported distortion nonincreasing along the budget path. This guarantee concerns the declared source-fitting objective and certificate; it says nothing about unseen physical contours or generalization.

Matched state timestamps: 13929 / 15060 proposed after the one-second warmup. Forecasts use fixed quarter-second cutoffs, with every unsupported row retained. Resampling intervals in `results.json` are conditional on this synthetic generator and cannot establish real-world significance.

The simulator samples at 50 Hz. Nominal 250/500 ms targets use the first sampled timestamp at or after the requested horizon, so the realized horizon can be up to one frame longer. Every method receives the same target and propagates using its actual elapsed time. These are not exact arbitrary-time continuation scores.

## Events on the same independent synthetic seeds

Truth events are threshold-defined analytic speed onset/release and direction reversal, not human contact events. A match must occur within 250 ms after the reference event, with one-to-one assignment. Unknown observation gaps reset the detector. The standard comparison uses one common detector, not process-jump flags from the adaptive filter.

| method | recall | false_alarms | quiet_false_alarms_per_minute | median_delay_seconds |
| --- | --- | --- | --- | --- |
| cv | 0.535 | 187 | 17.132 | 0.140 |
| d0_r0_j0 | 0.442 | 215 | 25.381 | 0.080 |
| d0_r0_j1 | 0.407 | 190 | 18.401 | 0.100 |
| d0_r1_j0 | 0.407 | 224 | 28.871 | 0.080 |
| d0_r1_j1 | 0.430 | 183 | 15.863 | 0.100 |
| d1_r0_j0 | 0.488 | 161 | 9.201 | 0.090 |
| d1_r0_j1 | 0.535 | 155 | 7.614 | 0.080 |
| d1_r1_j0 | 0.442 | 180 | 15.863 | 0.080 |
| d1_r1_j1 | 0.535 | 156 | 7.931 | 0.080 |
| nearest | 0.000 | 0 | 0.000 | None |

The full event ledger includes each event type, quiet exposure and train-selected threshold results. Zero alarms, when observed, must be interpreted with the actual quiet duration rather than claimed as a zero population rate.

## Source geometry and observation support

All 84 original source masks are retained in the denominator. These fits use the full source boundary; historical every-fourth-vertex holdouts changed some source topologies, so their scores are not a valid before/after accuracy comparison here.

| budget | proposed | fresh_supported | supported | carried |
| --- | --- | --- | --- | --- |
| 16 | 84 | 53 | 53 | 0 |
| 24 | 84 | 53 | 54 | 2 |
| 32 | 84 | 52 | 54 | 4 |

The new holdout helper requires at least four target vertices per ring and checks both full and retained topology. Dense source reconstruction still cannot establish independent physical-boundary accuracy.

Local material transport supports 2074 / 6078 decoded frames versus 349 for the original global-fit ledger. All 6480 timestamps remain recorded. This comparison changes the observation model and may accumulate drift. It does **not** prove a more accurate material trajectory, and the output is not silently fed into a calibrated filter. Neural re-detection is evaluated separately.

## Verification and limits

219 passed in 15.56s. Exact prefix-invariance tests perturb future data and analytic labels; translation transport and certificate fallback tests pass. Full budget paths have nondecreasing certified support and nonincreasing retained distortion. All 161 serialized geometry certificate bindings pass. All 444 prior experiment artifacts, 13 prior source/config files and 820 preserved input entries match their hashes.

Peak sampled numerical-run RSS was 0.207 GiB; minimum sampled available memory was 20.35 GiB. Four worker threads share one process. These logs exclude unrelated applications and the separately run test process. The guard stops at 10 GiB process RSS or below 6 GiB available memory. Observed timing is contended development throughput, not a dedicated-device inference benchmark.

The new components are not yet integrated into or promoted over the frozen Track 1 training reader. Offline teacher/student training, a viable neural observation stream, equal-budget learned memory comparisons, fresh independent annotations and external benchmark evaluation remain required before claims about the full research architecture. Improved engineering support alone does not prove scientific superiority.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track1_observation_revision.run --stage all --workers 4
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m pytest tests -q --basetemp=work/track1_observation_revision/full_pytest
MPLCONFIGDIR=work/track1_observation_revision/mpl PYTHONPATH=src:work/deps:work/track1_deps /opt/anaconda3/bin/python3.13 work/track1_observation_revision/summarize.py
```

Prediction caches are uncompressed, source-hash bound and reused only under the same protocol. Parameters are locked before validation is opened. Geometry outputs include serialized coefficients and decoder-bound certificate files. Raw data and all historical releases remain immutable.
