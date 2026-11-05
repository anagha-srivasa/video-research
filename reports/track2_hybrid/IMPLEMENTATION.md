# Track 2 plan implementation status

The bounded third comparison is complete. The report proposed Koopman, hybrid IMM and recurrent prototypes; existing Koopman and recurrent evidence was retained, and the missing actual IMM was implemented and evaluated. No full-scale Track 2 training, baseline promotion or final freeze follows from these results.

| Report item | Implementation and evidence | Remaining limit |
|---|---|---|
| Stable Koopman | Existing `reports/track2_feasibility` | Original 250 ms feasibility gate failed |
| Recurrent / MLP | Existing `reports/track2_kinematics` | Neither beat constant position across horizons |
| Stationary/CV IMM | New `src/track2_hybrid`; 12 settings, 60 participant/settings jobs, training-only selection | A bounded filter prototype, not learned shape dynamics |
| Optional geometry | Certified canonical M32 endpoint shape, frozen positive scale, causal predicted translation | 179/448 scorable; no future physical-shape certificate |
| Simple baselines | Same-query CV, constant position, nearest source observation | Retrospective source targets, not online RGB truth |
| Single-filter ablations | CV and stationary Kalman filters on identical observations | Share selected IMM parameters; not individually tuned |
| Event preservation | Exact synthetic episodes plus separate locked innovation evidence | No independent real motion/contact reference |
| Uncertainty | Full 4x4 mixture moment covariance, PSD audit, descriptive NEES/NLL/coverage | Conditional and uncalibrated; correlated source targets |
| Resource/inference audit | 12 CPU workers, bounded memory, cached forecast and complete window-refit timing | No upstream RGB/Track 1 cost included |
| Fresh locked final evaluation | Not performed | Exposed P08/P18 cannot be relabeled fresh |

The supplied report contains stale assumptions. The original base-v2 cohort had no usable future validation targets; this study retains the explicitly approved frozen six-second contexts. The existing polynomial switching bank was not an IMM. Retrospective interpolation that uses future knots is not a causal forecasting baseline. Qwen review outputs are weak annotations, not independently adjudicated physical motion/contact ground truth. No force or metric 3D supervision is manufactured.

The new IMM improves continuation against CV but fails against stronger simple baselines. At 250/500/1000 ms its position RMSE is 0.09608/0.16204/0.21626; nearest observation gives 0.09471/0.14108/0.18135. Optional shape transport improves conditional source IoU, but this does not meet the complete motion/contact acceptance requirement.

Synthetic scores use supported update timestamps in [1,3] seconds, not the old cached-between-update stress protocol. The two protocols' RMSE values must not be mixed. Position/velocity/continuation methods share supported cutoffs. Event denominators retain all onsets in that interval; missing update support can cause misses. Nearest-observation continuation has zero declared velocity and therefore no speed-threshold onset events; this is not a finite-difference nearest-position event detector. The separate innovation channel can observe between output updates and is reported separately. At 250 ms hops IMM speed-threshold recall is 0.294 versus CV 0.588, despite lower velocity RMSE. Neither posterior mode weights nor velocity threshold events establish physical contact.

No hyperparameters changed after validation. A synthetic-only dataclass access bug was corrected; the original lock, amendment and completed validation hashes are retained. Resume skips completed numerical stages and checks their output hashes. The full original data and prior models remain intact.

Next scientific dependency remains independent, blinded references for motion, contact and identity plus genuinely fresh evaluation. More correlated windows do not solve the five-participant conformal calibration limitation. This dependency is not satisfied by the present implementation or by another parameter sweep.

Method reference: [Dingler, State estimation with the Interacting Multiple Model method (2022)](https://arxiv.org/abs/2207.04875). The mixing equations and covariance include between-mode dispersion. [FilterPy's IMM implementation](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/IMM.py) was used as an additional primary implementation reference.
