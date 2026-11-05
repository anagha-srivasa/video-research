# Open Track 1 refinement laboratory

Track 2 is on hold. These experiments and working stores are **not finalized**. Frozen v1/v2 source, data and reports remain unchanged; `REFINEMENT.md` describes the protected v2 release.

The current evidence is in [the research dashboard](reports/track1_lab/dashboard.html), [the progress record](reports/track1_lab/progress.json), and [numerical verification](reports/track1_lab/verification.json).

## Results and limits

On the same 128 manual validation masks, the 32-control compound geometry model reaches IoU 0.9793, strict 2-pixel BF 0.8846, and scale-aware BF 0.9780. The paired v2 values are 0.9190, 0.6347, and 0.8548. These are supplied-contour reconstructions on a new validation cohort; do not compare them directly with the older reserved-test table.

The compound model allocates 32 controls as 32, 24+8, or 16+8+8 across at most three outer/hole rings. A paired compound temporal pilot now provides within-window loop correspondence; persistent correspondence across windows remains open. Single-ring 32-control temporal encodings were fitted for 14,665 observations across matched pilot windows.

On annotated validation interaction pairs, mean separation error is 3.7208 annotation pixels for v2 and 0.1881 for compound geometry. This is declared **2D proximity**, not independently labeled physical contact.

The actual stored 4 Hz slope-hybrid candidate has endpoint velocity RMSE 116.71 native px/s, compared with v2 262.36 and constant velocity 134.50, using the seven-sample 50 Hz RGB reference. No fast-proxy validation episode passed the reference screen. The reference is foreground-affine image motion, not human physical truth. At 500 ms, simple baselines can outperform the polynomial extrapolations; no Koopman/Track 2 model was trained.

Residual-aware normal bands improve held-contour calibration and interaction-region coverage for single-component shapes. They describe representation error, not independent human-label or physical-geometry uncertainty. Compound-loop uncertainty remains unvalidated.

The topology audit found self-intersections in roughly one fifth of sampled functional fits. Raw and shape-guarded variants are both retained. The guard preserves position coefficients but can suppress shape motion, so it is not silently substituted as the accepted model. Validity is checked at sampled times, not proved continuously.

## Working artifacts

- `derived/track1_lab/spatial`: improved 16-control spatial states for all seven train/validation videos.
- `derived/track1_lab/functional_fixed`: complete slope-hybrid fits, with raw and guarded coefficients.
- `derived/track1_lab/half_second_cubic`: complete 0.5 s / degree-3 fits, four GL nodes and physical derivative factor 4.
- `derived/track1_lab/one_second_degree5`: complete 1 s / degree-5 fits, six coefficients and six GL nodes.
- `derived/track1_lab/pilot32`: matched 32-control / 67-channel spatial and functional pilot tensors.
- `derived/track1_lab/compound32_pilot`: per-window raw and guarded 67-channel compound encodings, explicit ring allocations/roles, and uncalibrated covariance. No cross-window sequence index is exported.
- `reports/track1_lab`: parameter searches, paired metrics, reference-quality ledgers and review materials.
- `work/track1_lab`: scratch scripts, checkpoints, ROI caches and logs.

There is no finalized/default ingestion interface for these candidates. Use explicit variant names and validity flags. Original official source splits remain independent from the five-training/two-validation research split. Exposed test participants P06, P21, P27 and P30 are blocked by the lab's input guard.

## Reproduce and continue

Use the existing local runtime from this root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m track1_lab.run --stage report
```

`python3 -m track1_lab.run --help` lists experimental stages. `build-spatial` checkpoints per video/chunk and verifies its pinned build specification. Search stages are exploratory and can overwrite working candidate records; do not rerun them while a dependent build is active. No lab stage finalizes a release or starts Track 2.

The [independent-review worksheet](reports/track1_lab/independent_review_template.csv) has 140 episodes and deliberately blank independent labels. The 28-example AI visual screen does not replace expert identity/velocity annotation.

## Remaining scientific and engineering work

Persistent ring correspondence across windows and topology changes, calibrated compound-loop uncertainty, independent fast-hand/event references, identity labels, and a fresh participant holdout remain open. Larger per-frame control counts alone did not solve temporal endpoint extrapolation. Appearance alternatives trade foreground/background sensitivity against provisional-chain retrieval; none establishes IDF1/HOTA.

## Compound temporal pilot

Supported validation fits: 243 / 288 window/protocol pairs. These are repeated research probes, not unique full-timeline windows. Same-input, same-time compound encoding improves the 100 ms gap mask IoU from 0.8617 to 0.8846. The targeted per-ring guard gives 0.8828 while all declared loop checks pass in the validation cohort.

Raw and guarded states remain separate. The guard checks 512 spatial samples and 69 temporal samples at float32 output precision; this is not proof of continuous-time validity or inter-loop containment. Target topology changes and unsupported inputs remain counted. Matching is confined to retained observations within each window, so no cross-window identity/sequence claim is made. Compound covariance is explicitly uncalibrated.

## Numerical evidence and tests

Verification covers 413,830 working spatial observations and the complete fixed-format temporal variants. Maximum analytical rebuild error is 3.22e-06; significant relative covariance PSD violations are zero. Peak recorded process-tree RSS is 19.50 GiB, with no OOM observed. RSS includes shared mappings.

Tests include physical derivative units, degree-five quadrature, protected test-participant exclusion, hidden-frame independence and per-ring guard isolation. See `reports/track1_lab/tests.json` for the latest command and result.

## Exact synthetic motion stress test

Ninety deterministic episodes test stationary/linear motion, pickup, short out-and-back bursts and reversal, with Gaussian annotation noise, a displaced annotation or missing observations. Models consume only past observations; queries reuse the latest cached state at 0.25 s or 0.125 s output hops. The benchmark does not provide independent real-video event truth. The synthetic train/validation labels identify reproducible seed groups; no parameters are trained or selected on the latter.

At the 0.25 s hop, synthetic-validation slope-hybrid velocity RMSE is 243.01 px/s; linear endpoint continuation lowers it to 196.06 px/s, while event recall changes from 0.833 to 0.750. At the 0.125 s hop, the half-second cubic gives 179.84 px/s and its linear tail 173.07 px/s, with event recall 0.833 for both. Constant velocity gives 172.94 px/s on that same cohort. These are explicit trade-offs, not proof of real fast-event retention. The robust slope-only ablation gives a small gain and is not promoted. See [the protocol and subgroup evidence](reports/track1_lab/kinematics.json) and [traces](reports/track1_lab/kinematics.png).

## Research-report revision

The report-driven changes live in parallel working candidates. Joint topology validity on 243 validation probes changes from 0.9630 to 1.0000; mean IoU changes from 0.7886 to 0.7861. Strict containment, orientation, boundary separation and float32 output are checked at 69 times / 2,048 samples. This remains sampled polygonal validity.

The fixed coarse cubic now has a causal, energy-bounded position residual with weighted orthogonality. Train-only selection chose three atoms. Holdout position NRMSE reductions are endpoint100: 3.4%, gap100: 11.4%, single: 13.4%. The real-source holdouts and synthetic motion tests are separate evidence. The residual is not promoted: it remains worse than constant velocity on important synthetic error/false-alarm metrics. No adaptive-hop promotion was made.

`derived/track1_lab/compound32_topology` contains accepted/rejected geometry states, hierarchy, orientation and tri-state quality flags. `derived/track1_lab/causal_residual` contains 336 pilot coarse/residual states. `derived/track1_lab/joint_kinematics` retains uncalibrated joint endpoint and augmented-coefficient covariance, including cross terms. Train-derived scalar and heteroscedastic representation-discrepancy position calibration are compared explicitly; they do not certify velocity or compound geometry uncertainty.

Use `python3 -m track1_lab.run --stage review-fixes`, then `--stage residual-stress`, `--stage scientific-audit` and `--stage report` with the same local runtime prefix. The CLI fixes the Python hash seed, seeds random/NumPy/Torch and requests strict deterministic Torch algorithms. Process pools seed workers. Resource logs distinguish the 16 GiB steady target, 18 GiB warning and 20 GiB hard stop.

The [scientific gate ledger](reports/track1_lab/scientific_readiness.json) prevents unknown evidence from becoming accepted validity. The new candidate reader exposes `FunctionalEntityState` without a hard-coded geometry control count, retains at most two mmap shards, and exposes unavailable covariance/appearance as `None`. It does not certify ingestion readiness. The [210-proposal annotation queue](reports/track1_lab/independent_reference_queue.csv) has no completed independent labels. Local-polynomial and regularized-spline derivative helpers are available for future reviewed trajectories; neither helper creates independent truth.

The report recommends further contact-weighted and star-convex ablations, independent references and a fresh participant holdout. Those proposals are not authorization to freeze or start Track 2, and they are not claimed completed by this revision. Frozen v1/v2 remain protected.

## Option B: functional history plus causal kinematics

The user selected the hybrid architecture. `derived/track1_lab/causal_kinematics` contains 64,439 attempted timestamps and 33,706 recent-observation-supported states, including explicit failures. It uses 0.2 s local constant velocity and a separately retained robust alternative; neither comes from a polynomial derivative. Joint covariance is conditional, not independently calibrated.

`FunctionalEntityState.functional_endpoint_state` and `.functional_endpoint_velocity` are history diagnostics. `.kinematics.position`, `.velocity`, `.covariance` and `.support_valid` are the separate causal estimate. The development interface version is `track1-v3.0.0-dev.hybrid1`; removed ambiguous endpoint names intentionally fail instead of silently changing semantics. `CandidateStore` preserves the geometry pilot’s hidden-target exclusions. `KinematicStore` reads the full-stream data and must not be substituted into holdout evaluations. Both retain at most two video shards.

The [matched RGB-proxy audit](reports/track1_lab/hybrid_stream_evaluation.json) covers 327 / 328 original cutoffs and 84 hidden-target perturbation checks. It does not supply new independent fast-motion truth. Steady/abrupt/stationary acceptance thresholds must be preregistered after annotation reliability is understood; raw cubic forecast errors remain diagnostics rather than universal Track 1 gates.

The [reconciled probe ledger](reports/track1_lab/attempt_denominators.json) records 288 validation proposals, 283 source-timing-supported probes and 243 emitted compound states. All 45 non-emitted proposals remain counted. Source timing support is not independently certified eligibility. Conditional quality and attempted-state failure rates are reported separately.

Reproduce the new stages with `python3 -m track1_lab.run --stage hybrid`, then `--stage hybrid-evaluate`, `--stage scientific-audit` and `--stage report`, using the existing local runtime prefix. Full-timeline compound geometry and cross-window ring identity remain unfinished; this is not a complete default ingestion release. Track 2 remains on hold.

## Latest remediation report

Continuous-curve certificates distinguish sampled validity from conservatively established validity of `canonical_tensor_bezier1`. Validation: 174 / 243 curve-certified emitted probes; 128 also pass the retained source-epoch checks. These are not full-dataset coverage rates. All unresolved cases and annotation-mode changes remain recorded. The certificate is bound to coefficient bytes, ring contract and decoder source; it does not certify legacy SciPy/raster output or hidden source rings.

New compound fits retain only the latest stable source-topology/correspondence segment. Cached windows with known count changes or matching ambiguity are flagged; an annotation transition is not a verified physical split/merge. Persistent independent ring lineage remains unverified.

The optional endpoint-neutral detail uses a `(1-t)^2` envelope and a weighted null-space basis. Three atoms were selected using training participants only. All 336 serialized pilot states preserve endpoint position/velocity exactly, with float32 weighted orthogonality audited. Six multi-scale descriptors are optional history evidence; they are not authoritative motion or part of the core GL4 collocation.

Plain and robust-innovation Kalman CV models use CPU float64, Joseph updates and strict covariance validation. Existing RGB-proxy and exact synthetic results are retained separately. The synthetic event/position trade-off prevents default promotion; CV remains the primary causal baseline. No calibrated Kalman posterior, CA/IMM regime mixture or independent fast-motion improvement is claimed.

Independent-reference ingestion now validates reviewer/hash/path, sampling, visibility and native-pixel anchor contracts. Declared locked labels stay unopened in the development audit. All existing proposals still lack validated final references. Uncertainty components are bookkeeping with potentially overlapping errors, not an identified independent additive decomposition.

Reproduce with `--stage remediation`, `--stage remediation-audit`, `--stage kalman-stress`, `--stage scientific-audit` and `--stage report` through `python3 -m track1_lab.run`, using the existing runtime prefix. Evidence is in [the remediation manifest](reports/track1_lab/remediation_manifest.json). Contact-local fitting, independent contact/identity annotation, verified lineage, full calibrated uncertainty, a fresh locked participant audit and a clean full-timeline rebuild remain open. No finalization or Track 2 action was performed.
