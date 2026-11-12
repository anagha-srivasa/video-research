# Native observation and geometry failure audit

Audit date: 2026-11-12. Scope: the completed `track1_belief` experiment. Existing source modules, datasets, states and original reports were not changed. The audit reads saved point trajectories, refits diagnostic transforms to those points, checks geometry contracts, and renders nine native-frame diagnostic panels. These are failure-attribution experiments, not replacement training results or independent physical validation.

## Finding 1: most native observations never reached the proposed estimator

The ledger has 6,480 timestamps in 20 six-second contexts: 15 training contexts and five validation contexts, spanning five training and two validation participants. Eighteen contexts concern hands; the two remaining contexts concern a pan and a rice container. This is strongly biased toward articulated/deforming targets and is not a representative sample of all object motions.

The tracker initializes at the first complete manual source mask that exactly matches a native timestamp (`src/track1_belief/data.py:140-142`). It reads RGB only from that timestamp onward (`169-186`). Therefore **6,480 is the ledger size, not the number of decoded/tracked frames**: 402 timestamps precede initialization and 6,078 were actually tracked. Three contexts begin after delays of 1.00, 1.72 and 5.32 seconds. All validation contexts initialize at their first timestamp.

| Cohort | Ledger frames | Before seed | Tracked frames | Valid fit | Valid proxy | Fit and proxy | Too few surviving fit points | Enough points but rejected fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 4,680 | 402 | 4,278 | 160 | 268 | 138 | 2,381 | 1,737 |
| Validation | 1,800 | 0 | 1,800 | 189 | 228 | 179 | 1,028 | 583 |
| All | 6,480 | 402 | 6,078 | 349 | 496 | 317 | 3,409 | 2,320 |

The partition is exact: `402 + 349 + 3409 + 2320 = 6480`. Validity is 5.39% of all timestamps and 5.74% of post-seed timestamps. The median age of the **last** valid fit is 0.1284 seconds after initialization; for training it is 0.0400 seconds. Only four contexts retain six fit points at the final timestamp. Initialization delay is a contributor, but it does not explain the main failure.

All timestamp gaps are at most 0.020 seconds, below the tracker's 0.1-second kill threshold (`anchors.py:132-133`). The preserved data do not support blaming dropped decoder frames or timestamp gaps for the collapse.

Evidence: [per-context support](observation_support.json).

## Finding 2: the strongest admission failure is a global shape assumption

Every accepted fit maps the **initial** point configuration directly to the current configuration using one four-parameter similarity (`anchors.py:161-165`). In symbols,

\[
\widehat y_j(t)=a_t+s_tR_t x_j(0),\qquad R_t\in SO(2).
\]

RANSAC uses a fixed two-native-pixel residual threshold, requires at least six inliers, and requires at least half of all surviving fit points to be inliers (`43-50`). Articulation, projective viewpoint change, rolling shutter, nonrigid tissue and accumulated correspondence errors all enter the unmodelled residual

\[
u_j(t)=y_j(t)-a_t-s_tR_tx_j(0).
\]

The gate conflates "not explained by a single similarity to the initial shape" with "no useful motion observation." This assumption is particularly poor for a cohort that is 90% hands.

A read-only replay over recorded point coordinates isolates this gate from point detection and optical flow. There are 2,669 tracked frames with at least six surviving fit points. Using the same coordinates:

| Diagnostic fit | Accepted frames |
|---|---:|
| Original seed-to-current similarity, 2 px | 349 |
| Seed-to-current similarity, 4 px | 836 |
| Seed-to-current similarity, 8 px | 1,734 |
| Seed-to-current affine consensus, 2 px | 447 |
| Seed-to-current homography consensus, 2 px | 641 |
| Previous-to-current similarity, 2 px | 2,601 |

The original 349 count is reproduced exactly. The adjacent-frame result means 97.45% of the frames with enough points support a local similarity consensus, while only 13.08% support the original long-baseline similarity. For example, one left-hand context has enough points throughout all 300 frames; the original fit accepts three frames, while the local fit accepts 300. The rice-container context retains 25 fit points at the final frame, yet the original fit accepts only the first 12 frames.

**These counterfactual counts are not recovered tracking accuracy.** Adjacent-frame composition can drift, consistent points can follow a wrong surface, and affine/homography fits add degrees of freedom. Neither increased support nor a smaller residual proves identity or physical correctness. The replay establishes a specific modelling bottleneck; it does not justify loosening thresholds indiscriminately.

Evidence: [transform replay](observation_transform_replay.json), [replay script](../../work/track1_belief_audit/observation_replay.py), [native diagnostic panels](observation_examples.png).

## Finding 3: irreversible attrition makes six-second support structurally unlikely

Forward/backward error must stay below one native pixel on **every** update (`anchors.py:137-153`). Once a point fails, it is removed from `alive`, never reconsidered, and never replaced. This behavior is deliberate and honestly documented, but it is insufficient for the desired long-horizon tracker.

If the conditional per-update survival probability were a constant \(p\), the expected number of surviving points after \(n\) updates would be \(N_0p^n\). With 64 fit points and 300 updates, even \(p=.99\) gives only about 3.14 survivors; \(p=.995\) gives about 14.23. This is an illustrative calculation, not a fitted independent-loss model. Real losses are correlated through blur, occlusion and motion.

The observed 3,409 count failures are consistent with this absorbing-loss mechanism. The artifact does not save the individual forward-status/backward-status/out-of-image/FB-error cause at each death, so the audit cannot separate those causes retrospectively. The combined `reason` is also identical for point-count and transformation failures (`175-181`). Fine-grained observation diagnostics were missing where they mattered most.

Permanent IDs are valuable, but "no silent identity changes" does not require "no recovery ever." A defensible recovery system needs an explicit visibility state, occlusion hypotheses, re-detection evidence and uncertain identity links. A new point must not be declared the old material point merely because it lies in a mask.

## Finding 4: the output is not a directly observed material centre

The state position is the initial mask centroid transported by a fitted similarity (`anchors.py:89-90,51-68`). It need not coincide with any tracked physical landmark. It can lie away from the textured patch that determines the pose. For an articulated hand, its meaning depends on which subset of surface points remains visible and accepted.

The diagnostic proxy repeats the same operation using disjoint points. Disjoint fit and proxy subsets avoid literal reuse of point coordinates, but both share the image stream, initial mask, coordinate gauge, optical-flow assumptions and similarity family. The proxy therefore cannot independently certify the physical centre or velocity.

Conditional covariance is also narrower than the real uncertainty. The least-squares covariance treats the initial coordinates as fixed and ignores inlier-selection uncertainty (`anchors.py:58-61`). If the reference has uncertain bias \(b\) and the true transform is \(A_t\), the induced cross-time covariance contains

\[
\operatorname{Cov}(A_tb,A_ub)=A_t\Sigma_bA_u^\top.
\]

The implementation instead adds an isotropic seed floor and a scalar deformation allowance at each time (`175,190-192`). That can be a conservative engineering allowance in some settings, but it is not the same covariance model. RANSAC residuals describe the admitted consensus, not all correspondence ambiguity or physical deformation.

Camera-relative output uses a background similarity only when enough image tiles support it (`193-206`). The caller does not provide other-object exclusions, so the initial background mask excludes only the target (`data.py:181`, `anchors.py:95-99`). A globally coherent moving foreground/background plane is not necessarily camera motion; this view needs independent static-scene evidence or an explicitly conditional interpretation.

## Finding 5: increased geometry budget can wrongly reduce certified support

The geometry trial attempts 84 source masks at three budgets, giving 252 model/mask attempts. These are only 24 distinct video/timestamp pairs, not 84 independent image observations. It uses the first 12 manual rows per video (`experiment.py:90-91`), so scene/time stratification is absent.

| Budget | Supported | Too few ring points | Invalid retained source topology | Ring minimum exceeds budget | No certificate among attempted candidates |
|---|---:|---:|---:|---:|---:|
| 16 | 56 | 8 | 8 | 3 | 9 |
| 24 | 57 | 8 | 8 | 2 | 9 |
| 32 | 55 | 8 | 8 | 1 | 12 |

For two masks, budgets 16 and 24 succeed while budget 32 fails. This is a confirmed search defect, not an inherent capacity trade-off. The feasible allocation family at budget 32 contains the certified 16-control allocation. However, candidates are ordered by distortion and only two are certified (`experiment.py:101`; `geometry.py:168-180`). The 24- and 32-control candidates rank ahead of the 16-control candidate but both have unresolved certificates; the known feasible candidate is never considered.

Replaying those two fits with a larger certificate-attempt allowance recovered the same `[16]` allocation in both cases:

- `P02_122 / 680a85f044eea2665dda5b3c087cd344`: 24 and 32 fail local injectivity proof; 16 certifies.
- `P18_03 / fdca579a3650567fc01bcf68a5216794`: 24 fails nonlocal hull separation and 32 fails local injectivity; 16 certifies.

The mathematical invariant should be: when \(\mathcal F_B\subseteq\mathcal F_{B'}\), retain a certified incumbent from \(\mathcal F_B\) while exploring \(\mathcal F_{B'}\). A larger maximum budget should not discard an existing feasible solution. This does not require weakening the certificate.

Evidence: [geometry attribution](observation_geometry_attribution.json), [exact expanded-attempt outputs](../../work/track1_belief_audit/observation_geometry.log).

## Finding 6: the geometry holdout is a representation stress test with altered topology

Source contours come from `cv2.findContours(..., CHAIN_APPROX_SIMPLE)` on a rasterized mask (`pipeline/geometry.py:56-57`). These are simplified pixel-boundary vertices, not independently sampled continuous physical boundaries. The experiment withholds every fourth vertex and reconnects the retained and hidden vertices into separate polygons (`experiment.py:97-102`).

This has several consequences:

1. Vertex density is not uniform arc-length density. Corners, staircases and straight segments contribute differently, so vertex-count splitting is not a spatially uniform holdout.
2. Reconnecting retained vertices can change topology. Among the 76 masks that pass the initial point-count gate, 63 are valid both before and after thinning, eight are invalid both ways, and **five change from invalid full source topology to valid retained topology**. A certificate on the retained fit does not restore the original source topology.
3. The reverse distortion measures distance to the polygon connecting hidden vertices (`geometry.py:96-101,226-230`). That polygon is generally not the original boundary. Large arcs between held-out vertices can be replaced by chords; errors combine reconstruction error and changed-reference approximation.
4. All rings receive equal weight in the mean ring distortion. A tiny fragment or hole can contribute as much as a large outer boundary. This may be appropriate for a chosen fairness objective, but it is not perimeter-weighted physical boundary error and is not explained as such in selection.
5. There is a boundary contract inconsistency: the experiment allows 12 original vertices, producing only three held-out vertices, while `evaluate_geometry` requires at least four. A 12-vertex circle reproduces this: retained fit succeeds, evaluation raises `ValueError: Each ring requires at least four finite xy points`. This edge case did not account for any actual failures in the saved 252-attempt ledger.

Evidence: [holdout contract reproduction](observation_holdout_contract.json). The audit's geometry-attribution JSON uses the exact experiment order: thin original vertices, then normalize orientation. Reversing those steps changes which vertices are withheld and gives different counts.

## What should be tested next

These recommendations repair the observation experiment before adding a larger history model.

1. **Define the estimand and its observability.** A tracked material point, a pose of a rigid patch, a visible-mask centroid and an articulated object's semantic centre are different targets. Use a declared reference landmark/patch or an explicit deformation-and-pose model; do not treat arbitrary surviving-point centroids as interchangeable physical positions.
2. **Replace one long-baseline similarity with linked local measurements.** A candidate factor graph can retain material-point identity, local rigid/deformable patches, keyframe transforms and cross-time uncertainty. Explicitly model \(y_{jt}=A_tx_j+u_{jt}+\epsilon_{jt}\), with gauge constraints and a deformation prior. Add long-range re-detection factors only when independently supported. The local-fit replay motivates this test; it does not establish its accuracy.
3. **Use offline compute for evidence, not just more filtering.** Compare an established causal point tracker and a stronger offline teacher on the same seeded points, including occlusion and reappearance. Preserve time-of-availability distinctions. Offline teachers must not become evaluation truth merely because they are larger. Cross-check disagreements and use independent annotated episodes to establish a real error rate.
4. **Make support a measured objective with risk.** Keep every attempted frame. Report coverage, error conditional on coverage and error/abstention cost together. Record specific failure reasons and survival curves by blur, visibility, object size, articulation and elapsed time. Predeclare independent episodes before further parameter selection.
5. **Correct the geometry optimization contract.** Carry certified incumbents across budgets; distinguish source invalidity, model invalidity and unresolved proof. Use spatially stratified arc-length holdouts while preserving the source topology ledger. Evaluate against the full source boundary as a reconstruction diagnostic and against independently annotated physical boundaries for scientific acceptance.
6. **Only train the downstream memory after observation support is adequate.** The present native corpus provides one qualifying training feature row for the memory probe. It cannot test the proposed learned memory or distillation hypothesis. More optimization on those unsupported inputs cannot establish that the mathematics works or fails.

No new observation pipeline, changed thresholds, released models or scientific acceptance is claimed by this audit. The central result is that the previous experiment failed before most of its proposed downstream mathematical machinery could receive meaningful native data, while its geometry search introduced an avoidable feasibility regression.
