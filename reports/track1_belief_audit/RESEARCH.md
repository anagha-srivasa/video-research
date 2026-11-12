# Why the Track 1 belief experiment underperformed

Research and forensic audit, 12 November 2026. This report concerns the bounded `track1_belief` experiment, not a new evaluation of the whole dataset. Existing implementation, selections and results are preserved. Diagnostic scripts are in `work/track1_belief_audit`; no production model or default was changed.

## 1. Verdict and responsibility

The experiment did not establish that the proposed mathematical architecture fails. It did establish that the implemented experiment was inadequate to test that architecture end to end. I substituted a classical tracker without first establishing usable observation coverage, combined several estimator changes without isolating their effects, and evaluated instantaneous motion through a continuation model that was mismatched to the synthetic motions. These were research-design mistakes, not something that more optimization time would automatically fix.

The strongest findings are:

| Finding | Evidence | Consequence |
|---|---|---|
| Observation construction fails before learning | 349 usable anchor frames among 6,078 tracked post-initialization frames; 6,480 total ledger frames | The memory experiment has one usable training example and trains no probe model |
| A global similarity-to-initial-frame gate rejects locally trackable points | 2,669 frames retain at least six fit points; global similarity accepts 349, adjacent-frame similarity accepts 2,601 | A substantial part of the support collapse is a model/gate mismatch, not absence of all visual evidence |
| Lost points cannot return | Permanent LK rejection, no re-detection | Occlusion becomes an absorbing failure even if the same object becomes visible again |
| Better velocity does not imply better constant-velocity continuation | Exact error decomposition reproduces every matched forecast | A better state estimator and a better predictor are different claims |
| The synthetic study does not isolate its claimed effects | Motion duration is coupled to corruption; all trajectories have the same spatial direction; synthetic participant is motion kind | Seeds vary noise, but do not create independent real-world participants or sufficiently varied task families |
| Event comparison is not a controlled estimator ablation | Adaptive uses process-jump alarms; baselines use speed-threshold onsets | Event gains cannot be attributed only to the state estimator |
| Zero false alarms has little exposure | About 43 seconds of quiet motion | The observed zero is not evidence of a low operational false-alarm rate |
| Geometry search violates budget monotonicity | Two masks certify at smaller budgets but fail at 32; expanded search recovers the smaller feasible allocation | A search heuristic, not representational capacity, causes these failures |
| Several mathematical components were never exercised by the reported experiment | `fit_history`, `gauge_transform`, and `mixture_moments` have definitions/tests but no caller in the experiment; camera-relative observations are stored but not evaluated | Their implementation cannot explain measured gains, and their usefulness remains untested |

The earlier 199 passing tests remain useful engineering evidence. They did not test the validity of the scientific assumptions above. The native-frame statement also needs precision: 6,480 is the ledger count; 402 pre-seed frames were not decoded/tracked in this stage.

This investigation separates **confirmed defects**, **misspecified models**, **unidentifiable quantities**, and **untested research hypotheses**. It does not turn an exploratory replay into a fresh benchmark.

## 2. What exactly are we estimating?

Let `I[0:t]` be RGB available by decision time t. Define the causal information set F_t to include those frames, permitted initialization, and earlier emitted state—not later masks, future tracking refinements or retrospective label repair.

Track 1 currently mixes several targets:

1. The centroid of a visible source mask.
2. An initial reference point transported by a fitted image-plane transformation.
3. A material point on a moving/deforming object.
4. The history of an encoded contour and its coefficients.
5. Future image-plane position.
6. Physical contact, identity and motion events.

These targets are not interchangeable, even with a perfect optimizer.

### 2.1 Visible-mask centroids change under occlusion without object motion

For a visible region V(t), area A(t), centroid c(t), boundary velocity u_b and outward normal n, Reynolds transport gives

\[
\dot c(t)=\frac{1}{A(t)}\int_{\partial V(t)}(x-c(t))\,[u_b(x,t)\cdot n(x,t)]\,ds.
\]

The visible boundary can move because an occluder moves. For a stationary one-dimensional object [0,1] visible only on [a(t),1], the visible centroid is `(1+a(t))/2`; its apparent velocity is `a'(t)/2` although the object is stationary. Fitting a more accurate temporal model to this centroid can estimate the wrong physical quantity more accurately.

**Remedy:** name image-mask centroid, tracked surface point, virtual pose reference, and inferred object motion separately in the contract. Use a stable set of material anchors for surface motion; use visible boundaries for visible geometry. A material anchor is still an image-plane observation, not metric 3D motion.

### 2.2 Contour parameter motion is partly arbitrary

If a curve C(s,t) is reparameterized as `C(psi(s,t),t)`, then

\[
\partial_t \widetilde C=C_t+C_s\,\partial_t\psi.
\]

The extra term is tangential. A changing seam or correspondence can create large coefficient derivatives without any change in the geometric set. Normal boundary velocity is invariant to this pure reparameterization, while pointwise tangential velocity is not. Neither establishes material motion from silhouette alone.

**Remedy:** evaluate set/normal geometry separately from material tracks; condition coefficient dynamics on an explicit correspondence epoch; marginalize ambiguous correspondences or abstain. A fixed cyclic-shift covariance transform is correct only conditional on that chosen shift. It does not account for uncertainty in choosing the shift.

### 2.3 Camera motion is another nuisance, not a free correction

Foreground image motion contains camera movement, object movement, depth/parallax and deformation. A 2D background similarity can be useful for a restricted camera-relative view, but it is not a general 3D stabilization model. The current experiment stores such a view but does not include it among the evaluated sources.

The relevant null direction is easy to see. With observations `y_t = p_t + b_t`, replacing `(p_t,b_t)` with `(p_t+d_t,b_t-d_t)` leaves y unchanged. Without additional evidence or restrictions, object motion and shared tracker/camera drift are not separately identifiable. More compute cannot distinguish identical observations.

## 3. The observation bottleneck, quantified

The 20 selected contexts contain 18 hand episodes and two other objects. Fifteen are training contexts and five are exploratory validation contexts. This targeted cohort is not a representative sample of all objects or all EPIC-KITCHENS videos.

| Recorded outcome | Frames |
|---|---:|
| All proposed ledger timestamps | 6,480 |
| Before permitted seed initialization | 402 |
| Post-seed frames actually tracked | 6,078 |
| Fewer than six surviving fit points | 3,409 |
| At least six points, but global similarity gate fails | 2,320 |
| Accepted anchor measurement | 349 |

The last three categories partition the post-seed frames. Accepted support is 5.74% of post-seed frames and 5.39% of all ledger timestamps. The maximum recorded adjacent-frame interval is 20 ms; the tracker's 100 ms gap reset does not explain this collapse. Median last-valid time after initialization is only about 0.128 seconds.

### 3.1 The geometric assumption was stronger than the tracking task

The tracker requires many surviving points to fit

\[
q_{j,t}\approx s_t R_t q_{j,0}+d_t
\]

within two native pixels. Hands articulate; rigid objects under viewpoint change need not obey a 2D similarity either. Good individual correspondences can therefore fail the pooled model.

A counterfactual replay reused exactly the saved point coordinates:

| Fit on frames with at least six points | Accepted fits / 2,669 |
|---|---:|
| Initial-to-current similarity, 2 px | 349 |
| Initial-to-current affine, 2 px consensus | 447 |
| Initial-to-current homography, 2 px consensus | 641 |
| Initial-to-current similarity, 8 px | 1,734 |
| Adjacent-frame similarity, 2 px | 2,601 |

This is evidence about **gate rejection**, not evidence of recovered tracking accuracy. Adjacent transformations can accumulate drift. A looser threshold can accept incorrect matches. Affine/homography consensus checks here do not establish a calibrated transported-reference estimate. None of these counterfactuals is promoted.

The lesson is to separate local correspondence confidence from validity of one global object transformation. Keep reliable point tracks even when a global pose fit fails. Represent deformation or split local motion support; do not discard all observations because a hand is not a similarity transform of its initial pose.

### 3.2 Irreversible point loss makes long support structurally unlikely

Each point has a permanent alive flag. A forward/backward failure kills the point for the rest of the clip. Under an illustrative independent per-frame survival probability s, survival after T updates is `s^T`; even `s=.995` gives only about 22% survival after 300 updates. Real failures are correlated, making whole-region occlusion particularly damaging. This calculation illustrates the mechanism; it is not an estimated survival model for this dataset.

Modern causal trackers explicitly address memory and re-detection. Track-On2, Track-On-R and TAPNext++ are relevant because they test mechanisms missing here, not because their headline benchmark ranks guarantee success on this dataset. TAPNext++ specifically studies points returning after occlusion. [Track-On2](https://arxiv.org/abs/2509.19115), [Track-On-R](https://arxiv.org/abs/2603.12217), [TAPNext++](https://arxiv.org/abs/2604.10582).

**Required comparison:** persistent LK baseline; a causal neural tracker; and a bounded local-transform/keyframe baseline. All receive the same initial queries. Measure point accuracy, visibility, re-detection, identity switches, coverage and latency separately. A re-detected point needs identity evidence; a newly seeded point must not silently inherit an old point ID.

### 3.3 The reference is neither independent nor always the same estimand

Fit points and proxy points are disjoint, but share initial mask, image sequence, algorithm and global transformation family. Common drift can cancel in their difference. Under deformation, two subsets can also legitimately transport a virtual reference differently. Comparing source centroids with these reference points introduces an additional estimand mismatch.

If prediction and proxy errors are e and r, the covariance of their difference is

\[
\operatorname{Cov}(e-r)=P_e+P_r-C_{er}-C_{er}^{T}.
\]

The current real-data NEES uses the estimator covariance alone. It is therefore not an independently calibrated state-error test. Synthetic NEES against exact generated truth has a different evidential status.

EgoPoints is particularly relevant for a separate external tracking test because it explicitly addresses egocentric out-of-view and re-identification challenges. Before using it, check participant/video overlap with every development and reserved EPIC source; public availability does not make overlapping footage a fresh test. [EgoPoints paper](https://arxiv.org/abs/2412.04592).

## 4. Why improved velocity worsened forecasts

Define instantaneous errors `e_p=hat p_t-p_t`, `e_v=hat v_t-v_t`, and future deviation from true constant velocity

\[
a_h=p_{t+h}-p_t-hv_t=\int_0^h(h-s)a(t+s)\,ds.
\]

For the implemented forecast `hat p_t+h hat v_t`, the exact identity is

\[
e_h=e_p+h e_v-a_h,
\]
\[
E\|e_h\|^2=E\|e_p\|^2+h^2E\|e_v\|^2+2hE[e_p^Te_v]
+E\|a_h\|^2-2E[e_p^Ta_h]-2hE[e_v^Ta_h].
\]

Thus velocity MSE alone cannot determine continuation MSE. An inaccurate, smoothed velocity can partially cancel the error of assuming that a short burst continues indefinitely.

The forensic replay reproduced saved predictions and checked this identity numerically. The following numbers use **pooled matched-query MSE**, so they intentionally differ from the historical mean-of-synthetic-participant RMSE:

| Anchor method | Horizon | Matched queries | Pooled forecast RMSE | Velocity-error contribution to MSE | Velocity/curvature cross contribution |
|---|---:|---:|---:|---:|---:|
| Local CV | 250 ms | 355 | 22.81 px | 396.80 px² | −571.61 px² |
| Adaptive | 250 ms | 355 | 30.73 px | 347.65 px² | −91.98 px² |
| Local CV | 500 ms | 325 | 43.28 px | 1,812.06 px² | −2,907.81 px² |
| Adaptive | 500 ms | 325 | 62.23 px | 1,577.62 px² | −676.37 px² |

Adaptive has a smaller velocity-error term, but loses much of CV's accidental cancellation. Even **true current position and velocity**, extrapolated linearly, give approximately 26.05 px and 54.38 px pooled errors on these matched horizons. This is a diagnostic of the continuation assumption, **not a Bayes lower bound**: a predictor of future stopping can beat it.

### 4.1 The appropriate theoretical target

For squared future-position loss, the optimal forecast is `E[p_(t+h)|F_t]`, not necessarily `E[p_t|F_t]+h E[v_t|F_t]`. Equality requires the appropriate zero-mean future acceleration condition. Kalman filtering under a CV prior supplies that condition by assumption; our pickup/burst generator violates it.

Keep an honest current-velocity estimate. Do not deliberately corrupt it to optimize a forecasting score. Add a separately declared continuation head and evaluate it separately from Track 1 state quality.

### 4.2 A minimal, mathematically specified persistence baseline

Suppose current velocity persists until an independent stopping time T with exponential rate lambda, then becomes zero. For fixed current velocity v,

\[
\Delta p_h=v\min(T,h),\qquad
g(h)=E[\min(T,h)]=\frac{1-e^{-\lambda h}}{\lambda}.
\]

Then the squared-loss optimal mean is `p+g(h)v`, and

\[
E[\min(T,h)^2]=\frac{2}{\lambda^2}\{1-e^{-\lambda h}(1+\lambda h)\}.
\]

The displacement variance for deterministic v is `Var(min(T,h)) vv^T`. Let L=min(T,h). With uncertain p,v independent of T and mean velocity mu_v, the full future-position covariance is

\[
P_{pp}+g(P_{pv}+P_{vp})+E[L^2]P_{vv}+\operatorname{Var}(L)\mu_v\mu_v^T.
\]

The limit lambda→0 recovers CV. A mean-reverting velocity process gives a similar mean but a different covariance; these models must not be conflated.

This is a **candidate diagnostic**, not a universal motion law. It fails on sustained translation if damping is excessive and does not describe reversals. Start with training-only persistence selection and a sustained-translation constraint. Escalate to a small duration-aware mode mixture only if residual evidence supports it. A full learned Track 2 is not needed to test this assumption.

## 5. Statistical traps in the synthetic experiment

### 5.1 The factors are entangled

In `synthetic.py`, `kind=seed%5`, duration uses `(seed//5)%3`, and corruption uses `(seed//5)%6`. Consequently each corruption has a fixed duration category. `participant=synthetic_(seed%5)` is exactly the motion kind, not an independent sampling unit. Motion is always along the direction `(1,.3)`. All events occur near 1.8 seconds. The noise-level schedule also changes with the seed range.

The anchor input is the median of sixteen noisy copies of the same generated position. The outlier corruption affects only one of those sixteen points before the median. Occlusion shifts the centroid but does not remove anchor visibility or force re-identification. This makes the anchor experiment much easier than the real tracker and weakens attribution of any robust-filter gain.

**Replacement design:** independently randomize direction, speed, acceleration profile, duration, onset, object extent, frame rate, depth/camera motion, missingness and fault type. Cross motion and corruption factors rather than coupling their indices. Include common-mode drift, persistent camera bias, coherent wrong-object tracks, re-entry after occlusion, and two worlds with indistinguishable evidence. Hold out entire motion/fault families in addition to seeds. Evaluate homogeneous nulls and mixed regimes. Do not use synthetic labels as physical calibration for real videos.

### 5.2 Tuning fairness and causal attribution are distinct

Only adaptive q/threshold was selected in the original experiment. Gaussian controls inherited adaptive's q. This is useful as a fixed-parameter mechanism comparison, but not a fair comparison of each method's best training-selected configuration. Moreover, adaptive changes robustness, process inflation and event output together.

A post-hoc Gaussian q sweep is recorded in `math_probes.json`; it is exploratory, not a new validation claim. Within the original q grid, the Gaussian methods prefer different q under the original scalar objective. Retuning does not magically produce a universal winner; it demonstrates that the old ranking does not isolate a single theoretical contribution.

Use both experiments: (a) equal-parameter factorial ablations for attribution and (b) equal-budget train-selected baselines for method comparison. Declare the objective before evaluation. Report the Pareto trade-off rather than pretending that `RMSE + 20*(1-recall) + 2*alarms/minute` is a naturally optimal scientific quantity.

### 5.3 Event evidence is not a byproduct of filter aggressiveness

Adaptive reports events when innovation-driven process inflation occurs and speed exceeds 100 px/s. CV/Gaussian controls report speed-threshold crossings. A stopping transition, onset and observation failure can all cause large innovations, while sustained high speed need not.

The current true-event scorer includes pickup/burst threshold onsets but excludes reversals. False alarms count only unmatched detections during speed<5 intervals; unmatched detections during ordinary motion are not counted as quiet alarms. This is a limited metric, not a complete event precision evaluation.

For the adaptive anchor method, zero false alarms occurred over 0.7163 quiet minutes. Even under the favorable homogeneous-Poisson assumption, the one-sided 95% upper rate from zero counts is `-log(.05)/0.7163 = 4.18 alarms/minute`. The actual correlated episodes do not establish that Poisson model. Fifteen recalled events also remain a small sample.

**Replacement:** define onset, release, reversal and fault events separately; use the same detector on each estimator for an estimator ablation, and the same input estimator for detector comparisons. Report one-to-one event matching, delay distributions, precision/all unmatched alarms, quiet-minute rate and rejected/unsupported exposure. Zero speed is a legitimate nearest baseline, not an event-capable model. Threshold choice must reflect a declared latency/false-alarm trade-off.

## 6. Dependence and uncertainty: what is and is not justified

### 6.1 A shared-bias state does not create observability

For scalar state `[p,v,b]`, constant bias b, step delta and observation `p+b`,

\[
F=\begin{bmatrix}1&\delta&0\\0&1&0\\0&0&1\end{bmatrix},\quad
H=[1\;0\;1],\quad
\mathcal O=\begin{bmatrix}1&0&1\\1&\delta&1\\1&2\delta&1\end{bmatrix}.
\]

Rank is two; `[1,0,-1]` is unobservable. A bias prior can bound uncertainty, but cannot reveal the true offset. The existing filter correctly retains substantial uncertainty in a controlled constant-bias probe. A common ramp, however, is observationally equivalent to true motion unless some independent channel distinguishes it. Multi-point coherence alone does not: camera motion and common tracking drift are coherent too.

**Remedy:** specify which nuisance is constant, slowly varying, camera-related or identity-related, and how it is observed. Preserve null-direction uncertainty. For image-relative tasks, remove only declared gauge freedoms. For physical claims, obtain extra evidence rather than adding a more confident prior.

### 6.2 A covariance loading is a substantive model assumption

The current shared covariance uses principal square-root loadings L_t on a persistent latent, implying `Cov(b_t,b_s)=L_t L_s^T` within a group. Marginal variances do not determine those cross-time covariances. Different loadings with the same marginal covariance can encode different temporal dependence. Isotropic deformation residuals do not demonstrate that errors maintain the same latent direction over time.

Do not learn all of Q, R, group bias and camera discrepancy freely from the same unanchored position series and expect unique attribution. Use controlled corruption experiments and externally checked tracks to identify the measurement part; estimate only the remaining dynamics from training data. Innovation whiteness is a useful diagnostic conditional on a correct model, not proof of physical correctness.

### 6.3 The current Student-t label is approximate

The implementation scales R by a one-pass function of innovation NIS and switches Q using the same observation. This is a conditional robust Gaussian approximation, as the code states. It is not exact Bayesian inference for a jointly specified Student-t state/observation model. Conditional covariance omits uncertainty about the selected weight and process regime.

Heavy tails in measurement noise suppress bad observations; heavy tails in process noise permit genuine maneuvers. Those are different explanations of the same innovation. Established robust/trend-following smoothing treats those placements separately; generalized-Bayes filtering offers efficient robust updates but does not identify physical maneuver versus common-mode observation fault without additional information. [Aravkin et al.](https://epubs.siam.org/doi/10.1137/130918861), [Duran-Martin et al., ICML 2024](https://arxiv.org/abs/2405.05646).

Use an explicit small hypothesis set—ordinary measurement, maneuver, observation fault—with predictive likelihoods and model uncertainty, or keep the robust weight heuristic but evaluate/calibrate it as such. If multiple hypotheses remain plausible, a mixture should retain the between-hypothesis term:

\[
P=\sum_k w_k[P_k+(m_k-\bar m)(m_k-\bar m)^T].
\]

A positive-semidefinite matrix is necessary for a covariance; it is not sufficient evidence of coverage.

### 6.4 Quadrature weights and statistical information are different

Time-mass weighting makes a continuous reconstruction loss less sensitive to arbitrary sampling density. That does not make it optimal statistical GLS. Under an exact linear model with known covariance Sigma, GLS is `(X^T Sigma^-1 X)^-1 X^T Sigma^-1 y`. Replacing precision by `D^(1/2) Sigma^-1 D^(1/2)` generally changes weighting. Under an exact correctly specified unregularized model both can target the same coefficients; under misspecification they generally target different weighted projections.

A controlled irregular-sampling example with an exact cubic and independent unit-variance observations gives 1.5172 times the analytically integrated curve variance for the time-mass estimator versus ordinary GLS. Both are unbiased in that unregularized example. This is a Gauss–Markov counterexample to any claim that time-mass weighting is universally better; it is not evidence against its use for misspecified function approximation.

Deduplicate repeated provenance. Model correlated pseudo-observations through Sigma. Separately specify the time measure under which reconstruction distortion matters. A clean formulation first estimates a latent trajectory from its observation likelihood, then projects its posterior onto the desired time measure. Do not conflate “more correlated copies add little information” with “more genuinely independent observations add no information.”

## 7. What a defensible functional memory should preserve

The current joint exponential memories are mathematically meaningful linear functionals under the assumed CV process. Their presence does not improve the p,v filter itself: the zero-memory parity probe agrees to about 2.5e-14 in the mean and 2.3e-12 in covariance. Their intended value is in a downstream readout, which did not train because of the support failure.

Initialization also matters: the two-second exponential memory starts at the first measured position, so after one second it still carries a constant-prehistory weight of `exp(-1/2)=0.6065`. That is an initialization assumption, not observed two-second history. Supply age/support to consumers. A numerical null-space check constructs distinct 20-sample histories with identical three exponential summaries, endpoint and endpoint finite-difference slope; arbitrary event history cannot be recovered from these five numbers.

Let the latent history satisfy `dx=A x dt+G dW` and memory `dm=A_m m dt+B_m x dt`. Augment the state and propagate its **joint** covariance. When a new observation arrives,

\[
\hat m^+=\hat m^-+P_{mx}^-H^T S^{-1}(y-H\hat x^-).
\]

The new observation revises beliefs about past trajectory functionals. A memory of filtered point estimates that discards their cross-covariance is a different approximation.

HiPPO provides optimal projection relative to a specified measure and basis for a supplied signal; it does not prove that a particular compressed noisy trajectory is sufficient for contact, identity or future motion. Three exponential moments are also not an implementation of the full family of polynomial projection memories. [HiPPO](https://arxiv.org/abs/2008.07669).

### 7.1 A useful compression target

For causal history H, compressed state Z=f(H), and square-integrable future/evaluation target Y, define the optimal readout risk `R(U)=inf_g E||Y-g(U)||²`. Then

\[
R(Z)-R(H)=E\|E[Y\mid H]-E[Y\mid Z]\|^2.
\]

This follows from orthogonal projection of conditional expectations. It identifies what compression must preserve: predictive information for a declared target, not merely large-variance coefficients. It does not permit recovering information absent from H.

For a train-estimated linear task `Y≈J m`, memory covariance C and distortion metric G=J^T J, a rank-constrained approximation can be optimized in the transformed coordinates `G^(1/2)m`. G can be singular: use its range/pseudoinverse with explicit null-space treatment, or acknowledge that adding epsilon I changes the metric, as the current SPD implementation does. Optimality holds only for that fixed quadratic task and training distribution. To isolate value beyond current kinematics, evaluate memory conditional on current state; otherwise it can simply duplicate p,v. Population variation of posterior means and within-example posterior uncertainty must not be substituted for one another.

**Experiment:** equal-input, equal-byte comparisons of current state; recent knots; exponential moments; polynomial moments; and task-weighted moments. Include covariance storage and lookup latency in the byte/time budget. Train on sufficient independent episodes and cross-fit the task metric within training participants. Keep historical reconstruction, event evidence and future readouts as separate outcomes.

### 7.2 Why endpoint derivatives are fragile

For normalized shifted Legendre modes `sqrt(2k+1) P_k(2t/T-1)`, endpoint derivative magnitude is `sqrt(2k+1) k(k+1)/T`. Equal independent coefficient noise variance sigma² gives derivative variance

\[
\frac{\sigma^2}{T^2}\sum_{k=0}^{K-1}(2k+1)k^2(k+1)^2.
\]

For four modes the factor is 1,200. This is an illustrative coefficient-noise calculation, not the actual correlated fitted covariance. It explains why small reconstruction errors can coexist with unstable endpoint derivatives. The existing separate kinematic state is the right architectural direction; replacing it with higher-order contour derivatives would reintroduce this problem.

## 8. Geometry: separate capacity, search, source validity and physical accuracy

The 252 geometry records are three budgets on 84 masks, drawn from only 24 video/frame pairs. They are not 252 independent scenes. The supported counts are 56/84 at budget16, 57/84 at24 and 55/84 at32.

The candidate set at a larger total budget includes smaller allocations, yet certification is attempted only for the top two distortion-ranked candidates. Two previously feasible masks are lost at32 because the feasible lower-budget candidate is not tried. Expanding the diagnostic search recovers it. This is a confirmed algorithmic completeness defect under a fixed search budget.

### 8.1 A monotone feasible incumbent

Let F_B be certified representations of at most B controls. Ideally `F_B⊆F_(B+1)`, so `min_(C in F_B) D(C)` cannot increase with the capacity budget. Preserve the prior certified incumbent explicitly and attempt refinements around it. If refinement fails or proof is unresolved, return the incumbent with its unchanged binding. Report `search_exhausted`, `certificate_unresolved`, `source_invalid` and `capacity_insufficient` separately.

Do not claim mathematically nested spline spaces for arbitrary uniform control counts: a 16-control and a24-control uniform grid need not form nested spaces. Candidate retention ensures feasible-set monotonicity independently. Exact knot insertion gives a curve-preserving refinement where supported, but requires a decoder/certificate contract for the nonuniform basis and a new serialized proof check.

### 8.2 Withholding polygon vertices is not independent boundary evaluation

Thinning vertices can change topology and remove defects. The exact ordering of thinning and orientation normalization matters. The current geometry experiment checks topology on retained polygons, not on the untouched full source as a separate acceptance condition. A withheld sparse polygon also introduces chord approximation error into reverse boundary distances. These are source-reconstruction diagnostics, not physical boundary truth.

A separate minimum-size defect exists: twelve input vertices pass the experiment's input gate but leave only three held-out vertices, while the evaluator requires four. This is a reproduced edge case, not a cause of the recorded 252-run failures.

Keep complete source topology/provenance in a separate diagnostic ledger. Do not repair source labels using withheld information during fitting. Use dense independently reviewed boundary samples for final geometric accuracy, or name source-only errors precisely. A certified simple curve can still follow the wrong object boundary.

### 8.3 A promising bounded refinement objective

For observed visible boundary evidence Y, propose

\[
\min_C\ D_{\mathrm{visible}}(C,Y)+\lambda_{\kappa}\int\kappa^2ds
+\lambda_B\,\mathrm{bytes}(C)
\quad\text{subject to the declared topology/certificate constraints.}
\]

Visibility masks prevent penalizing an unseen boundary as though it were missing geometry. Contact-region weighting is legitimate only when the region is independently specified or explicitly treated as a model proposal. Retain complete rings when the contract promises them; represent unknown/missing pieces explicitly.

A varifold discrepancy can compare boundary sets without choosing an arbitrary seam, using positions and unoriented tangents. It is a candidate replacement for correspondence-dependent fitting losses, not a topology proof or a source of material identity. [Charon and Trouvé](https://arxiv.org/abs/1304.6108), [oriented-varifold framework](https://openaccess.thecvf.com/content_cvpr_2017/html/Kaltenmark_A_General_Framework_CVPR_2017_paper.html).

## 9. Best next architecture to test, with explicit assumptions

The most defensible combination is **causal persistent observations + a small uncertainty-aware state estimator + separate history and event channels + optional certified geometry**. The innovative part worth testing is the interface between these pieces, not piling their names into one model.

### 9.1 Observation layer: persistent local anchors with nuisance-aware pooling

Use a causal tracker with visible/occluded/out-of-view states and tested re-detection. Maintain local keyframes and stable point identity. A local linearized measurement model is

\[
z=A\theta+L b+\epsilon,\qquad b\sim N(0,K),\quad\epsilon\sim N(0,R).
\]

Marginalization gives `Sigma=R+LKL^T`; Woodbury permits efficient low-rank precision updates. If nuisance b instead has an uninformative prior, the nuisance-orthogonal precision is

\[
W=R^{-1}-R^{-1}L(L^T R^{-1}L)^{-1}L^TR^{-1}.
\]

Before estimating theta, inspect `A^T W A`. If the desired motion lies in the nuisance span, information vanishes; projection has removed the signal as well as the nuisance. This rank test prevents a supposedly clever camera/drift correction from manufacturing observability. Use pseudoinverses and explicit null-space handling when required, not silent regularization disguised as evidence.

Overlapping keyframes can maintain a common image-coordinate gauge, but each reanchoring must propagate its covariance and dependence. If no reliable overlap exists, start a new identity/coordinate epoch instead of stitching with an unjustified deterministic offset. Per-point visibility and identity confidence remain distinct from a global pose's validity.

### 9.2 State layer: bounded hypotheses, not one unexplained adaptive Q

Test a small bank for steady motion, stopping/turning, and observation failure. Use prequential evidence and explicit duration assumptions; preserve posterior mode uncertainty. The model bank should distinguish changing dynamics from a changing measurement process. Avoid overfitting a large IMM to five participants. A bank that cannot beat independently tuned simple filters should not be retained for elegance.

For event evidence, monitor signed, uncertainty-normalized motion residuals and declared change hypotheses. A robust changepoint method is relevant, but iid segment assumptions must be applied to an appropriately modeled/whitened residual process, not blindly to correlated frame positions. [Bayesian online changepoint detection](https://arxiv.org/abs/0710.3742), [robust variant](https://arxiv.org/abs/2302.04759).

The recent Bagel preprint studies bounded-cost approximate posterior merging for regression changepoints. It is a possible speed-oriented alternative after correctness is established, not a validated video-event solution. [Bagel, November 2026](https://arxiv.org/abs/2611.03230).

### 9.3 Offline teacher / causal student: distill the right distribution

Expensive offline tracking/smoothing can generate training supervision, but it must not leak future evidence into an online input. If a correct teacher has smoothed mean m_T and covariance P_T using more information than F_t, then

\[
m_t=E[m_T\mid F_t],\qquad
P_t=E[P_T\mid F_t]+\operatorname{Cov}(m_T\mid F_t).
\]

These follow from conditional expectation and total variance when the teacher and student concern the same latent quantity and nested information sets. Copying P_T into the student discards uncertainty resolved only by future frames. A deterministic teacher coordinate is not a certain causal target.

An exact scalar Gaussian check illustrates the size of the error: latent prior variance1, past measurement variance1, and future measurement variance0.01 give causal variance0.5 but smoothed teacher variance1/102≈0.009804. The missing between-teacher-mean variance is0.490196. Copying teacher confidence would understate the causal variance by a factor51 in this toy model, even with a perfect teacher.

Train a student density against a verified teacher distribution or samples using a proper score; for forward KL, the ideal unrestricted student is the conditional mixture of teacher posteriors given causal inputs. A Gaussian student must retain both within- and between-teacher-mean variance. Teacher bias breaks the ideal guarantee. Multi-teacher agreement does not remove shared bias.

PATT supplies recent evidence that privileged appearances can improve a deployable tracker while removing training-only branches at inference. It is object tracking evidence, not proof of calibrated point-trajectory or physical-motion distillation. Track-On-R's learned verification is also more substantial than agreement-only pseudo-label filtering. [PATT, September 2026](https://arxiv.org/abs/2609.02471).

Use train-only videos for teacher-derived targets, fresh independent references for assessment, and explicit teacher-selection coverage. Confidence-weighted training can bias the training distribution toward easy cases; report accepted and rejected cases by regime. Verify causal student outputs with future-suffix mutation and delayed-delivery tests.

### 9.4 A constrained research objective

Instead of one arbitrary scalar score, predeclare:

\[
\min\ R_{\mathrm{position}}+\beta R_{\mathrm{boundary}}
\quad\mathrm{s.t.}\quad
R_{\mathrm{event}}\le r_*,\;
\mathrm{coverage}\ge c_*,\;
\mathrm{p95\ latency}\le\ell_*,\;
\mathrm{state\ bytes}\le B_*.
\]

The constants must follow the intended downstream use and reference reliability; they are not published standards or guarantees. Report Pareto fronts while requirements are unsettled. Proper predictive scores assess both sharpness and calibration, whereas coverage alone can be improved by arbitrarily inflating covariance. [Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

## 10. Research sequence and stopping rules

| Stage | Question | Required comparison/evidence | Stop condition |
|---|---|---|---|
| 0. Correct experimental contracts | Are targets, clocks, units, factors and event definitions consistent? | Factorial synthetic generator; common event decoder; matched input/output denominators; call-graph coverage | Any mismatch blocks a headline improvement claim |
| 1. Establish observations | Can material queries persist and reappear accurately? | LK vs causal neural tracker vs local/keyframe baseline; EgoPoints and nonoverlapping local reviewed clips | Insufficient independent training/evaluation support blocks compression training |
| 2. Identify estimation benefit | Does dependence/robustness help with fixed observations? | Independent/dependent/robust/jump factorial arms; equal-budget tuning; matched and all-case risk | No gain outside one confounded synthetic family: retain simpler model |
| 3. Separate continuation | Is remaining error state estimation or future persistence? | Exact-state CV diagnostic; nearest/CV/damped/mixture readouts with common state | Do not degrade instantaneous velocity to hide predictor mismatch |
| 4. Preserve events | Are onsets/reversals/releases retained at acceptable delay? | Same events, episodes, quiet exposure and missingness policy; no reliance on NIS alone | Unacceptable recall/delay/false alarms blocks event claims |
| 5. Test memory | Is there useful history beyond current state? | Equal bytes/inputs, nested train-only task metric, independent episodes | If knots or current state match it, avoid extra memory complexity |
| 6. Test geometry | Is refinement genuinely useful and certifiable? | Feasible incumbent, full-source diagnostics, independently checked visible boundaries and contact subsets | Better source fitting without physical/reference gain is not acceptance |
| 7. Distill and profile | Can expensive training reduce deployment cost? | Fixed teacher quality, causal student, identical target suite, latency/accuracy trade-off | No promotion if student saves time by losing required event/coverage quality |
| 8. Calibrate and lock | Do uncertainty and final performance generalize? | Separate calibration units and genuinely fresh participants | No finite participant-level guarantee from the present five units |

A small controlled experiment must have a specific falsifiable claim before a large training run. More compute is best spent on informative failures, trusted tracking supervision and controlled ablations—not adding parameter sweeps after the evaluation cohort is exposed.

## 11. Calibration and limits of mathematical guarantees

For split conformal calibration with n exchangeable independent calibration units and miscoverage alpha, the standard rank is `ceil((n+1)(1-alpha))`. With n=5 and alpha=.05 it exceeds n; the usual guaranteed construction is unbounded. Nineteen units merely make this rank finite—it does not ensure useful tightness, conditional coverage, distribution-shift robustness or independence. Windows from the same participant do not automatically become new participant units.

Recent conformalized Kalman methods explicitly require a suitable labeled calibration set and exchangeability at the stated unit. Their sample-wise and trajectory-wise guarantees differ. They cannot turn Qwen weak labels or correlated source proxies into independent physical truth. [Conformalized Kalman Filters, September 2026](https://arxiv.org/html/2609.27506v1).

The mathematical identities and counterexamples in this report are established under their stated assumptions. The proposed combined architecture is an engineering/research hypothesis. Neither its novelty nor globally best performance has been established. No paper survey can honestly prove that a proposed combination is the best possible approach for an unmeasured task.

## 12. Evidence and reproducibility

- [Forecast audit](forecast_audit.json) and [per-query decomposition](forecast_decomposition.parquet): exact reproduction, common cohort, additive MSE terms, synthetic design and event exposure.
- [Mathematical probes](math_probes.json) and [q-sweep rows](math_q_sweep_rows.json): GLS counterexample, nonidentifiability, memory parity and exploratory baseline sensitivity.
- [Observation support](observation_support.json) and [transform replay](observation_transform_replay.json): full frame denominators and saved-point counterfactuals.
- [Geometry attribution](observation_geometry_attribution.json): source/retained topology and expanded candidate-search receipts.
- [Observation examples](observation_examples.png): source RGB and saved anchors; diagnostic visual evidence, not independent labeling.
- [Theory checks](theory_checks.json): stopping-model moments and risk, observability rank, causal-distillation variance, endpoint derivative amplification and finite-memory information loss. These are controlled mathematical checks, not newly trained-model results.
- [Audit figures](audit_summary.png): support attribution, matched forecast decomposition and conditional geometry yield.
- [Mathematical implementation audit](math_audit.md), [observation audit](observation_audit.md), and [primary-literature review](literature_audit.md): supporting detail.

The replay scripts reuse local data and write only new audit artifacts. The forecast replay asserts agreement with saved predictions and exact decomposition closure. This audit does not rerun the full training workflow or silently amend the prior results. The [integrity receipt](integrity.json) checks 444 original experiment artifacts, 13 experiment source/config files and 820 baseline inputs. Counts refer to each receipt separately and may overlap.
