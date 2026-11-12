# Track 1 failure investigation: primary literature and mathematical implications

Research checked on 11–12 November 2026. This is a focused primary-source review, not an assertion that every relevant paper has been found. Existing Track 1 sources and experiment artifacts were not changed. The mathematical deductions below are our analysis; they are separated from reported results in the cited papers.

## Finding

The last experiment is not a meaningful negative test of the strongest observation architecture in the preceding research. It substituted a permanently retiring, local Lucas–Kanade tracker for the persistent, re-detecting learned trackers on which the proposed observation strategy depended. The distinction matters mathematically: a downstream estimator cannot recover information that the observation channel stops supplying. This substitution was documented, but it substantially narrowed what the experiment could establish.

The literature supports testing stronger observations before expanding the state model. It does not support a promise that a larger filter, additional covariance terms, or a more elaborate history basis will overcome unidentified physical states, correlated teacher mistakes, or severe observation dropout.

## What the verified tracking papers actually support

| Primary source and date | Relevant established contribution | Application to this project and boundary |
|---|---|---|
| [Track-On2: Enhancing Online Point Tracking with Memory](https://arxiv.org/abs/2509.19115), first submitted 23 September 2025; v2 16 March 2026, TPAMI 2026 | Causal frame-by-frame tracking with an explicit expandable memory and coarse-to-fine localization. The paper studies training duration, sampling and inference memory, rather than treating more memory as automatically better. | A suitable persistent-observation baseline. Its reported throughput is measured on NVIDIA A100, not the local Apple device. More memory can harm shorter sequences; the paper's memory ablations do not justify maximizing memory indiscriminately. |
| [TAPNext++: What's Next for Tracking Any Point?](https://arxiv.org/abs/2604.10582), 12 April 2026 | Recurrent frame-by-frame tracking trained for long sequences, re-entry augmentation and occluded-point supervision. It evaluates re-detection explicitly. | Directly addresses the failure that permanent LK retirement cannot address. Compare its pretrained checkpoint before inventing a replacement tracker. Long-term survival and re-detection must be measured separately from localization on already-visible points. |
| [Real-World Point Tracking with Verifier-Guided Pseudo-Labeling / Track-On-R](https://arxiv.org/abs/2603.12217), 12 March 2026, CVPR 2026 | A learned verifier ranks predictions from multiple trackers using visual evidence; selected pseudo-labels adapt a tracker to real videos. | Supports expensive training-time verification followed by a single tracker at inference. It does not equate teacher consensus with truth. The verifier's temporal attention and windowed teachers must be treated as offline unless a prefix-only implementation is demonstrated. |
| [CoTracker3](https://arxiv.org/abs/2410.11831), 15 October 2024, ICCV 2025 | Teacher-generated labels on real videos improve synthetic-pretrained tracking; online and offline model variants are provided. | A strong teacher and comparison point. Its online/windowed setting must not silently be counted as zero-lookahead frame-by-frame inference. Record when each output becomes available. |
| [BootsTAP](https://arxiv.org/abs/2402.00847), 1 February 2024, ACCV 2024 | Student–teacher training on unlabelled real video improves point tracking. | Establishes a useful adaptation direction, not a guarantee that repeating self-training on seven participants will improve generalization. Pseudo-label provenance and independent validation remain necessary. |
| [EgoPoints: Advancing Point Tracking for Egocentric Videos](https://arxiv.org/abs/2412.04592), 5 December 2024, 2025 paper | An egocentric benchmark with substantial out-of-view/re-identification difficulty, and K-EPIC semi-real training data. | More task-aligned than relying only on generic short video tracking scores. Check participant/video overlap with local EPIC data before using any part for selection or an independent claim. |
| [TAP-Vid](https://arxiv.org/abs/2211.03726), November 2022 | Defines tracking a queried physical point and supplies point/visibility evaluation. | Its target is not the changing centroid of a visible segmentation mask. Use its distinction between correspondence and visibility when defining the local reference contract. |

The [official Track-On repository](https://github.com/gorkaydemir/track_on) currently supplies Track-On-R, Track-On2 and verifier checkpoints. Its documented stack uses CUDA/MMCV operators, and DINOv3 backbone access is separate from the tracker checkpoint. Therefore the pretrained model is research-accessible, but successful Metal execution, local latency, and parity after any operator port remain unverified. The [official TAP repository](https://github.com/google-deepmind/tapnet) supplies a PyTorch TAPNext++ path; that is a useful alternative to evaluate, not proof of Apple GPU compatibility.

Two compute details prevent misleading comparisons. Track-On-R reports adaptation using 32 A100 GPUs, while TAPNext++ reports long-sequence fine-tuning using eight H100 GPUs. Those are published training configurations, not minimum requirements. TAPNext++ distinguishes frame latency from 32-frame chunk throughput: its highest advertised FPS must not be substituted for a zero-buffer end-to-end latency measurement. [Track-On-R full paper](https://arxiv.org/html/2603.12217v1), [TAPNext++ full paper](https://arxiv.org/html/2604.10582v1).

There is no verified justification here to declare either model universally best for this dataset. A small paired native-video bake-off is the necessary next test. Model choice must include localization, visibility, re-detection, coverage, worst-case latency and memory, all on the same queries.

## Mathematical trap 1: a changing mask centroid is a different estimand

Let a projected object occupy a set \(S_t\), and let \(V_t(x)\in\{0,1\}\) indicate visibility. The visible-mask centroid is

\[
c_t=\frac{\int_{S_t}xV_t(x)\,dx}{\int_{S_t}V_t(x)\,dx}.
\]

Even for a stationary object, changing \(V_t\) changes \(c_t\). Its derivative includes visibility and shape terms, not just material velocity. For a fixed domain with smoothly varying visibility, the visibility contribution is

\[
\dot c_t=\frac{1}{A_t}\int_{S}(x-c_t)\dot V_t(x)\,dx,
\quad A_t=\int_S V_t(x)\,dx.
\]

This equation is an analytical demonstration, not a claim that binary occlusion is differentiable at every event; jumps require the corresponding finite-difference or distributional treatment. It shows why denoising centroid motion cannot, by itself, recover material motion.

A material anchor instead has a persistent surface identity \(q\), with observation \(u_t=\pi(C_t^{-1}X_t(q))\). Even this is image motion: object motion, camera motion and depth remain coupled. A similarity-transformed initial mask centroid is an explicitly defined reference-point surrogate; it is not automatically a material point or the center of mass of a deforming object.

**Test:** create a stationary textured object with controlled partial occlusion. Evaluate visible centroid, persistent point, transported reference point, and any physical center separately. A method must not win by changing the target definition.

## Mathematical trap 2: additional latent bias states do not create identifiability

Consider the scalar observation model

\[
y_t=p_t+b_t+\epsilon_t.
\]

The two latent explanations \((p_t,b_t)=(ut,0)\) and \((0,ut)\) induce the same observations and noise distribution. No estimator using only \(y_{1:t}\) can distinguish them without restrictions or additional evidence. A prior on \(b_t\) selects an explanation; it does not convert the explanation into an observation.

For a constant shared bias, position and bias have a gauge freedom \((p,b)\mapsto(p+d,b-d)\). Relative increments remove that constant, so velocity can be better identified than absolute position. A new arbitrary bias group can reintroduce ambiguity. Conversely, time-varying camera or tracker bias need not cancel in increments.

The correct local model should expose which uncertainty comes from shared initialization, current localization, camera alignment, identity ambiguity and deforming geometry. For example,

\[
y_{i,t}=h_i(x_t)+J_{i,t}b_{\rm seed}+G_{i,t}b_{\rm camera,t}+\epsilon_{i,t}.
\]

This is a proposed model family, not an identified model for the present data. The Jacobians and cross-covariances require validation; an isotropic persistent bias is only one approximation. Check the observable subspace of the linearized system and keep non-observable directions prior-dominated. Never report them as data-calibrated physical certainty.

**Test:** provide identical observations under two different latent explanations. The model should retain ambiguity or explicitly disclose its prior dependence. It should not claim to recover both explanations accurately.

## Mathematical trap 3: robustness and manoeuvre preservation conflict when evidence is shared

A Student-t observation loss has innovation weight proportional to

\[
w(r)=\frac{\nu+d}{\nu+r^\top S^{-1}r}.
\]

Large innovations are downweighted. That is helpful for a localization failure, but can suppress an actual sudden movement when the process model is too stiff. Placing heavy tails in the process instead makes jumps easier, but can chase measurement failures. These alternatives are explicitly different in [Aravkin, Burke and Pillonetto, 2013](https://arxiv.org/abs/1303.5588). [Roth et al., 2017](https://arxiv.org/abs/1703.02428) also makes clear that practical Student-t filtering involves approximation; its covariance is not automatically the exact posterior of an arbitrary heuristic update.

A single innovation and a scalar inlier fraction are insufficient evidence to guarantee which explanation is correct. Several points can agree because the camera moved or because all tracks attached to the same occluder. Conversely, a correctly tracked deforming hand may not fit one similarity transform.

**Proposed experiment:** retain a small explicit mixture of explanations—ordinary motion, manoeuvre, measurement corruption and visibility loss—and compare it with the current hard switch. Use likelihoods from distinct evidence channels, such as local appearance identity, point configuration and background alignment. Learn hazard/duration parameters on training episodes. This is a hypothesis requiring ablation, not a theorem that mixtures will outperform a robust filter. A separate event output must be evaluated on onset/release labels, not defined by the same innovation that triggers the process jump.

## Mathematical trap 4: better current velocity can predict the future worse

Write the true continuation as

\[
p_{t+h}=p_t+h v_t+u_h,\qquad
u_h=\int_0^h(h-s)a_{t+s}\,ds.
\]

If the predictor is \(\hat p_t+h\hat v_t\), its error is

\[
e_h=e_p+h e_v-u_h.
\]

Therefore

\[
E\|e_h\|^2=E\|e_p\|^2+h^2E\|e_v\|^2+E\|u_h\|^2
+2hE[e_p^\top e_v]-2E[e_p^\top u_h]-2hE[e_v^\top u_h].
\]

Minimizing instantaneous velocity RMSE optimizes only one term. A low-pass velocity can accidentally be a better continuation predictor for motions about to stop or reverse. That does not make it the more accurate instantaneous velocity. Conversely, perfectly recovering a short burst and extrapolating it for half a second can be worse than nearest position.

A useful minimal baseline is an integrated mean-reverting velocity model,

\[
dv=-\lambda v\,dt+\sigma\,dW_t,\qquad
E[p_{t+h}\mid p_t,v_t]=p_t+\frac{1-e^{-\lambda h}}{\lambda}v_t.
\]

The limit \(\lambda\to0\) is CV; large \(\lambda\) approaches nearest position over a fixed horizon. This is a standard stochastic model proposed as a diagnostic baseline, not a newly invented optimal physical law. Tune persistence independently of measurement filtering, on training data only, and compare equal-support forecasts. If this simple horizon-dependent baseline fixes the continuation gap, the failure concerned persistence assumptions rather than state estimation.

[Uncertainty-Aware Trajectory Forecasting from Imperfect Tracking](https://arxiv.org/abs/2608.30899), 31 August 2026, is relevant recent evidence for carrying localization and association uncertainty into a predictor and training with correlated corruption. Its pedestrian benchmarks do not establish egocentric hand-object performance. It supports a testable design principle, not a transferred score.

## Mathematical trap 5: agreement, independent point sets and covariance validity are different

Suppose teacher outputs are \(y_k=x+b+\epsilon_k\). Teacher disagreement estimates variation in \(\epsilon_k\), while a common error \(b\) can remain arbitrarily large with perfect agreement. Likewise, fit/proxy points sharing the same seed image, local tracker, reference transform and occluder are not independent physical references merely because their point indices are disjoint.

An ensemble covariance should at least include both within-component and between-component terms,

\[
\Sigma=\sum_k w_k\left[\Sigma_k+(\mu_k-\bar\mu)(\mu_k-\bar\mu)^\top\right],
\]

but this identity still assumes the components describe the intended distribution. It cannot expose a systematic error absent from every component.

When combining estimates with unknown correlation, [Split Covariance Intersection](https://arxiv.org/abs/2501.07915), 14 January 2025, and [Overlapping Covariance Intersection](https://arxiv.org/abs/2603.16768), 17 March 2026, revised 1 October 2026, provide principled conservative fusion directions under stated structural assumptions. They do not repair wrong means or invalid input covariance bounds. Use exact joint covariance where available; use conservative fusion where assumptions can be defended; otherwise call the reported covariance conditional and uncalibrated.

**Test:** inject a shared displacement into all teachers, leaving their mutual disagreement unchanged. The verification process must not interpret agreement as new independent confidence. Also split assessment by teacher disagreement, occlusion and texture, not only aggregate calibration.

## Mathematical trap 6: smoothing teachers cannot donate their confidence to causal students

Let \(\mathcal F_t\) be all allowed input through time \(t\), and \(\mathcal F_T\supseteq\mathcal F_t\) include future video available to an offline teacher. For an exact Bayesian teacher,

\[
m_t=E[m_{t\mid T}\mid\mathcal F_t],
\]

\[
P_t=E[P_{t\mid T}\mid\mathcal F_t]
+\operatorname{Cov}(m_{t\mid T}\mid\mathcal F_t).
\]

These are the tower property and conditional law of total covariance. A causal student has less information; copying the teacher's smaller smoothed covariance drops the second term. It can produce excellent mean fits with unjustified confidence.

Offline labels are permissible training targets if no future features leak into the deployed input or evaluation state. The student should learn the conditional distribution of teacher targets given the permitted prefix. Minimizing expected forward KL from teacher distributions yields their conditional mixture in an unrestricted student family. A restricted Gaussian student must retain the between-teacher/future-outcome component, and the teacher itself may be biased. Real calibration is still necessary.

**Proposed combination:** use expensive offline multi-tracker verification and smoothing to produce mean/distributional supervision; train one prefix-only recurrent tracker and a compact belief update; calibrate that causal student's error separately. Audit causality by perturbing future frames and confirming unchanged output at \(t\). Do not mistake a prefix-only student architecture for proof that its cached features or initialization were causal.

## Mathematical trap 7: a stable memory is not automatically a useful memory

The current exponential memory satisfies \(\dot m=(p-m)/\tau\), which is a valid low-pass linear functional. It is not equivalent to a multi-degree orthogonal history representation. [HiPPO](https://arxiv.org/abs/2008.07669), August 2020, and [HiPPO Zoo](https://arxiv.org/abs/2602.21340), 24 February 2026, revised 29 May, motivate structured and adaptive representations of history. Their existence does not establish sufficiency for this project's contact, identity or future-motion tasks.

For squared-error prediction, a memory \(M_t\) is task-sufficient only if the needed conditional mean satisfies \(E[Y\mid\mathcal F_t]=g(M_t)\). For distributional losses it must preserve the corresponding conditional distribution. Three smooth averages can map two histories with different latest reversal times to nearly the same state. This can be tested constructively before neural training.

Maintain separate exact summaries for discrete events or visibility ages where their semantics are known, and compare equal-byte representations: recent knots, cubic coefficients, exponential moments and orthogonal moments. Learn compression using explicit downstream risks. It is invalid to infer a new representation's usefulness from a memory stage that had insufficient eligible training rows to fit its probe.

## Mathematical trap 8: missingness is selective and coverage cannot be repaired algebraically

Track failure is linked to fast motion, occlusion, low texture and small objects. Thus evaluation conditional on tracker survival estimates a different population risk. Write \(A=1\) for accepted observations. The conditional quantity \(E[L\mid A=1]\) is not \(E[L]\), and the gap is unidentified without outcomes or assumptions for rejected cases.

Report risk–coverage curves, survival/re-detection curves and episode-level usable durations. Keep all proposed frames/queries in denominators. Inverse-propensity weighting does not recover a subgroup where acceptance probability is zero. Do not infer a special numerical critical rate for this dataset from [Kalman filtering with intermittent observations](https://web.eng.ucsd.edu/~massimo/Papers_files/Kalman.pdf): that theory studies a defined linear stochastic model and observation process, while this tracker has selective, often absorbing loss.

[Conformalized Kalman Filters](https://arxiv.org/abs/2609.27506), 23 September 2026, provides a recent route for calibrated regions. Its guarantee requires exchangeability across calibration and test trajectory units and a fixed learned procedure before calibration. Correlated frames inside one trajectory are allowed by that construction; treating overlapping windows as additional exchangeable participants is not justified. For the usual split-conformal finite-sample quantile at 95%, \(k=\lceil(n+1)0.95\rceil\). With five units, \(k=6>5\), giving an unbounded threshold under that construction. Nineteen units only make the rank finite; they do not guarantee tightness, subgroup validity or defensible exchangeability.

## Ordered experiments justified by the evidence

1. **Lock estimands and timing.** Separately label material image points, visible mask geometry, camera-relative 2D motion and contact evidence. Declare output availability times and privileged initialization. Preserve unknown labels.
2. **Establish a viable observation baseline.** Compare pretrained Track-On-R/Track-On2, TAPNext++, CoTracker3 and existing LK on identical native queries, including occlusion/re-entry. Run all methods with documented input resolution and real availability latency. Test supported correctness and all-case availability jointly. A small independently inspected cohort is more valuable here than another large filter grid.
3. **Cross observation and inference factors.** Hold observations fixed while comparing CV, damped CV, Gaussian filtering, robust filtering and mixture hypotheses. Hold the estimator fixed while changing the tracker. This separates an observation gain from a filtering gain. Tune each baseline fairly on training participants.
4. **Challenge assumptions before training a large student.** Include shared drift, independent anchor noise, nonrigid deformation, visibility-induced centroid shift, camera motion, long dropout, re-entry and identity swaps. Ground truth and algorithm inputs must be generated separately. Produce one counterexample per claimed advantage.
5. **Only then train the expensive teacher/cheap student path.** Preserve teacher identities and uncertainty provenance, student prefix constraints and held-out participants. Compare student quality with both individual teachers and verifier-selected labels. Track local inference p50/p95, initialization time, update time and actual memory rather than translating A100/H100 paper numbers.
6. **Compress only after a supported task signal exists.** Require a matched equal-byte probe to beat recent knots and the original representation across participants. Retain the full stronger observation and uncompressed belief as an oracle to locate loss due to compression.
7. **Acquire independent references before calibration/promotion.** Machine reviewers may propose candidates or detect anomalies, but cannot turn their own outputs into independent ground truth. Separate exploratory results from a genuinely fresh participant assessment.

No step guarantees a groundbreaking result. The most defensible innovation is to combine persistent identity-aware observations, explicitly identifiable belief components, task-specific continuation and event outputs, and distributionally correct causal distillation. Its merit must be established by the factorized experiments above, not by the number of mathematical components assembled.
