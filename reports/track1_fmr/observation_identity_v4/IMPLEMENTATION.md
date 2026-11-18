# V4 implementation and measurement contract

The benchmark selects v4 for `--keyframe-stride 10`; `--revision v4` is also supported. Earlier algorithms and immutable experiment source snapshots remain available. The qualified dorsal extractor and frozen spline/LK modules are unchanged.

Implemented changes:

- The permitted **single prompt-free head** option: `yoloe-26s-seg-pf.pt`, confidence 0.10, one segmentation pass per keyframe. No additional manipulator prompt pass or text encoding runs. Optional bounded SAVPE appearance rescue is separate and its cost remains included.
- Detector size is `min(480, max(32, ceil(max(H,W)/32)*32))`. DAVIS 256×256 inputs stay at 256; native 1920×1080 uses 480. Only the fixed stride-ten schedule triggers detection, including when no tracks exist; empty/flux/flow-loss bursts are disabled.
- LK uses one shared grayscale image with maximum span 320, retaining native 256×256 DAVIS images. Eight texture features are sampled inside current mask support, without erosion or NCC rejection. Three FB-valid anchors suffice; the window is 13×13 and FB error must be below 1.5 pixels. Failed anchors are discarded rather than predicted forward and recycled as measurements.
- All eligible anchors and direct query particles are concatenated into **one forward call plus one backward call**. FB verification requires two directional OpenCV calls. V3 already batched carrier anchors; no saving from eliminating hypothetical per-carrier calls is claimed.
- Small-support growth guard uses actual normalized support mass (mask area or rectangle fallback), not an unlabelled determinant proxy: mass <0.06 cannot match a detection above three times that mass. Both geometric and appearance-rescue routes enforce it, including dormant candidates.
- Contained subparts below 0.65 of prior mass, shape Mahalanobis distance ≤5.5 and six-moment Lab distance/100 ≤0.45 can be adopted. Shape scatter blends 0.25 parent +0.75 child; position uses a genuine detector-centroid Kalman update, as in the supplied code, rather than fabricating a blended independent measurement.
- Non-shrinking LK covariance propagation from v3 is retained, with v4 diffusion 0.012²·dt. Noise remains continuous-white-acceleration spectral density. Only genuine detector centroids condition the filter.

## Point-particle protocol

`keyframe_strict` and `keyframe_lineage` remain mesh-transport controls. `hybrid_strict` and `hybrid_lineage` add directly tracked surface-point particles. A query enters only once at its first visible timestamp and only if it binds current observed support. Later reference coordinates never enter tracking, association, branch selection or re-seeding.

Particles share the LK batch with anchors, but do not contribute to carrier displacement estimates or association. A regression test compares carrier states with and without query particles. The hybrid therefore has a query-seeded initialization contract; it is not blind object discovery or a controlled model-only comparison with standalone LK's all-query initialization.

An observed hybrid point requires its own valid FB correspondence, an observed parent carrier, and position within the carrier's 2.5-sigma shape support. A failed point is not retried from arbitrary predicted coordinates. Its geometric fallback is transported incrementally by surviving mesh correspondences or a fresh detector box transform, without assuming fixed anchor slots survive refresh. A recovery seed can be created only by current detector-supported geometry. That seed is **predicted**, not observed; a later successful image-correspondence step is required before it earns observation or recovery credit. This deliberately avoids the supplied code's unconditional retry of unlocked predicted points.

DAVIS timing includes particle LK, validity gating and recovery handling. Initial query binding and metric scoring are outside ingestion timing. Native timing has no query particles and remains a query-free ingestion measurement. Sparse VISOR probes never inject their detections into the causal tracker.

## Calibration reporting

The v4 replay uses actual detector observations, recorded LK displacements and v4 diffusion; only detector innovations fit the development objective. P08/P18 remain excluded from fitting. The old fit is an initializer, followed by a fresh development-only fit.

The unconditional silver NEES and existing qualification checks are retained. Added fields are:

- In-support proxy: native position error ≤50 pixels **or** shape-scatter Mahalanobis distance ≤6.25.
- In-support binding rate over all eligible high-confidence visible labels, with missing/unbound states failing; the rate among evaluable states is reported separately.
- In-track mean NEES restricted to that in-support subset, with its count and `in_track_nees_reference_conditioned=true`.

The subset uses reference errors and can look better by construction. It does not prove association identity or covariance calibration and cannot replace the unconditional qualification gate. Broad shape support can also pass a geometric proxy without proving correct identity. Model-reviewed labels remain `human_adjudicated=false`.

## Audit corrections

V3 already made one batched forward/backward pair for carrier anchors and never fell back to ellipse anchors after erosion. Its source did not call `containment_association_cost`; causal giant-mask hijacking was a hypothesis, not an independently established explanation for every silver error. V4 explicitly tests area guards and subpart adoption, but this combined revision is not an ablation proving which individual mechanism caused a metric change.
