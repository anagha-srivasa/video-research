# v3 implementation and evidence boundaries

V3 is an opt-in revision of `src/track1_fmr/keyframe_identity.py`; the v1 path and v2 source snapshots/results remain available. Stride eight selects v3 in the benchmark; `--revision` can select it explicitly.

Implemented changes:

- One local `yoloe-26s-seg.pt` detector at 480 pixels and confidence 0.10, with the supplied 44 prompts encoded once into a content-addressed local cache. No prompt-free detector is constructed or executed in v3. Optional bounded SAVPE re-ID remains separate, and its runtime cost is included.
- Detector calls on an eight-frame schedule or an empty active set. Flux and flow-loss triggers are disabled. Empty-track calls can exceed the nominal 12.5% detector rate; 14% is a hypothesis, not an enforced limit.
- One 320×320 grayscale pyramid, optionally supplied by the caller. The carrier does not concurrently run a dorsal extractor; no duplicate-dorsal preprocessing saving is claimed. The qualified dorsal extractor is unchanged.
- Up to 12 texture features inside a one-pixel-eroded 160×160 mask and its distance-transform interior. No fallback to the uneroded mask is used when erosion removes a thin support.
- Batched 9×9 forward/backward LK, FB error below 1.2 pixels, at least four surviving correspondences, and 5×5 patch NCC at least 0.65 against each anchor's detector-keyframe reference. Patches use subpixel extraction. Textureless patches fail photometric verification. Failed anchors are discarded, never shifted by predictions and later resurrected.
- LK-supported means move by measured median displacement and velocity is displacement/source-time difference. Covariance follows continuous-white-acceleration CV propagation, plus recorded displacement scatter/quantization and 0.015²·dt position diffusion. No LK conditioning or covariance cap is applied in v3. Only fresh detector observations condition the filter.
- Coasting candidates can receive a bounded SAVPE appearance rescue; its spatial reach grows with predicted positional uncertainty. Strong appearance can compete with a geometric assignment. This is provisional association, not physical identity certification.
- Strict-ID and model-only causal lineage point scores remain separate. Initial queries bind current measured support; later reference coordinates do not enter point transport or branch selection.
- Native archives additionally retain LK displacement/time, FB inlier count/error, and minimum NCC for independent evidence and covariance-equation checks.

Calibration replay was updated alongside the carrier: detector innovations alone fit the development objective, LK rows propagate recorded motion and diffuse covariance, and scalar replay independently checks the batched implementation. Replaying every LK row as an independent Kalman observation would undo the architectural change. P08/P18 remain excluded from development fitting; the old fit is initialization only.

Corrections to the pasted diagnosis:

1. V2 did not condition on its own current prediction. It used `z = previous_mean + measured_displacement`, against a CV prior. Its innovation was measured displacement minus predicted displacement, which is not identically zero. Dependence between successive LK-derived positions remains a real statistical concern and motivates testing open propagation.
2. V2 already sampled distance-transform interior features and used a 21×21 LK window, rather than the alleged boundary-only sampling and 15×15 window.
3. The keyframe carrier did not run `DorsalFluxExtractor` concurrently. V2 already had a single 320×320 grayscale buffer.
4. A 9-pixel window at 320 resolution spans 7.2 pixels at 256 resolution; the pasted 45-pixel conversion was incorrect.

FB consistency and patch similarity do not prove that an anchor stayed on the same physical entity or that an object is unoccluded. Model-reviewed references remain silver (`human_adjudicated=false`). Numerical gate success alone would not create human identity/contact ground truth. Stage 3 remains gated on the measured Stage 2 criteria.
