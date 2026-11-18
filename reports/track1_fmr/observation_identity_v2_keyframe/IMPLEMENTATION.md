# Keyframe observation–identity revision

This parallel experiment preserves the v1 results, qualified Stage 1 extractor, and frozen spline/LK sources. It uses local CPU inference; no dependency/model download or LM Studio reviewer is involved in this benchmark.

- Two detector heads (prompt-free and cached 24-label text prompts), confidence 0.10; normalized polygons rasterized directly on 160×160 masks. V1 already used this moment grid, so native mask upsampling was not an established v1 bottleneck.
- A detector call every six frames, plus empty-set, complete active-flow loss, or unexplained-background difference triggers. Individual weak regions do not force every-frame detection.
- Eight foreground-interior features at 320×320; both LK directions must succeed, FB error must be below 1.5 pixels, and four correspondences must remain. Actual median displacement conditions a CV prior. Failed anchors are discarded. Predictions receive no observed-point credit.
- Query transport uses inverse-distance weighted local anchor displacement; detector-box affine transport is an explicit coarse fallback when correspondences fail. Detector reseeding changes the local interpolation basis, without a new reference query. Incremental local offsets survive reseeding.
- CV association is independent of erroneous flow shifts; Lab rescue and bounded lazy SAVPE comparisons are available. Forty active / 64 dormant candidates and ten-second dormant TTL are explicit heuristic limits. Covariance inflation is capped; it is not a calibration certificate.
- Split/merge edges are provisional spatial hypotheses. Strict-ID and model-only lineage variants are both retained. Later reference coordinates never choose a branch.
- The new native cohort exports only actual detector or verified LK measurements, current support shapes/boxes and observed hit counts. Calibration excludes P08/P18 from development fitting; an old fit supplies initialization only, followed by a new development-only fit. Model-reviewed labels remain silver, `human_adjudicated=false`.

Ingestion timing includes both detector heads, LK, association and runtime bookkeeping. It excludes separate cold/cache warmup, source decoding and query/reference evaluation. VISOR detector-only probes are reported separately from scheduled causal tracker support; those probe detections never enter the tracker.

Numerical tests, source snapshots, independent denominator/transport replay and a passed software verifier do not imply acceptance of the requested accuracy, speed, calibration or physical-identity gates. See `assessment.json` and the separate calibration result for the measured gate outcomes.
