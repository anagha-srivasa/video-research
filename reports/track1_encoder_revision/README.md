# Track 1 report implementation and evidence

Open [dashboard.html](dashboard.html) and [the full implementation/loading guide](../../TRACK1_ENCODER_REVISION.md).

The completed work preserves the M16 carrier and old decoder, tests four retained-only phase variants, adds a causal innovation channel, performs localized certified repair, propagates conditional full covariance, and builds three new source-backed contexts with frozen appearance extraction. All 108 tests pass. The empty-root bounded rebuild matched all 75 data artifacts exactly.

- [Phase results and selection](phase.json), [all attempts](attempts.parquet), [entity/ring strata](phase_strata.parquet), and [per-probe target-exposed oracles](phase.parquet)
- [Local repair results](repair.json) and [failure/localization ledger](repair.parquet)
- [Synthetic/real event diagnostics](events.json), [training-only threshold selection](event_parameters.json), and [episode evidence](event_synthetic.parquet)
- [Conditional covariance scope](covariance.json) and [real fitted-contour details](covariance.parquet)
- [Stable-context selection](context_plan.json), [refit/continuity](contexts.json), and [exact-frame appearance extraction](context_appearance.json)
- [Strict independent-reference audit](independent_references.json)
- [Default 512-point inference component measurements](inference_default_resolution.json)
- [Numerical verification](verification.json), [clean bounded rebuild](clean_rebuild.json), and [final artifact/sequence audit](final_audit.json)
- [Scientific gates still open](scientific_readiness.json)

The original discrete gauge remains selected. Local repair improved reconstruction but recovered no additional certified states. Innovation improves over switching on synthetic recall but does not dominate CV. Geometry decoding is faster after compilation; these timings exclude RGB, masks, CNN, fitting, proof and pose projection. Scientific acceptance is false, default training artifacts are preserved, and Track 2 remains on hold.
