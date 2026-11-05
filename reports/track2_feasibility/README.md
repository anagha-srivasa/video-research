# Bounded Track 2 feasibility: NO-GO

Open [dashboard.html](dashboard.html) and [the implementation guide](../../TRACK2_FEASIBILITY.md).

The user approved the frozen six-second context cohort after the report's base-v2 M16 counts and future-target assumptions failed the audit. Twelve final models and sixty training-participant stopping fits are complete. All Track 1 hashes are unchanged.

Stable Koopman is 29.3% worse than CV at 250 ms, 20.5% better at 500 ms and 45.1% better at 1 s. The mandatory 250 ms and seed-regret gates fail. No architecture expansion, optional ablation or full Track 2 programme was started.

- [Approved protocol amendment](protocol_amendment.json) and [cohort/target availability](adapter.json)
- [Every opportunity](target_ledger.parquet), [all forecasts](predictions.parquet), [participant metrics](participant_metrics.parquet)
- [Locked gates](decision.json), [all model comparisons](comparison.json), [stable/plain spectra](spectra.json)
- [Training-only epoch selection](selected_epochs.json), [final training records](training.json)
- [Frozen inputs](frozen_inputs.json), [environment](environment.json), [numerical verification](verification.json)
- [Exact repeat](repeat_best.json), [final ledger/hash audit](final_audit.json), [resource summary](completion.json)
- [Inference cost](inference_latency.json): cached-state forecasting only, not raw RGB inference

Independent physical labels, contact/identity/event acceptance, independent forecast uncertainty and fresh participant evaluation remain absent. Scientific acceptance is false.
