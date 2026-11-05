# Training ingestion checks

The preparation package is `../../derived/track1_training_ready`. See [loading and reproduction](../../TRACK1_TRAINING_READY.md).

All 2,096 original machine-review rows were preserved exactly. Selected review questions: 834 completed, 206 preserved, 82 unresolved. Full native diagnostic coverage is not full semantic coverage. The 58,800-entry case/frame ledger retains absent evidence explicitly.

Preprocessing produces participant-separated, gap/epoch/episode-safe indices for lengths 1/4/8/16, referencing unchanged fitted tensors. Native frames are source-backed: 33,636 indexed frame identities/PTS were checked against canonical inventories. Original video SHA-256 values match for all seven development videos. Twenty-one actual native RGB decodes and twelve reader batches were sampled.

Checks passed for all 322 tensor files, 5,334 full-history certificate bindings, 2,564 latest-epoch certificate bindings and 6,984 sequence indices. Covariance PSD and analytical collocation reconstruction checks passed. All 97 tests passed; two targeted ingestion tests were rerun after the SQLite closure fix. All package artifact hashes and a real length-16 batch were checked again after readiness was written.

- [Counts and preprocessing](preprocessing.json)
- [Full verification and dependency hashes](verification.json)
- [Readiness flag](../../derived/track1_training_ready/status.json)
- [Release manifest](../../derived/track1_training_ready/manifest.json)

This certifies retrospective Track 1 numerical reconstruction ingestion, not independent scientific acceptance or physical contact/motion ground truth. No model training, promotion, final freeze or Track 2 work occurred.
