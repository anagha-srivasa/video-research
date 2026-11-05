# Track 1 readiness remediation

Track 1 remains open development; Track 2 is on hold. Frozen baselines are preserved.

The 19,974 model/window attempts represent 9,987 source opportunities, not distinct clips. Source-derived timing eligibility is not independently preregistered coverage.

The six-second supplement is targeted, not a random dataset sample. Parameters and appearance PCA are locked from v2; this is a coverage extension, not a new hyperparameter search.

Sequence counts are overlapping windows, not independent annotated episodes. No sequence may bridge a missing hop or source/correspondence epoch.

Transport preserves the certified anchor curve under a separate positive similarity pose. It does not certify future physical geometry, identity, covariance or raster topology.

Canonical transport and source-polygon CV use the identical motion estimator. Neither is a new learned dynamics model. Source observations are retrospective, so this is not online RGB forecasting.

Longer context covariance calibration is pending. Independent motion/contact/identity references and genuinely fresh locked evaluation remain absent.

Dataset product: 238.715 GB / 250 GB, uncompressed.

## Yield

[
  {
    "model": 16,
    "proposed": 9987,
    "timing_supported": 2633,
    "valid": 2608,
    "valid_per_timing": "99.05%",
    "epoch_failures": 0
  },
  {
    "model": 32,
    "proposed": 9987,
    "timing_supported": 2633,
    "valid": 877,
    "valid_per_timing": "33.31%",
    "epoch_failures": 1749
  }
]

## Sequence counts

[
  {
    "model": 16,
    "split": "train",
    "length": 4,
    "sequences": 1081,
    "complete_certified_sequences": 436
  },
  {
    "model": 16,
    "split": "train",
    "length": 8,
    "sequences": 762,
    "complete_certified_sequences": 284
  },
  {
    "model": 16,
    "split": "train",
    "length": 16,
    "sequences": 266,
    "complete_certified_sequences": 84
  },
  {
    "model": 16,
    "split": "validation",
    "length": 4,
    "sequences": 236,
    "complete_certified_sequences": 50
  },
  {
    "model": 16,
    "split": "validation",
    "length": 8,
    "sequences": 153,
    "complete_certified_sequences": 19
  },
  {
    "model": 16,
    "split": "validation",
    "length": 16,
    "sequences": 46,
    "complete_certified_sequences": 0
  },
  {
    "model": 32,
    "split": "train",
    "length": 4,
    "sequences": 480,
    "complete_certified_sequences": 460
  },
  {
    "model": 32,
    "split": "train",
    "length": 8,
    "sequences": 320,
    "complete_certified_sequences": 303
  },
  {
    "model": 32,
    "split": "train",
    "length": 16,
    "sequences": 110,
    "complete_certified_sequences": 99
  },
  {
    "model": 32,
    "split": "validation",
    "length": 4,
    "sequences": 67,
    "complete_certified_sequences": 60
  },
  {
    "model": 32,
    "split": "validation",
    "length": 8,
    "sequences": 29,
    "complete_certified_sequences": 28
  },
  {
    "model": 32,
    "split": "validation",
    "length": 16,
    "sequences": 5,
    "complete_certified_sequences": 5
  }
]

See [dashboard](dashboard.html), the per-video/model Parquet ledgers, and the independent-review contract. All 69 tests passed; 21 sampled RGB frames matched independent decode exactly.
