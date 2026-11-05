# Track 1 refinement and evaluation

The version-two store is `derived/track1_refined_v2`. It preserves the finalized architecture: 16 periodic cubic controls, invertible db2 shape coordinates, 35 state channels, four temporal Legendre coefficients, one-second windows, quarter-second hops, and frozen MobileNetV3-Small plus a train-only 128-dimensional PCA projection. `src/refinement` contains the fitting, calibration, build, reader and evaluation modules. Scratch scripts, caches and logs stay under `work`.

The original raw dataset, canonical store and frozen Track 1 v1 artifacts remain the baseline. The final refinement freeze rechecks every one of the 271 baseline file hashes.

## Quality changes

Spatial fitting uses 128 contour samples, robust periodic spline fitting and bounded, ordered point-to-curve correspondence refinement. It keeps the best valid fit and falls back to a previously valid baseline fit when the observed geometry gets worse. This uses only the contour being encoded. It does not use reserved-video aggregate results to tune parameters.

The temporal search compares variance/provenance weighting, robust losses, observation coverage weights and mode-specific regularization. The selected fit uses L2 loss, variance power 0.25, provenance power 0.5 and mode penalty `[0, 0, 1, 9]` with strength one third. L2 won the participant-held-out validation objective; robustness diagnostics expose its outlier sensitivity. Fit/covariance calculations stay in float64 on CPU; stored tensors use float32. CNN extraction uses Metal, bounded batches and three workers, with at most one transient feature map batch per worker.

Calibration uses training videos only. Spatial ribbon calibration measures held-out contour representation error, conditional on usable geometry and estimated correspondence. Temporal multipliers calibrate marginal channel errors and protect endpoint coverage. These are approximate empirical covariances, not independently validated human-label uncertainty or a full joint Gaussian state model.

## Reserved-video results

| Metric | Frozen baseline method | Refined | Change |
|---|---:|---:|---:|
| Withheld contour normal RMSE, native pixels | 5.9733 | 3.9533 | 33.8% lower |
| Full-mask reconstruction IoU | 0.8874 | 0.8968 | +0.0094 |
| Strict 2-pixel boundary F-score | 0.5674 | 0.6210 | +0.0536 |
| Single-frame temporal holdout, normalized RMSE | 0.5982 | 0.5722 | 4.3% lower |
| 100 ms internal gap, normalized RMSE | 0.6622 | 0.5804 | 12.4% lower |
| 100 ms endpoint holdout, normalized RMSE | 1.6062 | 1.0574 | 34.2% lower |

Temporal comparisons isolate the fitter: both methods receive the same refined states, variances and retained observations. The denominator is a per-channel robust scale learned on training videos; results are means of per-video RMSE. Targets are removed before fitting. Interior holdout centers vary deterministically across 0.25, 0.5 and 0.75 seconds. Endpoint targets occupy the final 100 ms, after all retained observations. These are local functional-reconstruction tests, not an RGB forecasting benchmark.

Nearest and linear interpolation still outperform the compact cubic on several holdouts. Those methods retain all supplied knots. CubicSpline and PCHIP can be unstable when extrapolating near irregular/outlying measurements; endpoint extrapolation is reported explicitly. The four-coefficient capacity constraint and independent identity/velocity evidence are not resolved by running more optimization on the same data. The validation search stopped after two consecutive rounds with less than 0.2% objective improvement. Reserved test scores never select parameters.

## Split and ingestion contract

The source-compatible tuning split is five training, two validation and four test videos, with disjoint participants. Training is P02_122, P03_112, P04_109, P11_16 and P22_117; validation is P08_09 and P18_03. Official VISOR/action assignments and the old engineering split remain separate metadata. Use the new `indices/{train,validation,test}_index.parquet`, whose `tuning_split` follows this policy. Never substitute the old engineering split for the new training split.

The store contains 603,041 observations, 94,216 attempted windows and 42,349 valid windows. Sixteen-state consecutive sequences number 4,344 training, 771 validation and 3,102 test. Invalid and unsupported windows remain recorded; training consumers must filter `valid` and check `appearance_valid`. The overall attempted-window yield must not be confused with yield among independently coverage-eligible windows. Provisional annotation-chain identities are preserved; there is no independently certified IDF1/HOTA result.

```python
from refinement.store import RefinedDataset
import pyarrow.parquet as pq

sequence = pq.read_table(
    'derived/track1_refined_v2/indices/train_index.parquet'
).to_pylist()[0]
with RefinedDataset() as ds:
    state = ds.get_state(sequence['video_id'], sequence['entity_track_id'],
                         sequence['start_us'] + 1_000_000)
    assert state['coefficients'].shape == (4, 35)
    assert state['coefficient_cov'].shape == (35, 4, 4)
    assert state['xi'].shape == (4, 68)
```

Endpoint velocity and acceleration are physical-time derivatives. The GL4 collocation tensors retain the same contract as v1. The reader memory-maps at most two video shards. Camera coordinates are intrinsic-normalized 2D; this is not metric 3D scene stabilization.

## Evaluation and readiness

Open `reports/track1_refinement/dashboard.html` for the filterable manual-reconstruction view and staged gates. `scorecard.json` covers all 12 families from `configs/track1_evaluation_brief.txt`, distinguishing measured metrics, proxies and missing references. Parquet files provide per-video, per-window and per-mask evidence. `quality.png` plots temporal errors, contour quality, coverage, rate–distortion, robustness and memory. The eleven contour sheets show source masks and both fits, including small, disconnected and difficult shapes.

Read `derived/track1_refined_v2/status.json` before ingestion. `ingestion_ready` certifies numerical/artifact gates. `scientific_acceptance_gates_passed` is separate and remains false until the proposed independent-reference gates pass. The strict boundary score is below the brief's proposed 0.85 gate. True endpoint velocity/acceleration, contact-event retention, tracking identity accuracy and statistically valid overlap covariance consistency remain unverified. Full-mask errors include absent holes and disconnected components. Static-scene compensation metrics are inapplicable to the current depth-free 2D coordinate mode.

Public VISOR video object segmentation predicts unseen masks from RGB given an initial mask; this encoder receives the target contour. Its reconstruction scores therefore do not establish public VISOR benchmark parity. See the official benchmark definition: https://github.com/epic-kitchens/VISOR-VOS . The proposed scorecard thresholds are engineering targets, not published VISOR standards.

## Reproduction

From this project root, use the locally installed runtime:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m refinement.run --stage all --resume
```

The workflow reuses locked selection after test exposure. Individual stages are listed by `python3 -m refinement.run --help`. `evaluate` runs matched holdout comparisons; `diagnostics` runs the expanded scorecard and exact sampled numerical refits. `freeze` requires passed tests, unchanged reviewed images, matching locked parameters, finite tensors, valid schemas, analytical derivative/collocation reconstruction and unchanged baseline hashes. Regenerated visual sheets require review if their hashes change.

Run tests with a temporary directory inside this root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest tests -q --basetemp=work/track1_refinement_pytest_rerun
```

Resource guards stop before summed process-tree RSS exceeds 20 GiB or system available memory falls below 5 GiB. RSS includes shared mmap pages and is not unique physical allocation. The immutable release manifest and source/evidence snapshot are under `reports/track1_refinement`; continuously appended resource logs remain outside that freeze.

The expanded scorecard records scale-aware test boundary F-score 0.8399 (1% bounding-box diagonal, minimum one annotation pixel), median nonconstant-channel in-sample R² 0.6545, manual spatial 95% coverage 0.9556 and no significant covariance PSD violations. Coverage-eligible yield is 99.92%; yield over all attempted windows is 44.95%, because unsupported windows are retained in the ledger. Geometry-only feature-cell foreground precision is 0.8070, below the proposed 0.95; 88/528 sampled manual masks contain no foreground at any 8×14 feature-cell centre. This identifies a small-object/resolution limitation, not a certified CNN background-leakage score.

The all-observation cubic compression lower bound is a separate diagnostic that sees every input target. Stored normalized RMSE is 3.19% above that lower bound on validation and 3.73% above it on test. It is not a held-out prediction oracle. The current fit is close to the best cubic on this in-sample objective, while true short-event fidelity remains unverified.
