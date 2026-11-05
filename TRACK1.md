# Track 1 functional evidence

The controlling blueprint is copied unchanged to `configs/track1_implementation_plan.md`; its checksum is included in manifests. Track 1 fits mathematical representations and calibrates their uncertainty. MobileNet stays frozen. The neural encoder, stable generator and prediction training described in the blueprint belong to Track 2 and are not trained here.

## Structure and execution

`src/track1/` contains numerical kernels, spatial fitting, source-aware tracks, camera checks, appearance, temporal fitting, calibration, evaluation, a reader and a controller. Configuration is in `configs/track1.yaml`; tests are in `tests/test_track1.py`. Temporary packages, decoder requests, pooled feature caches and logs stay in `work/`. Raw and canonical source stores are unchanged.

```sh
PYTHONPATH=src:work/deps:work/track1_deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python3 -u -m track1.run --stage all --resume
```

Stages are `audit`, `spatial`, `calibrate`, `appearance`, `functional`, and `evaluate`. `--video P04_109` supports isolated spatial/evaluation debugging. Calibration and PCA should use the complete configured pilot, because their train/validation policy is defined across videos. A final visual review and gate audit follow evaluation; executing a stage alone does not mark a freeze ready.

The controller admits CPU workers against available unified memory, uses one BLAS thread per worker for tiny solves, and permits multithreaded large PCA operations. Geometry is CPU float64. The frozen CNN uses Metal float32, frame batches of 16, bounded decode prefetch and CPU mask workers. It runs once per selected frame and pools all its entities before discarding the feature map. No RGB tree, dense SDF cache or CNN feature maps are persisted. Temporary pooled 576-D vectors support train-only PCA without a second RGB decode; final embeddings are 128-D float16.

`work/track1_run.log` records stage progress, fitting counts, throughput and calibration/PCA progress. `reports/track1_resources.jsonl` records total controller/child RSS, system available memory, CPU usage and Metal allocation when active. The configured process budget is 20 GiB with a 5 GiB system reserve; resource guards stop safely at checkpoints if those limits are crossed. Other applications can change available memory, so these controls reduce OOM risk rather than guaranteeing system-wide memory availability.

## Mathematical contract

The baseline is 16 periodic cubic B-spline controls fitted to 64 arc-length samples with three Huber IRLS iterations, scaled curvature regularisation, Cholesky and augmented QR fallback. The bootstrap seam uses the +pi/2 direction; subsequent seams are cyclic index changes that minimise centred, scale-normalised contour discrepancy. Geometry is never rotated to hide deformation.

The dominant outer ring is the fitted loop. All source components and holes remain in canonical data, with counts and flags in the spatial index. Self-intersecting fitted splines are retained with `spatial_valid=False`. Appearance support subtracts original holes and uses only the conservative uncertainty core. The halo is reproducible from the stored spline/covariance and is not used for semantic pooling.

Full 16x16 residual-scaled control covariance is retained under an isotropic independent-coordinate approximation. Normal variance is propagated through the spline basis, not read directly from coefficient diagonal entries. One-level periodised db2 is a fixed linear transform. Wavelet covariance can be regenerated exactly from the stored control covariance; temporal feature variances use a documented diagonal delta-method approximation for centre/scale uncertainty, including an area-based scale term. These are representation-error approximations, not calibrated human-label uncertainty.

Covariance calibration fits held-out contour vertices on manual masks from the intersection of engineering-train and VISOR-train videos. The 1.0/0.9/0.6 provenance values are declared reliability priors, not probabilities. Validation and test videos are excluded from calibration and PCA fitting. Manual and interpolated coverage are reported separately; interpolated contours cannot independently validate human annotation quality. Temporal calibration holds out physical observations; its neighbouring dense evidence may share upstream annotation endpoints, so it is an interpolation diagnostic, not future-prediction evaluation.

The 35-D temporal state contains intrinsic-normalised 2D centroid (2), dimensionless centred db2 shape (32), and log apparent scale (1). Appearance is separate. Weighted degree-three Legendre fits use native timestamps, time-gap quadrature, observation/provenance reliability, propagated feature variance, Huber IRLS and the fixed higher-mode prior. Functional windows use the blueprint's closed `[T-1,T]` fitting interval; canonical evidence API windows remain half-open. Derivatives carry the physical `2/T` and `4/T²` factors. The store preserves full 4x4 covariance per feature, endpoint position/velocity/acceleration, endpoint variance and position–velocity covariance. GL4 values, derivatives, 68-D derivative-aware `xi` and `xi_dot` are provided for later Track 2.

Functional windows with effective sample count below six, span below 0.6 seconds, gaps above 0.35 seconds or unhealthy conditioning remain indexed with `valid=False`. Unresolved timestamps, quarantined geometry and duplicate track-time observations have explicit exclusion records; their original evidence remains in canonical/v1.

## Identity and camera validity

Source mask IDs are observation IDs. Track identity uses unique name/class within a source interpolation chain and source subsequence. Shared endpoints may join adjacent chains only with unique same-class/name agreement, matching physical timestamps and bbox IoU above 0.6. Manual observations attach only through unique shared endpoint geometry. Long unsupported gaps split segments; subsequence changes and ambiguous duplicate nouns do not merge. Physical object lineage is not inferred from nouns: multi-component and duplicate cases remain flagged and unverified.

Camera resize checks project static EPIC-Fields points onto three native-PTS images per video. Actual RGB residuals are compared with four shifted controls and the resulting images are visually reviewed. This tests sampled pure-resize/crop consistency; it does not certify every pose or metric depth. Intrinsics and distortion are explicit; 2D centroids are distortion-corrected. Rotation-compensated world bearings are stored only with an unambiguous aligned pose. Metric 3D remains disabled.

## Artefacts and access

The derived store stays shallow:

```text
derived/
  track1_spatial_v1/       # per-video .pt, .parquet, exclusions, camera metadata, manifests
  track1_calibration_v1/   # held-out cases and learned uncertainty model
  track1_appearance_v1/    # frozen pooled embeddings and learned PCA projection
  track1_functional_v1/    # per-video functions, derivatives, covariance and collocation
  training_v1/            # participant-separated 16-state sequence indices
  track1_status.json
```

Engineering train/validation/test is grouped by complete video/participant before window generation. Both official source splits remain independent columns. The engineering split is not an official EPIC benchmark. Sequence indices expose source-compatible segmentation/action training flags; downstream scientific evaluation must respect the relevant source assignment.

```python
from track1.store import Track1Dataset

with Track1Dataset() as dataset:
    # Choose a valid row from derived/track1_functional_v1/index.parquet.
    state = dataset.get_state(video_id, entity_track_id, window_end_us)
    nodes = dataset.get_collocation(video_id, entity_track_id, window_end_us)
```

The reader uses memory-mapped per-video shards with a two-shard cache. `derived/track1_status.json` and `reports/track1_readiness.json` describe actual gates and limitations; sampled QA images are `reports/track1_spatial.*.jpg`. `require_ready=False` is reserved for explicit QA.

## Completed pilot results

The final Track 1 gates pass for all 11 pilot videos. The store contains 603,041 fitted spatial observations, 597,702 valid spatial fits, 11,104 conservative source-linked segments, and 41,015 valid functional windows out of 94,197 retained windows. It includes 592,881 usable appearance vectors.

The 16-state engineering sequence counts are 5,567 train, 722 validation and 1,271 test. These are overlapping sequences, not independent examples. Official split columns and source-compatibility flags remain essential.

The learned projection uses 316,386 source-compatible training observations; its first 128 components retain 90.2% of training feature variance. Held-out manual 95% contour coverage is 94.31% on engineering validation and 95.06% on source VISOR validation. Per-video median dominant-component IoU ranges from approximately 0.946 to 0.964 in the sampled manual-mask evaluation.

All 15 tests pass. The gold spatial and functional rebuilds match tensors and file hashes exactly. Real-frame CNN features agree between CPU and Metal, and software/VideoToolbox RGB matches exactly in four sampled codec/timestamp checks. Sampled camera and spatial overlays were reviewed for every video. Reported process-tree RSS peaked at 13.41 GiB; available system memory stayed above 19.31 GiB. No OOM occurred.

The fixed degree-three temporal representation has higher held-out interpolation error than nearest-observation lookup on this pilot. It is a continuous compression with uncertainty and derivatives, and has no demonstrated forecasting advantage. The degree and robustness ablations are recorded without changing the prescribed baseline. Small or thin objects can also have weak feature-grid cores; inspect appearance validity and core/halo mass.

A tested reader example is:

```python
from track1.store import Track1Dataset

with Track1Dataset() as dataset:
    state = dataset.get_state(
        "P02_122",
        "00247fc090178ee5a8b72c9e",
        370750000,
    )
    assert state["valid"]
    # coefficients: (4,35), coefficient_cov: (35,4,4), xi: (4,68)
```

See `reports/track1_readiness.json`, `reports/track1_diagnostics.png`, `reports/track1_ablations.json`, `reports/track1_covariance_ablation.json`, `reports/track1_identity_audit.json` and `reports/track1_freeze_manifest.json`. The implementation snapshot includes the exact source/configuration, experiment helpers and official frozen backbone weights. The raw dataset and canonical/v1 remain unchanged.

Primary implementation references: [SciPy B-splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html), [PyWavelets periodised DWT](https://pywavelets.readthedocs.io/en/stable/ref/dwt-discrete-wavelet-transform.html), and [TorchVision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
