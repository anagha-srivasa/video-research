# Video research evidence pipeline

The original sources stay in `dataset/`. Processing code is in `src/pipeline`, configuration in `configs`, rebuildable staging and temporary tools in `work`, and QA artefacts in `reports`. `canonical/v1` uses flat `<table>.<video_id>.parquet` files so each video can be rebuilt independently without a deep folder tree.

Run from this directory using the local decoder/geometry packages installed in `work/deps`:

```sh
PYTHONPATH=src:work/deps python3 -m pipeline.run --config configs/pilot_v1.yaml --stage all --gold-only --verify-reproducibility
PYTHONPATH=src:work/deps python3 -m pipeline.run --config configs/pilot_v1.yaml --stage all --resume --verify-reproducibility
PYTHONPATH=src:work/deps python3 -m pipeline.run --config configs/pilot_v1.yaml --video P04_109 --stage normalise_camera
```

The first command builds the configured gold video, P04_109. Inspect `reports/qa.P04_109.jpg` and its sample metadata before expanding to the remaining ten. Reproducibility uses an independent full rebuild in `work/repro_v1` and compares bytewise SHA-256 hashes of every Parquet table. It preserves the original output while comparing it. Completed stages are skipped only when the code, configuration, source hashes, and output hashes agree.

The raw audit requires complete source videos with the publisher's MD5s. `.part` files are unfinished downloads and are never treated as valid sources. The controller emits flushed timestamped stage progress. Interrupted table writes stay unpublished; the readiness flag remains false.

The M4 configuration uses VideoToolbox for full timeline decoding; hardware access must be available to the process. Set `hardware_device: null` for software decoding. The backend is recorded with the video metadata. Decoder processes are isolated from OpenCV because their macOS wheels bundle conflicting FFmpeg device libraries. A real irregular-PTS video test covers indexing and the isolated lazy reader.

The official VISOR correction script and timestamp map are pinned under `work/vendor_support`, with the upstream commit and hashes recorded in stage manifests. Corrections run on extracted staging JSONs; original ZIPs are unchanged. The two special correction dependencies are support files, not added pilot videos.

Native decoded PTS minus the first valid PTS defines physical time. Frame ordinals and provider filenames remain separate references. The official VISOR timestamp file covers sparse frames only. Dense interior frames use explicitly flagged native ordinal transport only when both sparse endpoints agree; inconsistent or missing anchors remain unresolved. Camera timestamps use the sparse VISOR-to-EPIC correspondence bridge and explicitly flagged interpolation of that correspondence clock between anchors. Camera poses themselves are never interpolated. There is no frame-number/FPS clock in the canonical pipeline.

Camera matrices retain both world-to-camera and camera-to-world directions. Candidate camera image scale matrices have `transform_valid=false` until crop/field-of-view agreement is verified. Reconstruction units are not certified metres. Manual annotation dimensions are checked against official sparse JPEG headers, fetched with bounded ZIP ranges; dense annotations retain the publisher's 854×480 grid. Annotation-to-native transforms and their evidence are explicit. The reader exposes candidate annotation-to-camera transforms with a separate validity flag. Camera clock estimates, native alignment error and local tolerances remain separate fields.

Masks preserve original polygon values, manual/endpoint/interpolated provenance, open names, source IDs and exhaustiveness. Self-touching polygons can receive a separately recorded GEOS linework repair for derived rasterisation; raw polygons remain unchanged. Every repair has a code, policy and quality flags. Unusable geometry stays quarantined and cannot be rasterized through the default reader. Raster contours preserve components and holes. No B-splines, seam choices, coefficients, learned confidence, neural tracking or depth is generated.

Source identities remain provisional `(video, subsequence, source ID)` scopes. When subsequence association is unavailable, scope narrows to the interpolation chain. Missing observations are not absence. Contact changes are interval-censored observation changes; event-evidence records are not latent event ground truth. Actions remain overlapping intervals and keep `action_split` independent of `visor_split`.

In the full pilot audit, all 699,496 proposed source-ID scopes were single observations. These mask IDs therefore cannot be used as persistent tracks. Frame-local relations remain useful, but Track 1 temporal fitting requires an independently validated association stage. First/last-observed evidence is an observation extent, not an assertion that an object appeared or disappeared.

Read the frozen store with:

```python
from pipeline import Dataset

with Dataset() as dataset:
    sample = dataset.get_window("P04_109", end_time_us=144_000_000, duration_us=1_000_000)
    rgb = dataset.get_frame(sample.video_id, int(sample.frame_timestamps_us[0]))
    mask = dataset.get_mask(sample.mask_observations[0]["mask_uid"])
```

The reader refuses a store that has not passed readiness gates by default. `require_ready=False` is reserved for explicit QA. RGB and raster masks are decoded on demand into a bounded memory cache. No persistent full-frame tree is created. Windows use half-open physical intervals; action overlap queries remain independent of frame labels.

Readiness requires all 11 videos, source integrity, explicit alignment/geometry exception handling, reference/split/pose checks, lazy decode and window-query checks, independent reproducibility, and recorded visual review tied to the generated artefacts. See `canonical/v1/status.json` and `reports/qa.*.json` for actual status. A populated folder alone does not mean ingestion-ready.

Run contract checks with:

```sh
PYTHONPATH=src:work/deps python3 -m pytest -q -p no:cacheprovider --basetemp work/test_tmp
```

Source data and derived evidence retain CC BY-NC 4.0 attribution; see `dataset/README.md`, the original source notices, and `canonical/v1/provenance.json`.

The completed freeze contains 11 videos (7,334.244433 seconds), 390,570 native frames, 699,496 mask observations, 6,084 contact relations, 2,377 action intervals and 29,342 physical windows. All 121 Parquet tables matched independent full rebuilds. Source-integrity checks, nine contract tests, 99 live reader cases and sampled visual review of every video passed. Actual counts and hash-bound verification are in `reports/readiness.json`.

Training adapters must explicitly handle the 54,859 quarantined masks and 8,857 annotation frames with unresolved timestamps. Timed window queries omit unresolved observations because they have no accepted timestamp; they remain queryable in the tables. Mask queries retain quarantined records, while `get_mask` rejects them. Filter `geometry_status != "quarantined"` before raster losses, preserve repair/clipping flags, and distinguish the 18,742 manual masks from endpoint and automatic masks. Do not interpret missing segmentation as a negative label.

The evidence API is ready for task-specific research ingestion. Temporal Track 1 fitting still requires independently validated object association; source scopes must not be treated as tracks. Geometry-dependent experiments must reject `camera_transform_valid=False`. Model-specific features and a training loop remain separate derived work.

`configs/requirements.lock.txt` records the tested package versions on Python 3.13/macOS arm64; install them into an environment with `python3 -m pip install -r configs/requirements.lock.txt`. The current local command prefix remains `PYTHONPATH=src:work/deps`. `reports/implementation_snapshot.tar.gz` preserves the production code, configuration, tests and pinned vendor support used for this freeze. Raw sources are read-only; staging and temporary scripts remain in `work/`.

Track 1 is now implemented and fitted end to end on the pilot. Its versioned outputs are in `derived/`, with 41,015 valid functional states and participant-separated sequence indices. See [TRACK1.md](TRACK1.md) for the implementation, commands, tested reader example, measured results and limitations. The canonical source store remains architecture-neutral and unchanged. `reports/track1_readiness.json` records the completed Track 1 gates.
