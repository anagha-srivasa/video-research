# Causal neural observation evaluation

This is a frozen published tracker comparison, not a claim that our research architecture has surpassed the published model. Track 1 baselines remain unchanged.

Completed: **3 / 30 TAP-Vid DAVIS videos**, **0 / 20 existing native context episodes**. Pipeline state: `pilot`. Partial aggregates change as the fixed alphabetical benchmark cohort finishes; no parameters are selected from these scores.

| Tracker | Videos | Average Jaccard | Average position accuracy | Occlusion accuracy |
| --- | --- | --- | --- | --- |
| LK_forward_backward | 3 | 0.4331 | 0.4896 | 0.6498 |
| TAPNext++512 | 3 | 0.6960 | 0.8536 | 0.8736 |


The metric implementation is the exact NumPy `compute_tapvid_metrics` function from the pinned [official TAP repository](https://github.com/google-deepmind/tapnet), commit `ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b`. All first-visible query points are used. Queries are supplied to the official recurrent network using its future-query masking; no future image is supplied during a step. Scores use the official 256-coordinate metric space and per-video averaging. The published 480-pixel-high source images are bilinearly resized to 512 square for the neural model. LK receives the same source frames at their supplied resolution. This is a documented local reproduction, not an exact match to every published implementation setting.

TAPNext++ uses the official 512 checkpoint, checked against publisher MD5, with float32 weights and inference. No half precision, quantization, skipped native frames, stochastic checkpoint selection or validation fine-tuning is used. A tested alternative attention path produced identical coordinates in the integration probe but was slower, so the original path is retained. Current device throughput is approximately one frame per second; this is an offline observation generator, not the intended low-latency deployed encoder.

The native experiment starts from the exact same manual seed, persistent query identities and PTS ledger as the preserved LK experiment. Only subsequent RGB frames enter tracking. Reappearance is a neural prediction, not certified physical identity. Both neural and local-LK observations use the same material-point transport rule. 0 / 0 completed native frames currently support material transport; this is support, not a native-domain accuracy result. No native RGB files are duplicated.

The publisher's TAP-Vid annotations provide an independent point reference relative to predictions. They do not validate the project-specific hand/object identities, contact geometry, velocity uncertainty or scientific acceptance gates. Published benchmark data are also not a genuinely fresh final evaluation for a public checkpoint.

Each video is atomically checkpointed with prediction and protocol hashes. Interrupted work resumes from completed videos. A process lock prevents duplicate GPU jobs. Memory guards check process RSS, Metal driver allocation and available system memory before each frame; execution stops at 18 GiB RSS/driver allocation or below 6 GiB available. Metal's allocator is additionally capped at half its recommended working set. The full benchmark and native stages run sequentially.

## Reproduction

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -u -m track1_neural_observation.run --stage all
```

Metal requires host GPU visibility. `work/track1_neural_observation/monitor.py` runs this command with live checkpoint counts, a log, and periodic report refreshes. See `run_state.json`, `summary.json`, stage-specific protocol files and `derived/track1_neural_observation` for evidence. The model download approval failure from the earlier attempt was resolved on retry; the comparison is no longer blocked by that failure.
