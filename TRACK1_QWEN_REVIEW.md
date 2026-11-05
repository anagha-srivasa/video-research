# Track 1 adaptive Qwen review — active v3

The active workflow is `derived/track1_qwen_review_v3`. Open **http://127.0.0.1:8767** and refresh an already-open dashboard. The original exhaustive run is preserved in `derived/track1_qwen_review`; it is no longer the running worker.

All **592 completed legacy reviews** were saved with a consistent SQLite backup, including their raw model responses, flags and provenance. `work/track1_vlm/preserved_manifest.json` records the backup path and SHA-256. A row-by-row comparison confirmed that all original review records are identical in v3. Human A/B/adjudicated files and frozen Track 1 stores are separate and unchanged by this workflow.

## What runs now

1. Original MP4 hashes and task/frame-map hashes are checked. Exact PTS decoding refuses approximate substitutions.
2. Every one of the **6,480 native frames** receives luminance, image-change and forward/backward optical-flow diagnostics. Native-resolution attention-region feature tracking uses previous direct machine observations where available. It expires after 0.5 seconds and stops at consistency failures. This is an attention-region proxy, not verified object identity, segmentation or physical motion.
3. The 183 case cues form 81 distinct clip/kind/cue groups. Identical visual questions can share evidence at the same timestamp. Matching class names never establish that source instances are identical; shared records are flagged accordingly.
4. Qwen is scheduled for original query targets, one-second sentinels, context endpoints, and representative boundaries/peaks of motion or support-loss episodes. Gaps up to 100 ms are merged to avoid repeatedly asking about a jittering diagnostic failure. All native diagnostic measurements remain recorded; selection does not certify that every physical event was detected.
5. The resulting plan contains **1,122 semantic questions**, of which **206 were already covered by preserved observations**, leaving 916 additional questions at cutover. This replaces the 31,800-batch dense sweep. It is a different evidence workflow, not a changed percentage denominator for the same exhaustive semantic task.
6. Each selected request uses one case, native RGB, exact timestamps and the original neighbors three frames before/after. The installed Qwen3.8 27B 4-bit model, 16,384 context, temperature zero and **8,192-token reasoning/output allowance** remain. No reasoning-off setting, image downscaling, model precision reduction or generative enhancement was introduced.
7. Close-ups are conditional: a small visible target or a decisive contact claim can require another view. Hidden/ambiguous surfaces stay unknown. A decisive new physical-contact claim without a valid recheck is not accepted. Model responses, normalization notes, uncertainty and cross-view conflicts are retained.
8. At clip boundaries, attention-region diagnostics are refreshed using newly available anchors. Gray-pixel hashes must match the original scan; existing global diagnostics and semantic labels are not overwritten by tracking.

Native images are streamed through bounded memory. Two CPU clip workers each use bounded OpenCV/decoder threads; Qwen remains a single GPU inference stream. No additional RGB shards, transcoded clips or compressed dataset products are created.

## Output integrity and usefulness

The preserved ledger, native checks, attention diagnostics, selected-question plan, raw responses and failures coexist in one uncompressed SQLite database. Existing observations are never replaced. Single-case requests remove model-generated case IDs from the transport contract. Parsing may retain a single uncertainty string as a one-item list or remove unusable localization coordinates with an explicit uncertainty flag; it does not guess a coordinate scale or fabricate geometry. Truncated, malformed or semantically invalid responses fail closed. One bounded retry is allowed, and failures remain visible.

The current MLX backend's constrained-output pilot produced empty final messages with JSON in the reasoning field on all three tests. Those responses were rejected. Production uses final-message JSON with strict local validation and full reasoning; reasoning text is never accepted as a final annotation. This compatibility observation is recorded under `work/track1_vlm/adaptive_pilot.log`.

**Frame checks are not direct semantic review.** Frames without a direct or explicitly shared same-cue Qwen observation have no new contact/identity label. No temporal propagation fills those labels. The dashboard reports native diagnostic coverage and selected semantic progress separately, alongside the retained machine record count. Matching a cue does not verify source-instance identity.

These remain approximate machine observations, not independent references. They do not provide pixel-accurate boundary traces, certified physical contact/onset, IDF1/HOTA, true velocity or calibrated physical uncertainty. Scientific acceptance remains false; Track 2 remains on hold. Accuracy equivalence to exhaustive Qwen review has not been independently established. Existing review evidence is preserved and uncertainty gates are explicit; the redesign does not claim otherwise.

## Verification and performance

All **95 tests** passed. Checks cover original-record preservation, source pixel consistency, synthetic known translation, featureless-frame rejection, short detected-motion selection, typed single-case parsing, conservative contact rechecks, shared-cue provenance and tracking-refresh integrity.

The three-case single-request pilot returned three valid final replies, retained 1,260 / 3,089 / 2,289 reasoning tokens, and matched the preserved native RGB hashes. Mean elapsed time was about 125 seconds per primary request. This is an engineering pilot, not independent annotation-accuracy validation or a reliable deadline forecast. The native CPU checks took 273.8 seconds summed over 20 clips with two clip workers. The live selected workload and conditional rechecks remain a substantial background job; it has not already finished.

See `reports/track1_vlm/redesign_validation.json`, the live SQLite/status ledger, and `work/track1_vlm/adaptive_review.log` for evidence.

## Service control

```sh
python3 work/track1_vlm/service_control.py status
python3 work/track1_vlm/service_control.py pause
python3 work/track1_vlm/service_control.py resume
```

The per-login macOS launchd supervisor runs `track1_reference.adaptive_review`, restarts unexpected exits and honors explicit pauses. The dashboard is a separate read-only service. The legacy `start_frame_review.sh` now delegates to this controller, so it cannot silently start the old exhaustive loop. Old dense-worker flags are rejected. LM Studio must remain available with the selected local model.

Status, provenance and configuration are in `derived/track1_qwen_review_v3`. Logs, service plists, PID/control files and pilot scripts remain under `work/track1_vlm`. Memory guards require headroom before clip scans and model requests; they reduce OOM risk but cannot guarantee safety against unrelated system-wide allocations. The services survive Codex-session closure for the current user login; no system-wide startup files were installed.
