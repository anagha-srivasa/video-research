# Track 1 independent-reference milestone

## Designated machine reviewer

Qwen3.8 27B is now the designated machine reviewer. Its read-only, frame-wise results are at http://127.0.0.1:8767; see [TRACK1_QWEN_REVIEW.md](TRACK1_QWEN_REVIEW.md). The adaptive v3 workflow checks every native frame and schedules Qwen on selected evidence; separate progress counters prevent diagnostic coverage being mistaken for semantic completion. Preserved observations and new machine-only outputs remain separate from human references. The human review/adjudication protocol below remains a distinct evidence path. Machine completion does not satisfy independent-reference or scientific acceptance gates.

The user is reviewer A. Reviewer B and a distinct adjudicator remain unassigned. The initial queue contains 163 contact cases plus 20 motion/identity episodes. Detailed per-case raw-only instructions are in `references/track1/CASE_GUIDE.html`; the practical workflow is in `references/track1/REVIEW_GUIDE.md`.

Start the local workspace:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 \
 python3 -m track1_reference.review --role A --port 8765
```

Open http://127.0.0.1:8765 . Contact reviews use a guided Watch → Find area → Hand edge → Object edge → Finish workflow, with autosaved editable drafts. Playback streams byte ranges from the original MP4; annotation uses a separate fixed native frame. Hidden edges can be marked unavailable without inventing a line.  Role B and adjudication use separate operator-started services (`--role B` or `--role adjudicator`); no role selector or other-review route is exposed to reviewer A. This is operational blinding on the local machine, not protection against a filesystem owner. Prior candidate exposure is recorded per case.

Native RGB is served losslessly in memory. Where an existing NPY shard remains, the indexed native frame is used. Where copied shards are absent, an isolated decoder seeks the exact source PTS in the original MP4, after a source checksum check. It refuses approximate substitutes. No RGB copies are restored or recompressed; PNG responses are not stored on disk.

`src/track1_reference` contains the protocol/validation, loopback service, reviewer UI, independent evaluator and separate event detector. Private source mappings and scratch scripts are in `work/track1_reference`. Human submissions are under `references/track1/{A,B,adjudicated,source_audit}`. Candidate predictions are never imported into the reviewer service.

After review submissions, update reference readiness and run gated evaluation:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 \
 python3 -m track1_reference.evaluate
```

Every planned contact case remains in the ledger: pending references, unscorable references, missing model geometry and scored cases remain distinct. Conditional boundary quality is reported alongside reference-eligible coverage and coverage-weighted boundary scores. Full versus latest-epoch support, missing geometry, independent visual conditions and source diagnoses remain separate strata. Physical 3D contact is not inferred from projected overlap.

Geometry reconstruction uses the earliest stored history ending at/after the annotated time, chosen before support filtering. This is explicitly retrospective history reconstruction, not online geometry forecasting. No later easier history is substituted when the first opportunity is unsupported.

Motion evaluation uses source-centroid observations with one past, reviewed-landmark seed offset shared by all methods. Only past observations enter CV or the unchanged switching bank. Reference targets are interpolated only across <=80 ms independently marked gaps; velocity sensitivity uses two offline reference fits. Motion onset scoring uses human intervals and explicit quiet-image-motion intervals. Contact onset is not substituted for motion onset; independent physical contact timing and full identity IDF1 remain unverified.

A separate raw-evidence detector is implemented without altering the switching position bank. Its deterministic synthetic checks are run by `python3 -m track1_reference.events`. Thresholds are fixed, and it is unpromoted. Synthetic event definitions and quiet alarms/minute differ from earlier fraction-based diagnostics; scores are not interchangeable. Independent episode results remain blocked until adjudication.

The current copied RGB partition is incomplete on disk; the earlier 238.715 GB figure describes a historical complete payload. The original source videos remain the review fallback. No claim of current complete partition readiness or clean full rebuild is made. Existing model baselines are not promoted or replaced. Fresh participants remain reserved, scientific acceptance is false, and Track 2 remains on hold.
