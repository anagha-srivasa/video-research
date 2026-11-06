# Track 1 follow-up: online replay and report reconciliation

Track 1 remains open. New Track 2 development is on hold under this report; previously completed bounded prototypes and their negative results are preserved. No fitting, threshold tuning, model promotion, final freeze, fresh-data acquisition or Track 3 work was performed.

## Engineering changes

`src/track1_replay.py` provides a chronological adapter for the existing event detector. It feeds an observation only when its timestamp is at or before the decision cutoff, latches events until that cutoff, and records how many observations were received. Observation and decision clocks must be strictly increasing. Future measurement payloads are not inspected before arrival. This models timestamp availability; actual source-mask availability can be retrospective, so this is not a claim of an online RGB system.

`tests/test_track1_replay.py` checks delayed delivery, exactly-once events, future-suffix mutation, empty/unstarted streams, invalid clocks and decoder bounds. The broader audit replays the existing 90 synthetic episodes under 125/250/500 ms delivery schedules, matching the original event outputs with delayed decision timestamps. It changes no detector parameters. A passing replay establishes causality/scheduling consistency, not event recall, contact accuracy or human-label validity.

## Geometry contract

`CompiledGeometry` is a low-level canonical curve decoder. Its certificate binds coefficients, allocation, roles, parents, duration and decoder implementation. It rejects extrapolation outside the certified history. It can correctly decode a valid curve even when source completeness is false. It does not promise a complete visible mask, hidden physical surfaces, persistent physical identity, future geometry or calibrated geometry uncertainty.

Use the higher-level dataset reader with `require_geometry=True` when complete represented source geometry is required. Its `geometry_training_eligible` policy combines completeness and certification; low-level curve validity alone cannot replace that policy. Missing geometry must remain explicit. Completeness here still concerns retained source annotations, not the unseen physical object. A positive pose transform preserves the modeled curve topology, not future physical truth.

All-steps geometry availability and single-epoch continuity remain separate. A sequence may have geometry at every step and still contain topology resets. M16 numerical validity is also distinct from M32 completeness and certification. The reported 99.05% is a conditional numerical yield among source-timing-supported windows, not a scientific coverage guarantee.

## What the report does and does not establish

The endpoint-centroid oracle uses target information. Its IoU improvement motivates studying causal localization, but does not prove that the remaining error is only a few pixels or that source-centroid velocity equals physical velocity. Existing causal transport and IMM experiments already test part of that hypothesis. Their results do not justify another automatic search.

Koopman, recurrent/MLP, actual stationary/CV IMM and compact M32-input ablations have already been evaluated. They did not pass all forecast/acceptance gates against strong simple baselines. The older polynomial switching bank is not the actual IMM. A separate innovation event detector already exists; it is not a certified physical-contact detector.

The covariance Jacobian is useful engineering evidence. It does not include all uncertainty sources or establish an independently calibrated joint geometry/motion/identity distribution. The existing covariance report explicitly marks missing phase/identity/topology uncertainty and unavailable post-repair geometry covariance.

P08/P18 are excluded from training but repeatedly exposed in development evaluation. They cannot be renamed a fresh final test. Overlapping sequence counts are not independent episodes or participants. Under the previous participant-level split-conformal rank formula, 95% requires at least 19 valid independent exchangeable calibration units for a finite rank; five units yield an unbounded threshold. Simply adding 5-10 subjects to the current five does not meet that numerical minimum, and training subjects are not automatically valid calibration subjects.

## Remaining dependencies

The current audit still finds zero usable independently adjudicated references. Existing machine reviews are weak supervision, not independent physical ground truth. Real motion/contact/identity acceptance therefore remains unmeasured. The existing review contracts and tools are retained; no reference labels were invented and no reviewer was contacted.

Before final acceptance, establish genuinely fresh evaluation participants and a valid calibration design, complete blinded independent labeling with disagreement recorded, and preregister coherent gates before exposing the new labels. The report proposes several different event/contact thresholds; none was silently substituted for existing gates or applied retroactively to declare success. Resource estimates and calendar dates in the report are proposals, not scheduled jobs.

Reproduce the causality audit from the project root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m track1_replay --output reports/track1_followup/online_replay.json
```

Evidence: [online replay](online_replay.json), [report reconciliation](report_reconciliation.json), [baseline preservation](preservation.json), [independent-reference audit](independent_references.json), [verification](verification.json). This report preserves earlier frozen artifacts rather than rewriting their historical status fields.
