# Track 1 dependence-aware state experiment

The development modules live in `src/track1_belief`; uncompressed state artifacts are in `derived/track1_belief`; reports are in `reports/track1_belief`; scratch files and resource logs are in `work/track1_belief`. Existing raw data and frozen releases are reused in place.

Run the complete bounded experiment:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -u -m track1_belief.run --stage all --resume
```

The process monitor requires host process visibility. The run enforces an 18 GiB process-tree RSS limit, a 6 GiB available-memory reserve, six numerical workers and two native RGB workers. GPU use is absent in the current classical tracker and small matrix estimators. No speed or accuracy claim about a neural tracker follows from these measurements.

Stages are anchors, synthetic, real, geometry, verify and report. Resume validates the source/config protocol and stage output hashes. Native tracking checkpoints separately bind their input and output hashes. Initial integration outputs are retained under `work/track1_belief` with reasons for the two adapter corrections.

The implemented components are:

- Native-resolution, forward/backward-checked persistent LK anchors with permanent point IDs, disjoint fit/proxy point sets and a confidence-gated background view. Initialization uses a privileged manual mask; tracking after initialization reads only past RGB.
- Continuous-time CV transitions, persistent correlated group biases, robust measurement updates, coherent process jumps and joint position/memory covariance.
- Duplicate-invariant, time-mass history reconstruction with full propagated covariance and exact endpoint position/velocity constraints when requested.
- Gauge/mixture covariance propagation and bounded all-ring static spline allocation using the unchanged canonical decoder/certificate machinery.
- Train-only task-weighted projection and compact linear auxiliary probes. The real-data probe stage explicitly refuses insufficient training support.

Synthetic training selects process/event parameters from 60 generated training episodes. Thirty separate generated episodes evaluate the selected candidate. These are controlled analytic diagnostics, not independent real-world acceptance. The selection objective is fixed in `evaluate.py` and recorded in `selection.json`.

The native cohort contains the 20 existing six-second contexts from five training and two previously exposed validation participants. Every context and query remains in the attempt ledger. The disjoint point proxy shares the RGB sequence, initialization and algorithm with the fitted points, so it is not an independent physical reference. Full and common-support scores are both retained.

Read `reports/track1_belief/ASSESSMENT.md` and `derived/track1_belief/status.json` before consuming the candidate. Numerical correctness does not imply useful dataset coverage, independent uncertainty calibration or model promotion.

Remaining proposed work is explicit: neural point-tracker adaptation and re-detection, multi-neural-teacher verification/distillation, local adaptive spline knots, temporal compound geometry and independent motion/contact/identity references. The static geometry allocation is not local knot refinement. The linear auxiliary probe is not neural teacher distillation. The complete proposed research program has not been finished by this bounded experiment.

Tests:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /opt/anaconda3/bin/python3.13 -m pytest tests -q --basetemp=work/track1_belief/pytest_full
```
