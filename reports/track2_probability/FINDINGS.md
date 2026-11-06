# Findings and reproduction

The shared-feature synthetic stationary IID control has squared-increment correlation 0.251 and approximately 1.2% Brier gain with a shared anchor; an independent anchor reduces both to approximately zero. The context-dependent-variance positive controls retain approximately 13–17% Brier gains under independent anchors. These are regular-time synthetic controls, not an audit of the complete irregular mask-to-kinematics pipeline.

On exposed validation participants, hand position+motion+metadata Brier is approximately 0.203 / 0.217 / 0.244 at 250/500/1000ms. The position-only arm is approximately 0.215 / 0.221 / 0.249. Local motion provides useful source-level discrimination. Functional position coefficients and projected memory provide no consistent incremental score benefit. These models do not consume full M16 shape coefficients. Stronger capacity or physical-motion claims are unsupported.

Training-fold scale recalibration raises the hand combined-input nominal 95% coverage to approximately 88% / 82% / 76%. The scale model remains undercovered. The independent physical-reference count is zero. No stochastic model, event hazard, model promotion or new deterministic forecast was added.

Every horizon uses its own supported training rows for RMS feature normalization, threshold and fit. This differs slightly from v7, which normalized across all input-supported training rows. No validation-driven hyperparameter selection was performed. Fit residual is an observation-fit statistic, not a pure measurement-error label. Feature dimensions differ across arms, so gains are not causal attribution.

All 166 repository tests passed. Final source/input, output-manifest and 431 preserved-file hashes were checked. Memory figures in status.json are sampled at fit boundaries, not a continuous whole-run peak.

Reproduce from the project root:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track2_probability.run
```

Raw data is reused in place. Lightweight weights, prediction ledgers, per-participant scores and calibration provenance are stored separately. The run refuses changed protocol/input dependencies. Full irregular preprocessing controls, matched background-motion controls and independent physical reference evaluation remain outstanding.
