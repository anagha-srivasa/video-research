# Bounded Track 1 remediation candidates

This development programme implements the four candidates in the latest assessment. Track 2 remains on hold. No candidate is promoted and no scientific freeze is authorized.

`src/track1_remediation` contains correspondence/epoch fitting, causal motion/background alignment, the tiered reader and evaluation. Its tensors and indices are in `derived/track1_remediation`, evidence in `reports/track1_remediation`, and scratch work in `work/track1_remediation`. Existing raw RGB, model selections, PCA, default readers and baseline artifacts are preserved.

- A diagnoses every original M32 epoch-failure opportunity. Source graph changes are observed annotation events, not verified physical birth/death or occlusion. Persistent ring IDs are valid only within a stable source topology/assignment epoch. The optional geometry candidate uses the latest stable segment, with at least eight observations, 0.3 seconds of support and a decoder-bound continuous certificate. It never interpolates across an event. This is a different history-duration view, not recovery of the missing full one-second trajectory.
- B compares stationary, CV and a fixed causal three-hypothesis bank on matched inputs at 100/250/500 ms. Its weights and covariance are uncalibrated. A background partial-affine view excludes annotated foreground and requires spatially distributed RANSAC support. Camera failures reset its epoch. Future camera continuation uses only past matrices; this remains a 2D diagnostic on retrospective source masks.
- C uses the unchanged M16 core plus optional M32 geometry. Full one-second certified geometry is preferred; otherwise the latest certified stable segment is available with its own interval and epoch. Missing geometry is `None`. Sequence indices carry the availability mask and an explicit flag for every adjacent geometry-epoch pair. A continuous core sequence does not establish continuous compound geometry across these resets. Indices distinguish `all_core16_geometry_complete_certified` from `all_compound_geometry_available` and store per-state geometry durations.
- D compares geometry, geometry plus appearance, and geometry/appearance/motion-uncertainty association. Scores are agreement with provisional source chains, with a correctly initialized past memory, not independent tracking accuracy or IDF1. Categorical contacts select review proposals; they do not certify boundary distances or physical contact events. Two-reviewer worksheets remain pending.

```python
from track1_remediation.store import TieredDataset
import pyarrow.parquet as pq

sequence = pq.read_table(
    'derived/track1_remediation/context/indices/train.length16.parquet'
).to_pylist()[0]
with TieredDataset(source='context') as ds:
    end_us = sequence['end_times_us'][-1]
    state = ds.get_state(sequence['video_id'], sequence['entity_track_id'], end_us)
    if state['m32_available']:
        curves = ds.geometry(state, end_us)
        interval_seconds = state['quality']['geometry32_history_seconds']
```

The reader's `kinematics` retains the original local CV estimator. Switching/background candidates are stored for sampled diagnostic cutoffs; they are not a full-timeline replacement motion release. Geometry covariance and calibrated identity/topology probabilities remain unavailable.

Reproduce all four candidates and numerical verification from the existing dataset:

```sh
PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python3 -m track1_remediation.run --stage all
```

Stages are `correspondence`, `camera`, `motion` (including matched synthetic tests), `tiered`, `evaluation` and `verify`. A process lock prevents concurrent writers. Existing process-tree RSS, available-memory and disk guards remain active.

Read [the report](reports/track1_remediation/REMEDIATION.md), [dashboard](reports/track1_remediation/dashboard.html) and [status](derived/track1_remediation/status.json). Quality tables distinguish matched source reconstruction, source-centroid continuation, exact synthetic references and source-identity association proxies. None is an independent contact/identity/motion acceptance test. All overlapping-window counts must be assessed at participant/video/episode level.

The simultaneous-band check uses the finite-sample rank `ceil((n+1)*0.95)`. Five participant units require rank six, outside the calibration sample; even under exchangeability the threshold is infinite. Here a fresh exchangeable calibration split is also absent, so no formal 95% trajectory coverage is claimed. See [Ryan Tibshirani's conformal prediction notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf), page 2, for the rank and infinity convention.
