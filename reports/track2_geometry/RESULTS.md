# Matched functional-history and optional-geometry ablation

Bounded experiment complete. Forecast proxy gate passed by any arm: False. Scientific acceptance and full Track 2 GO remain false.

Four arms use the same 5,196-parameter MLP, input histories, target masks, seeds, initialization, minibatch policy and residual loss. Kinematics uses four relative position/velocity states. Functional adds 16 training-only PCA coordinates of M16 history. Availability adds four optional-M32 availability flags. Geometry adds eight endpoint descriptors per available state: log width, log height, log net area, two normalized second moments, hole area fraction, outer-ring count and hole count. Missing descriptors are zero after training normalization; their availability is explicit. No missing-geometry history is excluded.

This tests a compact geometry summary, not raw vertices, full learned shape latents, contact constraints or learned shape dynamics. Optional descriptors come from certificate-bound historical M32 endpoints only. Future geometry does not enter inputs. Model output remains source position and velocity, with no event/contact probabilities or calibrated covariance.

| Arm | 250 ms | 500 ms | 1 s |
|---|---:|---:|---:|
| kinematics | 0.10263 | 0.14770 | 0.20839 |
| functional | 0.10569 | 0.15809 | 0.24654 |
| availability | 0.10480 | 0.15597 | 0.24448 |
| geometry | 0.10368 | 0.15478 | 0.24552 |
| cv | 0.13484 | 0.25960 | 0.46500 |
| constant_position | 0.09912 | 0.14448 | 0.18252 |
| nearest_observation | 0.09471 | 0.14108 | 0.18135 |


Position RMSE uses intrinsic-normalized coordinates. Calculate RMSE within each participant, average participants, then take the median of three seeds. Detailed velocity and availability strata are in metrics.json. Strata may contain only one participant; participant counts are explicit. Histories overlap and do not constitute independent episodes.

Support: [{"all_four_geometry": 458, "any_geometry": 668, "histories": 1047, "no_geometry": 379, "split": "train"}, {"all_four_geometry": 60, "any_geometry": 132, "histories": 224, "no_geometry": 92, "split": "validation"}].

All normalization and PCA fitting use training participants only, including inside each of 60 participant-held-out stopping fits. Median selected epochs determine 12 all-training refits. Validation P08 and P18 have been exposed repeatedly and remain exploratory; this report does not call them fresh or claim p<0.05 from overlapping windows. Epoch choices can differ across feature arms because each uses the same training-only stopping policy. Three seeds are repeats, not new independent subjects.

The supplied report is stale on several points: Koopman, GRU/MLP and actual stationary/CV IMM prototypes already exist and were evaluated. The older degree-0/1/2 switching bank is not the actual IMM. CV's event recall is not zero in the earlier synthetic results. Existing sequence counts are overlapping indices, not thousands of independent episodes. Temporal interpolation using future observations is not a causal baseline. Track 1 is an engineering baseline, not independently certified production-ready physical understanding.

Independent motion/contact/identity references and genuinely fresh participants remain missing. Qwen review is machine supervision, not independent adjudicated physical truth. Proposed contact-boundary and event gates cannot be evaluated by inventing labels. No baseline promotion, full-scale training, final freeze or Track 3 transition is performed.

Original Koopman, neural and hybrid evidence is preserved. No prior validation inference was rerun; this is a new, prelocked descriptor ablation with identical current-arm input cohorts. The fixed geometry/seed-17 deterministic refit is checked bitwise. Dataset images and source tensors are reused without alteration.

Reproduce: `PYTHONPATH=src:work/deps:work/track1_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.13 -m track2_geometry.run --resume`. Scratch files are under work/track2_geometry. CPU float32 models use bounded workers and process-tree memory guards; upstream feature generation is excluded from model latency.
