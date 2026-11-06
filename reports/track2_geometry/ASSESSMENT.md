# Report reconciliation and decision

The supplied November 6 report largely describes work already completed. This follow-up implemented its remaining compact optional-M32-input ablation, rather than repeating the three existing architecture searches.

- Completed: four matched 5,196-parameter MLP arms, three seeds, 60 training-participant stopping fits and 12 final refits; certificate-bound endpoint descriptor extraction, missingness control, validation availability strata, baseline equality checks, inference profiling and deterministic replay.
- Matched controls: kinematics only; add a 16-coordinate training-only PCA of functional history; add geometry-availability flags; add eight geometry descriptors. The same network initialization and minibatch policy are used. Epoch selection follows the same training-only policy but may choose different epochs.
- Result: geometry improves over availability flags by 1.06% at 250 ms and 0.77% at 500 ms, below the predeclared 5% increment. It is worse than kinematics-only and nearest-observation continuation at all three horizons. All overall forecast gates fail.
- Coverage: 1,047 training and 224 validation histories retained. Only 60 validation histories have certified M32 at all four input timestamps; 72 have partial availability and 92 have none. Eligibility and conditional results are separate.
- Validation: 135 tests passed; 10,080 finite forecast records; all 2,016 baseline records exactly match prior artifacts. The geometry/seed-17 refit matches its checkpoint bitwise. Peak measured process-tree RSS was 7.06 GiB; minimum available memory was 28.26 GiB. No OOM occurred.

The four stored timestamps span 0.75 seconds at 0.25-second hops. Each functional state summarizes one past second; this is not four independent frames or a four-second clip. The existing explicitly approved six-second contexts and participant split remain unchanged. Targets at 250/500/1,000 ms number 203/184/146, with missing targets recorded rather than filled in as truth.

This experiment tests compact endpoint shape summaries, not every possible use of M32, raw vertices, contact constraints or a full shape latent model. Lower training losses do not establish generalization. This result does not prove that geometry is universally unhelpful; it shows that this bounded implementation has not justified extra inference features on this cohort.

Corrections to the supplied report:

1. Koopman, GRU/MLP and actual stationary/CV IMM prototypes have already been evaluated. Earlier polynomial switching hypotheses are a different model. Earlier CV event recall was not zero.
2. P08/P18 have been evaluated repeatedly. They cannot become a fresh final test by renaming a split. Overlapping windows and random seeds are not independent participants; no p-value claim is made.
3. A historical curve certificate is conditional on its decoder and inputs. It does not certify future physical geometry, contact, identity or general production readiness.
4. Interpolation using future knots is retrospective reconstruction, not a causal forecast baseline. Nearest-observation and constant-position baselines remain explicit and stronger than CV here.
5. Qwen outputs are machine annotations, not independent adjudicated physical ground truth. The current reference audit found zero usable independent references. Contact-boundary accuracy, physical event probabilities and identity acceptance cannot be manufactured from source contours.
6. Decoder-only timing and model-only timing exclude upstream preparation. New inference timing uses cached PCA/descriptors; it is not full RGB-to-prediction latency.

The report's proposals for independent annotations and a genuinely fresh final evaluation remain necessary external evidence. Neither this report nor the experiments justify a scientific freeze, promotion, full-scale training or Track 3 transition. Previously completed artifacts remain unchanged.
