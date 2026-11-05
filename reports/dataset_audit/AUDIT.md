# Dataset and preprocessing audit — 29 October 2026

The preprocessing concern is justified. This audit found a reproducible contour-schema defect, an overly coarse quarantine policy, and substantial limitations in the evidence available for motion, identity and contact. These coexist with representation limits: fixing data will not make a four-coefficient function represent every abrupt event. A groundbreaking result cannot be promised, but a stronger dataset contract and independent evaluation are the highest-value next investments.

This is an audit, not a rebuild or scientific freeze. Frozen sources, canonical/v1, v1/v2 models, lab coefficients and historical dashboards were not changed. Track 2 remains on hold. Current dashboard claims must be interpreted together with the corrections below.

## Audit scope and evidence

Inspected acquisition documentation, PTS/annotation alignment, raster geometry, association, camera conversion, appearance, compound geometry, holdouts, uncertainty and reference ingestion code. Recomputed aggregate inventory across all eleven videos. Detailed contour-role inspection and partial-geometry checks use only the seven development participants; no test model scores were computed or used to select changes. Queries use a 2 GB DuckDB memory bound and four threads; mask parsing streams batches. This is not a new full raw-video decode, all-file checksum audit, frame-by-frame visual review or independent annotation exercise.

Reproducible evidence: `measurements.json`, `additional_measurements.json`, `reproductions.json`. Scripts: `../../work/dataset_audit/audit.py` and `reproduce.py`. The first evidence file records inspected source hashes. Existing test success is not proof of a correct inter-module data contract: synthetic tests explicitly supply `hole`, bypassing the actual canonical `inner` label.

| Measured inventory | Result | Meaning |
|---|---:|---|
| Native frames | 390,570, all 1920×1080 | Native PTS indexing exists |
| Source mask observations | 699,496 | Not independent labels or identities |
| Manual masks | 18,742 (2.68%) | Sparse human source supervision |
| Dense interpolations | 658,522 (94.14%) | Automatic, temporally dependent evidence |
| Dense source endpoints | 22,232 (3.18%) | Not 22,232 new independent annotations |
| Untimed mask observations | 23,835 (3.41%) | Separate from 8,857 untimed annotation-frame records documented in the canonical README |
| Quarantined masks | 54,859 (7.84%) | Overlaps with untimed masks; do not add percentages |
| Multi-component masks | 139,440 (19.93%) | Dominant-component models discard information |
| Masks with holes | 1,013 | Canonical inventory, before downstream eligibility |
| Seven-video selected spatial observations | 413,830 | 398,749 interpolated, 13,658 manual, 1,423 endpoints |
| Seven-video canonical masks with holes | 805 | Not necessarily all selected downstream |
| Selected spatial rows reporting holes | 0 | Confirmed schema defect; not evidence that no holes exist |
| Compound store ring roles | 1,023 outer, zero other roles | Current real-data certificate cohort has no hole roles |

## Ranked findings and remedies

### P0 — One ring schema across every consumer

**Confirmed bug.** `src/pipeline/geometry.py` emits `role='inner'`, `depth`, and `parent_ring`. `src/track1/spatial.py:dominant_ring` counts only `hole`. `src/track1/appearance.py` ignores `inner` during hole exclusion. The same mismatch appears in lab `appearance.py:roi_masks`, `architecture.py:source_geometry` and render loops, and `compound.py` source-topology counts/rendering. No normalization adapter was found. A tiny polygon-union fixture produces one canonical hole and zero downstream counted holes.

Impact: wrong metadata; absent hole masking; contaminated foreground targets for some crop diagnostics; inappropriate source topology signatures; possible loss/rejection/misinterpretation of inner rings before certification. Current multi-ring IoU was measured against a canonical target, so the reported arithmetic is not automatically fabricated, but the claimed hole-aware implementation and its subgroup interpretation are not supported. Certificate validity for encoded outer-only curves cannot certify completeness against source masks. Do not extrapolate this local bug to every result or claim an unknown corrected score.

Remedy: a versioned canonical-to-model geometry adapter, strict enum validation, preservation/remapping of parent indices and depth, and a hierarchy-ordered renderer. Simply replacing strings in one file is insufficient. A two-pass outer-then-hole painter also destroys nested islands; process the containment tree by depth. Test an actual canonical row through geometry, fitting, appearance, serialization, decoder and metric computation, including hole/island/disconnected cases and tiny rings. Unknown labels must fail loudly.

Rebuild affected development spatial metadata, hole-aware appearance, PCA if feature inputs change, compound fitting, topology epochs, certificates, calibration and reports under a new version. Keep old frozen evidence immutable and explicitly mark superseded claims. Bind schema/adapter/renderer versions and upstream data hashes to caches; model-code hashes alone are insufficient. Measure per-hole and multi-component quality before discussing average gains.

### P0 — Separate offline reconstruction from online information availability

**Confirmed source dependency; not an accusation of participant-split leakage.** Official VISOR dense masks are generated from two sparse endpoints. A retained interpolated mask before a cutoff can already depend on a future endpoint. Holding out an interior target while retaining neighboring masks generated by the same endpoint pair also leaves strongly dependent evidence. Current hidden-target perturbation tests verify the fitter's exclusion logic, not the upstream mask-generation dependency graph.

Introduce `observation_time_us`, `available_at_us`, source endpoint IDs/times, teacher/version, and dependency IDs. Availability must be derived from the actual generator and its RGB access, not assumed equal to the right annotation timestamp. Preserve two explicit products: retrospective contour encoding and online RGB-derived measurements. For genuine forecasting, replay forward-only tracking/segmentation from permitted initial observations; no future-assisted mask propagation, bidirectional smoothing, future appearance aggregation or post-cutoff reassociation. If generator dependencies are unknown, label availability unknown and exclude from strict online acceptance.

Evaluate reconstruction with source masks, causal measurement estimation with independent anchors, and forecasting using inputs available at the cutoff. Group pseudo-label holdouts by complete interpolation chain/dependency closure, not only target timestamp. Equalize or cap weight per chain/manual anchor interval, and report effective independent units rather than dense frame count. Block-bootstrap by participant/video/episode; overlapping windows are not independent trials.

### P0 — Fix the target definition before optimizing velocity

**Confirmed semantic limitation.** Position currently derives from visible-mask centroid and area. It moves when an object deforms, rotates, becomes occluded, changes component visibility or touches an image edge, even if a material point stays stationary. Shape seam alignment is geometric correspondence, not material point tracking. Camera-normalized 2D coordinates still include camera motion and perspective.

Keep visible centroid as a useful separate feature. Add persistent visible anchors, rigid-object pose where defensible, hand keypoints where available, visibility/occlusion/truncation flags, and confidence in anchor identity. Estimate object motion from stable anchors with a documented coordinate frame and camera motion context. Do not relabel these as centre of mass or metric 3D. Use independent dense anchors to establish whether CV/Kalman differences reflect motion, segmentation flicker or changing support.

### P0 — Independent references and an untouched evaluation population

All 210 current independent-review proposals are pending. None of the accepted RGB proxy validation episodes establishes fast-hand reference quality. Existing development validation is repeatedly exposed; old test videos cannot be made fresh by assigning new names.

Create a review pack initially covering 200–400 short episodes from new participants, with stationary motion, steady translation, onset, reversal, brief events, occlusion, small objects, contact and topology changes. This is a proposed starting budget, not a power-calculated sample size. Pilot double annotation, estimate disagreement, then size participant-level evaluation for the smallest meaningful improvement. Reserve fresh participants and manifests before parameter work; increase diversity rather than collecting many correlated frames from one kitchen. Split reviewer calibration/development and locked evaluation. Include multiple same-noun objects and difficult negatives. Reviewers should not see candidate predictions when labeling references.

For each episode collect timestamped native-grid anchors, masks/boundaries, visible/occluded/truncated state, persistent instance and lineage, contact pair/state/onset/release/ambiguity, two annotator IDs and disagreement. Derivatives come from reviewed trajectories with smoothing sensitivity and timing uncertainty. Contact class annotations alone do not supply contact geometry or exact event timing.

### P1 — Recover partial geometry without silently inventing complete masks

`pipeline/geometry.py:_raster` rejects the entire observation as soon as any segment is degenerate. Of 54,859 quarantined observations, 52,478 carry `degenerate_polygon`; flags overlap. `reproductions.json` counts development masks with a degenerate fragment and another individually rasterizable segment. The streamed check found **33,243 of 34,774 quarantined development masks** with this combination. These are candidates for partial evidence, not automatically recoverable full masks.

Retain segment-level validity, rejection reason, raw coordinates, and completeness. Separate empty, invisible, malformed, outside-frame, tiny and partial cases. Preserve usable components for appropriate partial losses with completeness masks; require review before they become full-mask truth. Do not repair by unrestricted convex hull, star-convex filling or area interpolation. Measure repair distortion and topology changes on clean and difficult examples. Retain pixel-exact raster evidence and subpixel vector geometry as separate views; round-trip errors must be measured, not hidden by resampling.

### P1 — Timing and resolution need image-based validation

Good existing foundations: full native PTS, pinned official correction, explicit inferred dense timing, raw sources preserved, unresolved timestamps not guessed by FPS. No new evidence of widespread clock corruption was found.

Dense masks use 854×480 whereas manual masks use 1920×1080. Upscaling cannot restore missing detail. One dense pixel spans roughly 2.25 native pixels; strict 2-pixel native boundary thresholds need provenance-aware interpretation. Pixel-centre conventions, clipping, resize versus crop, undistortion and RGB/mask transforms require one explicit chain.

Validate sparse and dense RGB correspondence at start/middle/end and high-motion frames of each chain, against official extraction where available. Compare offsets −2…+2 frames using reviewed boundary/foreground alignment, and record uncertainty rather than picking the model-friendly offset. Audit duplicate PTS, dropped frames and endpoint consistency. Improve unresolved records only using additional verified correspondences, never interpolation over an inconsistent clock. Calibrate timing uncertainty into motion error budgets: timing jitter times speed can dominate a boundary error. Per-frame RGB checks and per-chain timestamp shifts were not performed in this audit.

### P1 — Identity and topology are different problems

Current association groups unique names within source chains and bridges unique shared endpoints using bounding-box IoU. Duplicate same-name observations cause conservative fragmentation. This is reasonable provisional evidence, not independently verified persistent identity. Canonical source IDs are observation-local in this pilot.

Store persistent object identity separately from visible-component identity and annotation-chain ID. Record uncertain links, alternatives, birth/death visibility, and split/merge lineage. Disappearing components need not imply object destruction; cutting can change physical identity. Match with appearance, motion, mask overlap and explicit constraints, then review difficult cases. Use bidirectional association only in the offline product; maintain a separate online replay. Report identity switches, fragmentation, label coverage and contact-episode consequences on independently reviewed examples.

Continuous topology certificates should certify the represented curves; source fidelity/completeness and correspondence require separate gates. Reset or segment at real/ambiguous topology transitions rather than smoothing through them. Keep all failed attempts in denominators. Fix the ring contract before spending more compute on certificate subdivision.

### P1 — Appearance and uncertainty must consume correct evidence

The frozen full-frame backbone produces an 8×14 map; fine objects can disappear at cell centres, and large receptive fields admit background. ROIAlign on an already coarse map cannot recover spatial detail. Existing crop/retrieval diagnostics are small, source-chain-based and affected by the hole issue where applicable.

After fixing masks, compare native-resolution object/context crops, aspect-ratio-preserving resize, area-integrated pooling and multiscale features. Keep object and context embeddings distinct. Compare frozen backbones before committing to expensive fine-tuning, using independent instance retrieval, foreground interventions, small-object performance and downstream sample efficiency. Fit PCA/normalizers only on training inputs, with per-instance/chain balancing. Ground-truth-mask-guided embeddings are privileged inputs; future RGB-only tracks need predicted-mask counterparts and a measured domain gap.

Fit residual covariance is not annotation reliability. Add empirically measured mask/timing/anchor error by provenance, scale, occlusion and motion; propagate selected correlations and association ambiguity. Do not count interpolated masks as independent Gaussian measurements. Calibrate on independent training references and assess joint coverage plus proper scoring rules on disjoint references, with subgroup and participant uncertainty. Keep unknown terms explicitly unknown; do not sum overlapping covariance allowances as if independent.

### P1 — Camera, contact, actions and missingness for later tracks

Canonical camera rows contain 350,668 aligned poses, but every canonical pixel transform remains marked invalid. Track 1 has separately reviewed camera artifacts: consumers must state which validation they use, rather than silently interpreting pose-valid as pixel-transform-valid. Static reconstruction and photometric projection checks do not establish metric object depth or moving-object pose. Add reprojection-based calibration over time and static-background motion residuals; gate camera compensation where correspondence or pose support fails.

All 6,084 contact records are sparse manual relations: 4,859 resolved targets, 381 no-contact, 346 unsegmented targets, 465 inconclusive and 33 missing target IDs. Preserve these distinctions. Unsegmented/inconclusive is not negative. Use interval-censored contact changes until dense review supplies timing. Define signed 2D separation carefully and do not call overlap physical penetration. Action intervals may overlap; maintain independent action/VISOR split policies, narration availability and event uncertainty. Missing masks are not absence; absent camera poses are not zero motion; invalid features are not zero-valued valid observations.

## Capacity plan: use 300 GB deliberately

Observed locally: 48 GiB physical RAM, about 22 GiB available at inspection, and 237 GiB disk free. Existing project allocation from `du`: dataset 23 GiB, canonical 1.3 GiB, derived 8.9 GiB, work 39 GiB, reports 0.12 GiB (rounded allocated sizes). The 300 GB statement could mean storage or actual provisioned RAM. This plan treats it as a total dataset storage envelope until higher RAM is present; a future 300 GB RAM machine can increase concurrency but should still stream data. Do not raise this workstation's RAM limits to 300 GB.

All native RGB frames as uint8 would require **2.43 TB decimal** before indexes, masks and features. Therefore full raw frame extraction is not suitable for a 300 GB envelope. Keep compressed source videos and native PTS, bounded caches and selected lossless reference crops. Estimate measured bytes per episode/video before expanding; avoid extrapolating only from duration because bitrate and label density vary.

Proposed total allocation, decimal GB (caps, not measured requirements):

| Purpose | GB |
|---|---:|
| Immutable compressed sources, expanded participant coverage | 100 |
| Canonical tables, geometry, provenance and indexes | 20 |
| Reviewed reference clips/crops/labels and review versions | 35 |
| Bounded multiscale crops/features and causal pseudo-label shards | 70 |
| Rebuild workspace / checkpoints | 35 |
| Free reserve | 40 |
| Total | 300 |

Existing artifacts count against these caps. Review the 39 GiB work area by manifest before any cleanup; do not delete caches, partials or source archives in this audit. Use atomic per-video/episode shards, content hashes, resumable stages and bounded prefetch. CPU workers decode/validate/associate; GPU batches run feature extraction and segmentation. Tune concurrency from measured throughput and memory, with separate CPU/GPU limits; using all cores on an I/O-bound stage can reduce throughput. Archive reusable pooled features, not every transient full feature map. Retain enough clean rebuild space and checksum gates.

## Execution sequence and acceptance evidence

1. **Repair contracts first.** New geometry adapter/schema and end-to-end real-data fixtures; audit cache dependency keys and affected descendants. Acceptance: hole/island/component preservation, explicit unknown-role rejection, exact provenance and no silent label conversion.
2. **Matched development rebuild.** Same participants/probes, new version, same per-protocol denominators; compare original and repaired pipeline separately from architecture changes. Include contour-first versus raster-first, partial-evidence policies, and corrected appearance. Acceptance: subgroup errors, failures and resource costs reported, no reference leakage. Do not declare gains until measured.
3. **Construct independent references and dependency-aware splits.** Pilot annotation agreement and statistical power; acquire fresh participants within measured capacity. Lock the evaluation assignment before selection. Acceptance: review provenance, disagreement, dependency closure and protected evaluation labels.
4. **Build dual offline/online observation products.** Available-at timestamps, source generators and causal replay; anchor/identity/contact evidence and optional camera context. Acceptance: perturbing any post-cutoff RGB/annotation cannot change online state; offline smoothing stays explicitly separate.
5. **Ablate causes, not only models.** Hold the model fixed while changing timing, geometry, observation provenance, anchor target, identity and appearance one at a time. Then test combinations; report paired participant results and failure coverage. Train-only choose settings, use fresh lock once. This distinguishes actual data improvements from validation adaptation.
6. **Demonstrate benefit beyond Track 1 when authorized.** Export task-neutral observations, identities/lineage, timestamps/availability, transforms, contact/action intervals, quality and dependencies. Later evaluate RGB-only tracking, event/contact anticipation and forecasting using identical information budgets. Do not build a future task on target-derived contour or future-assisted pseudo-label inputs while claiming online prediction.

The most defensible research opportunity is a provenance- and availability-aware entity representation that preserves topology and uncertainty, separates visible shape from motion, and retains short events under a compact budget. Its contribution must be established by independent measurements and downstream utility, not by matching a segmentation benchmark with access to the target contour.

## Primary external references checked

- Official VISOR schema and endpoint-conditioned interpolations: https://github.com/epic-kitchens/VISOR-VIS
- Official mandatory dense correction/extraction: https://github.com/epic-kitchens/VISOR-FrameExtraction
- VISOR task definitions: https://epic-kitchens.github.io/VISOR/site
- EPIC Fields camera data and benchmark scope: https://epic-kitchens.github.io/epic-fields/site

These establish source semantics; all local counts above come from this audit's artifacts. Dataset expansion, independent annotation, corrected model scores and clean rebuilds are proposed work, not completed results.
