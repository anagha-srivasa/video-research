# Guided review: start here

For contact cases, follow the five steps across the top:

1. **Watch** the original video. Choose slow playback if needed. This is context only; you cannot draw while it plays.
2. **Find area** opens one fixed frame, selected in advance for this case. Click near the hand/object interaction to zoom, or use the image centre. This click is not a boundary annotation.
3. **Hand edge**: draw a short line on the visible skin border nearest the object. Drag to draw, or click successive points. You do not need to outline the whole hand or arm.
4. **Object edge**: draw a short line on the visible object border next to that skin border. For the sponge example, use the yellow sponge border next to the fingers, only where you can actually see it. If the fingers cover it, choose **I can’t see this edge**. Never draw an imagined hidden border. The diagram is a generic explanation, not a prediction for your frame.
5. **Finish**: describe whether you see a gap, meeting edges, overlap, or cannot tell. Physical touching is a separate question and can remain unknown. Add your name and a brief note, then save a draft. Submission is a separate action that locks the review.

Your marks remain in original-image coordinates when zooming. Undo, Redo and Clear act on the current drawing tool. Drafts autosave after edits; wait for **Draft saved** before closing. You can also use **Save draft** explicitly. Watching or moving between steps does not submit anything. Earlier draft marks remain available under the final step.

If upgrading from the previous interface: save your old page first, then refresh. Unsaved marks exist only in that browser tab.

The technical details below describe reference requirements, not a checklist you must complete before drawing your first line.


You are reviewer A. There are 163 contact cases and 20 motion/identity episodes. Start with **five contact cases and two motion cases**, then check whether the instructions and workload are practical. Do not rush through all 183 cases before this pilot.

Open **http://127.0.0.1:8765**. Every case has its own guide beside the raw frame. The [case-by-case catalogue](CASE_GUIDE.html) links directly to each case. No candidate curves, predicted states, scores or source masks appear in the initial review. Your own marks are displayed.

## Before marking a case

1. Enter a stable reviewer identity (your name or consistent reviewer ID).
2. Record whether you previously saw candidate predictions for that case. The app hides predictions now, but cannot erase earlier familiarity. Do not open model dashboards while labeling.
3. Read the entity cue and confirm it visually. The cue comes from the source roster and can be wrong. Use **Entities you actually reviewed** to name the actual target or explain ambiguity in notes.
4. Watch or step through the full six-second context first. Playback is deliberately slow; timeline stepping is often faster. Use the query-frame button to return to the frame being annotated.
5. Zoom to 100% native or 200% for small gaps. Display zoom does not change native coordinates. Frame time is relative to the clip; annotation uses the exact native timestamp internally.

## Contact case: the query frame

The task is to establish what a human can actually see at the interaction boundary—not to agree with a source contact label.

1. Select **Hand boundary** and click vertices along the visible interaction-side outline. Trace the skin/glove edge, not a shadow. Select **Object boundary** and trace the relevant object edge facing the hand.
2. Choose **Contact region** for short open traces. Choose **Full visible outline** only if the entire visible outline is deliberately traced as a closed shape. Do not use a short trace as a full mask.
3. For a thin knife/utensil, zoom in and distinguish the blade edge from fingers, shadows and motion blur. For transparent/curved objects, distinguish the physical edge from reflections. If the edge is not defensible, mark uncertainty rather than tracing an imagined boundary.
4. Record **Projected image-plane relation** separately from **Physical contact judgment**. A visible 2D overlap does not prove physical touching; physical contact may remain unknown.
5. Inspect neighbouring frames to bracket first contact and release. Select the event, mark the earliest plausible frame, then the latest plausible frame. Keep a nonzero interval when timing is ambiguous.
6. If contact is already active at clip start, record a left-censored onset in notes; do not invent an onset at frame zero. If release is outside the clip, record right censoring. Occluded contact is unknown unless independent visible evidence justifies a judgment.
7. Record visual difficulty flags such as `occlusion`, `blur`, `small_object` or `ambiguous_identity`. These are your independent observations, not automatic model strata.
8. Add a brief explanation of the evidence and uncertainty. Choose usable, uncertain or unscorable. Usable visible boundaries can coexist with unknown physical contact; later scoring keeps those observables separate.

## Motion/identity episode

1. Define a fixed physical landmark before marking points: for a hand, a recognisable wrist crease or knuckle; for a rigid object, a corner or handle junction. Do not use a silhouette centroid that changes with occlusion/deformation.
2. Use **Fixed landmark** to mark the same physical point over native frames. Aim for roughly 25 Hz (40 ms) on visible spans, with denser marks around rapid starts/reversals. Seven points are only the submission minimum, not proof that the episode supports velocity evaluation.
3. Mark visibility per frame. For occluded, out-of-frame or blurred frames, leave the landmark absent. The app never auto-fills human points through gaps. Evaluators retain sparse-reference failures in their coverage ledger.
4. Bracket `motion_onset`, `motion_stop`, `occlusion_start` and `occlusion_end`. Motion onset and contact onset are different events. Use `quiet_interval` for visibly quiet **image-plane landmark motion**; a physically stationary object can still move in the image because of camera motion.
5. For reappearance, judge same entity, different entity or unknown from visible evidence. Explain which cue supports the judgment. Do not assume the source chain is correct.
6. Keep onset/release intervals broad enough to reflect uncertainty. Do not type velocity: it is derived later from the reviewed positions with two declared smoothing methods.

## Controls and saving

- Click to add points, or type native X/Y and use **Add native point**. Keyboard arrow keys on the canvas step frames. Previous/next and ±5-frame buttons are also available.
- Undo removes the last point for the selected tool. Clear removes only that tool's marks in the current frame.
- **Save draft** preserves your work without making it a completed reference. Changing cases saves a dirty draft; closing the page warns about unsaved changes.
- **Submit this review** locks your A review after confirmation. Do not submit while experimenting with the drawing controls. A submitted review cannot silently overwrite its earlier evidence; corrections belong in a documented revision/adjudication round.

## What happens after your review

A second independent reviewer uses role B and cannot retrieve your labels through the app. A distinct adjudicator sees the two human reviews, raw frames and disagreements, but no candidate predictions. The A and B files are retained unchanged and their hashes are bound into adjudication. One person filling A and B does not establish independent agreement.

Original source annotation diagnosis is a **separate later phase**. The source outline becomes available only after adjudication is locked. Fragmentation, missing components, identity errors and source occlusion ambiguity are recorded separately; they cannot rewrite the reference established from raw video.

These are exposed development participants, not a fresh final evaluation set. Your review is a first reference pass; it does not by itself authorize baseline promotion, a scientific freeze or Track 2 training.
