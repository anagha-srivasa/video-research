# Guided review interface

Contact review now follows Watch → Find area → Hand edge → Object edge → Finish. Original-video playback is separate from the exact fixed annotation frame. Short visible facing edges are explained with generic diagrams and case-specific wording; hidden edges are recorded as unavailable rather than inferred. Draft autosave, explicit lock confirmation, undo/redo, zoom and earlier-draft inspection are available. Adjudication retains its separate interface.

Verification: 80 tests passed; JavaScript syntax checks passed. An isolated Chrome fixture exercised original-video playback, fixed-frame transition, zoom, hand drawing, undo/redo, hidden-object uncertainty and draft persistence. No real review was submitted. The existing reviewer A saved draft matched its pre-edit backup byte for byte. At 390px viewport width, page width was 390px and footer controls remained within the viewport. Browser console reported no errors. QA marks exist only under work/track1_reference/redesign/fixture.

Original videos stream directly with byte-range responses; no additional RGB product or transcoded video is stored. Annotation still uses lossless exact native-frame responses. Candidate predictions remain hidden. This update does not certify independent references or change Track 1 scientific readiness.

Before refreshing an already open old interface, save its draft to retain any marks still held only in that tab.
