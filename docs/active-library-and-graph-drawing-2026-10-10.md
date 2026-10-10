# Active library and graph drawing

Owner direction, 10 October 2026. This scope supersedes family expansion lists in older project documents. Earlier audits, experiments and release receipts remain historical evidence; their old inventories are not the current library.

## Active library

The read-only top-level folder check matches the owner's supplied screenshot:

| Folder under `E:\models\loras` | Current capability and development scope |
| --- | --- |
| FLUX | FLUX.1 is the first end-to-end block proposal/export target. Compatibility remains per adapter and loader. |
| Flux.2-Klein | Active expansion target; catalogue recognition does not grant verified block export. |
| LTXV2 | Active expansion target; establish exact adapter/version/modality evidence before analysis/export. |
| LTXV2_5 | Active expansion target; separate architecture and loader validation required. |
| MiniMax-H3 | Active expansion target; separate architecture and loader validation required. |
| WAN2.1 | Active expansion target; mode-aware recognition is not block export support. |
| WAN2.2 | Active expansion target; mode-aware recognition is not block export support. |
| Z-Image | Active expansion target; separate architecture and loader validation required. |

`LoRA_Manager_Images` and `recipes` are auxiliary directories, not model families. Inventory excludes them as before.

**Pony, SDXL and Illustrious are retired from the owner's library and future development scope.** They are removed from active UI family choices, including the fallback choices. SD 1.x and the separate Flux Krea folder are also outside this active folder list; this does not reject a valid FLUX.1 adapter merely because it was trained against a particular FLUX checkpoint.

Harmless legacy identity mappings and historical tests remain so old catalogue rows, missing-file records and saved recipes retain their meaning. They enable no analysis or export for the retired families. No new scanner, balancing policy, loader integration or UI feature should be developed for them. The model-family API labels active and retired scope explicitly. Backend recognition remains broader than current owner scope for historical compatibility.

The scanner discovers files under the configured root; it never requires every registered family folder to exist. Removing whole family folders marks old records as missing without deleting IDs, profiles or recipes. A regression verifies a FLUX-only library with missing Pony/SDXL/Illustrious history. Scanning, development and validation never recreate, move or delete owner model files. Historical library counts must not be presented as current counts.

## Draw block values

In either graph, press the primary mouse button on a bar, hold it and move horizontally/vertically to draw the shape. Pointer position chooses the current bar even while the first bar holds pointer capture. Fast strokes interpolate through crossed bars in both directions. Release, cancellation or lost pointer capture ends the stroke. Another pointer and non-primary buttons cannot take over an active stroke. Moving outside the graph horizontally pauses drawing; the next position resumes without painting across the outside excursion.

Each block retains its sign. Exact numeric fields can change a sign. Ordinary graphs draw multipliers; measured graphs draw contribution magnitudes and calculate the corresponding multiplier using that block's measured norm and model strength. Zero-update blocks and zero model strength cannot manufacture a measured contribution. The scale and signs are fixed for a stroke, and the original line stays unchanged. The graph viewport can be scrolled before drawing; drawing does not auto-scroll it.

Edits are personal drafts. Multiple crossed-bar updates in one event accumulate into the same draft; they do not overwrite earlier crossed blocks or unedited peer proposals. Immutable Defaults, reset controls, keyboard editing, saved revisions, explicit recipe loading and fresh preparation before copying remain intact. Drawing changes the chosen trial values, not the role-aware proposal algorithm.

The pressed graph remains anchored in its vertical scrolling container while a stroke is active. Invalidating the prepared export can remove a summary above it; this must not move the graph underneath the held pointer and distort the drawn values.

## Acceptance and next work

The owner reports that the preceding package tested fine. That confirms their reported application acceptance; no new controlled render evidence was supplied and it is not a claim that the earlier Sabrina/tights conflict is repaired. Validate this package locally, in the browser with synthetic data, and on GitHub/Farnsworth against the exact source. Promote through the official launcher only after required checks pass and a verified preview-data backup exists. Continue within the active families above; FLUX.1 experiment usefulness remains a controlled-render question.

Local validation: 523 backend tests passed with the real CPU tensor environment; 146 UI tests, all 23 tooling tests, lint, production build and whitespace checks passed. Regressions cover crossed-bar interpolation, opposite signs, measured norm conversion, cancellation/lost capture, saving every crossed value, peer-proposal preservation, active model choices and absent retired folders. A clean-session synthetic browser review used actual held-pointer drags in both directions, retained the original line and stable graph position, and reported no console errors. Remote exact-source checks and final runtime promotion are recorded in the local release receipt.
