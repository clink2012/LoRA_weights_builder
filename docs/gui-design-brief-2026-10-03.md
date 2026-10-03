# GUI design brief — 3 October 2026

Status: owner preferences recorded; implementation brief. The questionnaire is complete. This establishes the selected direction, not acceptance of a finished application or proof of block-weight recommendations.

## Source

The owner identified `completed-20261003T194608-df92269d.json`. The local submission and latest answer file agree; `submitted` is true and the saved time is `2026-10-03T19:46:08.678538+00:00` (20:46 BST).

Receipt SHA-256: `8d3524d81e75807e1d334a64bee86cd42f0acae134f4367a02c58e2385b93987`.

Raw answers and supplied reference images stay in ignored local storage. This document records the product decisions so development can continue without asking the same questions again. Earlier visualization-widget events, including Guided Studio, are superseded by this explicit completed questionnaire.

## Selected direction

Two switchable themes share the same layout, controls, saved recipes and behaviour:

- **Prism:** violet glass with violet and cyan accents. The selected questionnaire direction makes this the initial theme; remember subsequent local theme choices.
- **Atelier:** light and emerald with mint and warm gold accents, explicitly requested in the owner's note.

Use the references numbered **5, 1 and 7** for translucent layered surfaces, luminous accents and clear multicolour comparison graphics. Selection order is not a ranking. Adapt those qualities to LoRA tasks; do not add unrelated dashboard statistics or decorative gauges.

| Decision | Owner selection | Implementation consequence |
| --- | --- | --- |
| Layout | Studio: library beside editor | Keep selected LoRAs visible while inspecting one vector; preserve chain order. |
| Density | Balanced | Comfortable spacing without spreading the main task over unnecessary screens. |
| Library | Small thumbnail and name | Keep role, filename and compatibility readable alongside the thumbnail. |
| Block display | Bars | Separate architecture-specific groups and make each slot selectable. |
| Editing | One block at a time | Exact numeric input and a bounded control for the selected block; use a personal variant. |
| Explanation | Plain advice first | Short reasons and uncertainty beside the relevant action; evidence can be opened when needed. |
| Comparison | Overlay with separate colours | Compare aligned block positions, with a labelled legend and individual series controls. |
| Experiments | Guided A/B with visible bounds | Show affected slots, A/B values, explicit min/max and the basis of each range. |
| Variants | Named variants with history alongside | Keep Default and previous versions reachable beside the editor. |
| Export | Card per loader in chain order | Each card identifies its LoRA and target node and includes the complete vector. |
| Motion | A little more expressive | Use transitions for panels and selection changes; keep editing steady and honour reduced motion. |
| Text | Larger | Larger everyday labels and explanations; do not squeeze them to preserve dashboard density. |

## Workspace specification

1. **Top bar:** current composition, model family, explicit loader target and theme switch. Put support status near the export action rather than presenting an unexplained global score.
2. **Library/selection column:** search and compatibility filters, small local thumbnails, names and roles. Keep the selected stack and its loader order easy to distinguish from the wider library.
3. **Block workspace:** the selected LoRA's complete group-labelled bars, a clearly marked selected slot and its exact value. Let the owner compare the stacked LoRAs in an overlay. Model/CLIP strength remains secondary to individual block values.
4. **Variant/history area:** show Default or the selected named version, its provenance and recent history beside editing. Editing Default creates a separate variant; returning to an earlier version preserves later history.
5. **Experiment area:** guided A/B values and their bounds appear near affected blocks. Numeric export remains available. Distinguish a suggested trial range from a loader limit or a visually evaluated range.
6. **Copy area:** one card per LoRA loader, in chain order. Show target, slot count, selected variant and full comma-separated values, with separate buttons for resolved numeric output and any supported A/B template/settings.

On narrower windows, stack these areas without truncating vectors or hiding exact controls. Both themes must provide readable text, visible keyboard focus and selection cues beyond colour. Larger text takes precedence over fitting all panels into one viewport.

These arrangements are implementation decisions derived from the answers, not additional owner claims. A normal theme toggle and responsive rearrangement need no further preference round. A materially different workflow should be brought back with a concrete example.

## Behaviour and acceptance checks

- The model/adapter/loader contract determines slot count, grouping and ordering, including BASE where required. The questionnaire's 58-slot FLUX.1 sample is not a universal vector format.
- Render signed block values faithfully. If a graph intentionally displays magnitudes, label it as such and retain the sign in the exact controls and export. Do not silently turn a negative value into positive influence.
- Comparisons use current selected variants, not stale Defaults. Keep LoRA identity colours separate from group labels so the overlay and editor remain understandable.
- Treat overlapping block positions as structural evidence. Do not claim measured image conflict, semantic influence or guaranteed image improvement from these graphs.
- Screen values, saved compositions and copied values derive from the same authoritative result. Preserve every required slot and numeric precision; no abbreviated vectors or frontend recomputation.
- Returning to Default or a prior version never deletes personal history. A saved recipe captures exact LoRA identities, profile versions, loader contract, settings and result.
- Unsupported export remains unavailable with a useful explanation; it must not look ready merely because a model was catalogued.
- Verify both themes, larger text, keyboard use, loading/empty/error states and narrow windows using a representative stack. Render-quality acceptance remains a separate controlled comparison with the owner.

## Delivery integration

GUI preference discovery is complete. Continue R0/R1 from `restart-2026-10-03.md`: preserve the baseline/data and prove FLUX.1 loader mappings and result ownership. In parallel, prepare the selected two-theme Studio shell and its component states against labelled fixtures; connect real data only through the verified result contract. Follow with profile/history integration and the complete choose → inspect → copy → save workflow.

No further colour/layout decision is currently needed. A current FLUX.1 acceptance specimen and controlled render judgement remain outstanding; the old supplied workflow is still a connection example. These do not block the baseline, loader tests or GUI foundation. No ComfyUI or model-tree modifications are authorised by this brief.
