# Supplied workflow reference

Read-only inspection, 3 October 2026. The owner confirms this is an old illustrative workflow adapted to include FLUX.2; the library has changed since it was used. It demonstrates the loader arrangement and export needs, not a working current test or visual acceptance fixture. ComfyUI and the source workflow remain unchanged.

## Source identity

- Local source: `E:\ComfyUI_New\ComfyUI\user\default\workflows\Flux2-Master Template_Block_Weights.json`.
- SHA-256: `A60870E39C0D837ADAEBCC42D3DBE53ED8F464DF81F19BE15CCEDB2CDC6B9BDD`.
- Source prompts and private LoRA filenames are deliberately not reproduced here.

## Observed arrangement

Both MODEL and CLIP follow the same chain:

`LoraManager 377 → Inspire 608 → 625 → 609 → 611 → 610 → model/clip reroutes`

| Inspire node | Saved vector length | Observation |
| --- | ---: | --- |
| 608 | 20 | Graph-level bypass (`mode: 4`). |
| 625 | 20 | Graph-level bypass; saved LoRA path resolves at inspection. |
| 609 | 20 | Graph-level bypass. |
| 611 | 58 | Graph-level bypass; vector includes A/B placeholders. |
| 610 | 58 | Graph-level bypass. |

All five nodes are bypassed by the workflow graph even though their inner bypass widgets are false. The main loader is configured for FLUX.2 Klein; a separate upgrade section retains FLUX.1-dev. These are distinct model paths, not interchangeable evidence of loader compatibility. At inspection, only node 625's saved LoRA reference resolves; the other four have no exact basename matches in the current library. Those findings are capture-time facts, not grounds for modifying the old example.

## Product requirements confirmed by the owner

Provide one clearly labelled **complete individual block-weight vector per LoRA/node**, preserving the selected chain order and required supporting loader settings. Export must satisfy the verified architecture/adapter/loader contract, including BASE and group ordering; the saved 20/58-value strings above are examples, not universal valid lengths.

Make architecture-specific block groups such as FLUX single/double understandable in the GUI. Overall strength cannot replace individual block weights. Optional A/B experiments need explicit minimum/maximum values and provenance, with a fully resolved numeric export always available. Preserve Default and earlier variants.

Use a separate, controlled workflow with current compatible files for later FLUX.1 comparisons. This source can inform connectivity and export test cases; it does not establish that a balancing policy improves image quality.
