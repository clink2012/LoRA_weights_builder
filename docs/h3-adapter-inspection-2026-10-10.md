# MiniMax H3 adapter inspection

This package adds a visible, read-only H3 inspection path. The earlier PR #102 package characterized candidate loaders without adding usable H3 controls to the app.

## In the app

Choose **MiniMax H3** under Base model, then **Run search**. Each H3 library card offers **Inspect**. Inspection shows complete ordinary low-rank pair counts and observed ranks for main-block, token-refiner and explicitly recognized other modules. It does not add the H3 adapter to a FLUX stack, change a Default or create a saved recipe. An existing FLUX stack remains available.

The empty H3 workspace names adapter inspection instead of displaying a FLUX export target. The loader preparation button remains disabled for an empty stack. Full H3 multipliers, editable contribution graphs and ComfyUI export are still unavailable; inspection does not grant those capabilities.

The owner's empty-list screenshot was not reproduced on the running PR #102 build: the catalogue and filesystem both contained 92 current H3 adapters, and selecting H3 and running the search displayed them. No catalogue repair or automatic filter application was introduced. Reload the browser and run the search if the displayed results are stale.

## Evidence and limits

`GET /api/h3-inspection/{stable_id}` resolves a catalogue identity within the configured library. It accepts no client file path. Each request reads a fresh, bounded safetensors header, checks file identity and refuses missing files, outside-library paths, links/junctions and files that change during inspection. It writes neither catalogue nor model data.

The graph's bar heights are **pair counts**, not learned tensor magnitudes, original block weights or suggested multipliers. Ranks are dimensions from header shapes. Sparse indices do not establish total model depth. Names and shapes alone do not verify the checkpoint variant, actual target aliases, patch application or image quality. Even modules whose names fit a group remain observations, not validated native targets.

Unknown tensors and modules never become implicit OTHER. Incomplete, mixed-format, duplicate or shape-incompatible pairs contribute no coverage. Issue examples are capped at 30 while the total issue count remains visible. LoKr and other unsupported adapter formats receive a review state, not a corruption diagnosis or a synthetic weight vector.

The local read-only characterisation on 10 October observed 92 current files: 90 with accounted ordinary pairs and two requiring review because of LoKr tensors. It read 5,232,893 header bytes in total and no tensor payloads. These are capture-time library facts, not test fixtures or permanent inventory assumptions. Local details remain in `.local/h3-header-audit.json`; that report and the models are excluded from source transfer.

## Validation and next work

Regression coverage includes pair formats/ranks, sparse indices, unknown modules, incomplete/mixed/duplicate pairs, malformed headers, missing/outside/linked files, fresh inspection after changes, file changes during a request, source/database preservation, UI errors and stale-response cancellation. UI checks also preserve an existing FLUX stack and keep H3 inspection separate from preparation/export.

Before enabling H3 block analysis or export, prove the adapter-to-native-target map against the pinned ComfyUI model and loader sources. Check complete model groups, task/checkpoint variants, adapter format and alpha handling, required control/distillation adapters and actual loader application. The preferred maintainer candidate has separate main/refiner/other controls; its research serializers in PR #102 remain non-exporting until those prerequisites are met. A required installation into the read-only ComfyUI tree remains a separate owner involvement point.

GitHub and Farnsworth must validate the exact authored source before merge. Back up and verify the preview database before rebuilding/relaunching with the official launcher. Record source, merge, runtime and preservation receipts locally without transferring personal data.
