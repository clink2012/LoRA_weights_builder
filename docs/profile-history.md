# Defaults, personal variants and composition recipes

> Current owner scope (10 October 2026): [active library and graph drawing](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

Default is a captured, immutable neutral starting point. It records the LoRA's source evidence, ordered architecture slots, target and loader fingerprints, engine/policy versions, exact values and supporting settings. Changing a block, role or experiment does not overwrite it: save a named personal revision, with its previous version retained. Choosing Default or an earlier revision changes the selection, not the historical entries.

## Studio workflow

1. Select compatible LoRAs and prepare their block values.
2. Open **Variants & history** for a LoRA to capture its current Default. Select a block, apply the exact value and save the draft under a personal name. The copy result becomes unavailable as soon as a draft or selection changes.
3. Optionally define A/B personal trial bounds for named blocks. The current value must be within the stated minimum and maximum. Bounds are personal experiments, not measured image-quality limits; numeric export always contains the resolved values.
4. Prepare again to obtain the chosen versions' fresh per-loader output. Returning to Default follows the same preparation step.
5. Capture any remaining Defaults and save a named **recipe**. It preserves ordered LoRA/version references and the exact server result. Loading a recipe restores the selection; preparing checks the current sources again before Copy.

A recipe save sends version references and the digest of the displayed preparation. The server recomputes it and rejects a changed result instead of silently saving values the user has not seen. Historical recipe payloads remain readable but are explicitly marked as requiring revalidation. They cannot directly supply current copy output.

## Changed sources and legacy records

A source file, target, loader or engine change starts a distinct Default lineage. Old versions remain preserved, but export is blocked if their binding no longer matches current evidence. Header/stat identity is explicit: it is not a claim that every tensor byte has been hashed or that an actual running checkpoint was verified.

The original personal and combined-profile tables remain intact. Legacy records do not become verified modern profiles by guessing a layout from vector length. The legacy UI is read only; new editing uses the versioned path. An internal explicit import helper exists for a separately verified mapping, but no automatic legacy import was performed.

New additive tables are `lora_profile_versions`, `lora_profile_selections` and `lora_composition_versions`. SQL triggers reject updates and deletions of historical rows. Local-state snapshots include all these tables and their triggers through SQLite's backup API. Development and restoration testing use copied data.

## API and authority boundaries

- `/api/profile-versions/{stable_id}` provides history, trusted Default capture, revisions and append-only selection.
- `/api/lora/prepare-blocks` accepts explicit `profile_version_ids`; every selected version is checked against fresh source and contract evidence before export.
- `/api/composition-versions` saves/list recipes; a historical read nests `historical_snapshot`, while `/{version_id}/prepare` performs fresh validation.

The native FLUX target covers model patches only. A saved CLIP-enabled experiment cannot be exported through that target. Unsupported mapping, invalid values, wrong-LoRA versions and stale source bindings remain explicit failures.

This package provides exact manual control and preservation. Automatic balancing, calibrated experiment advice and generated-image acceptance are separate work.
