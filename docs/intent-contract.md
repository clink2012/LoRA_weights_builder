# Product intent and completion contract

Authority: the owner's restart instructions and completed GUI questionnaire, reconciled in October 2026. This current contract supersedes incompatible historical phase plans and Nibbler deployment assumptions. Historical evidence remains available in the restart assessment.

## Product

LoRA Comfy Combiner runs for one user on Bender. It helps select compatible LoRAs, record intended roles, compare their block contributions, adjust or propose individual block multipliers and copy complete values into separate ComfyUI loaders. A saved composition makes the choices reproducible. It does not merge or train LoRA files.

The primary output is a full ordered numeric vector **per LoRA and loader**, not one global strength. Its grouping and slot count derive from the model architecture, adapter coverage and exact loader contract. Global model/CLIP strengths and A/B variables are supporting settings. Always distinguish a canonical architecture vector from a target loader's actual input sequence when they differ.

## Invariants

- Runtime binds to Bender loopback only; no LAN listener, account, remote API, telemetry or required runtime download. Development research and source control are separate from runtime requirements.
- ComfyUI and model trees remain read-only. A required companion node may be developed here; installation is a separate owner involvement point.
- Stable file identity, architecture, adapter kind, coverage and loader identity underpin compatibility. A folder name, block count or model-family registration alone cannot establish export support.
- Preserve immutable Default settings and all named personal revisions. Manual values, roles and experiment bounds belong to separate versions. Returning to Default must not delete history.
- The same authoritative calculated result supplies the screen, copy action and saved recipe. Do not recalculate or round it differently in the UI.
- A/B experiments name affected slots and expose current values, min/max and the basis of bounds. Always offer fully resolved numeric export.
- Structural overlap and effective-update measurements can inform experiments; neither establishes semantic conflict or guarantees render quality. Fixed role policies are visibly identified as heuristics until calibrated.
- Unknown coverage, unknown loader behaviour or incompatible architecture blocks the affected export with an actionable explanation. Do not silently fall back to a generic vector.
- The approved preference brief controls GUI implementation: two-theme Studio, larger text, block bars and overlays, focused editing, preserved history and full-vector cards in chain order. See `gui-design-brief-2026-10-03.md`.

## First useful release: FLUX.1

Completion requires all of the following, with evidence recorded against the released source revision:

1. The user can browse the current local library, select compatible LoRAs across search/pages, confirm their roles and maintain their loader order.
2. FLUX.1 architecture and adapter coverage are identified from actual files. Independent contract fixtures prove BASE, group ordering, partial coverage, signed values, precision and unsupported cases against the chosen loader version.
3. The UI shows complete per-LoRA values and explains suggestions. Selected blocks can be edited into a separate named version. Default, prior versions and saved recipes round-trip exactly.
4. Every copied value and supporting setting belongs to an explicit loader/card. The copied result matches the saved result. A/B and numeric routes agree after substitution.
5. Both selected themes, larger text, keyboard access and empty/loading/error states work. No synthetic fixture is displayed as a real analysis result.
6. Focused tests, full applicable offline tests, UI lint/build, representative read-only file Characterisation, independent review and configured CI pass. Skipped optional tensor tests are stated explicitly and do not imply tensor correctness.
7. Controlled ComfyUI comparisons use pinned checkpoint, LoRAs, prompts, seeds and sampler settings. The owner judges whether the result is useful relative to individual LoRAs and ordinary stacking. This is a real involvement point, not replaceable by a green test suite.
8. A documented loopback launcher starts the built local app. State backup and restore into a separate location recover profiles and recipes. Required source, dependencies and irreplaceable data coverage are documented, including the limits of same-disk snapshots.

## Family expansion and project finish

FLUX.1 is the first end-to-end release, not removal of other families from scope. Existing catalogue families remain visible with truthful capability states. LTX-2.3, LTX-2.5 and MiniMax H3 are required expansion targets; exact versions, modality/control adapters and generation variants need separate evidence. Extend architecture and comparison foundations before export where necessary. Investigate existing loaders before building a narrow companion node.

Use separate states for catalogue recognition, architecture identification, analysis, composition, verified loader export and visual evaluation. Each released family repeats the relevant mapping, export, persistence and render checks. Finishing FLUX.1 alone is not completion of the entire requested project.

## Work and authority

Management may dispatch independent agents, review, test, commit, push and merge successful packages and continue automatically. Preserve data before migrations and use copies during development. Stop only for required owner input, access/authentication, a consequential scope change or an unresolved material defect. Do not turn routine implementation choices into approval gates.

Current development order: protected baseline and checks → trustworthy FLUX export/result ownership → immutable variants/recipes → integrated Studio workflow → controlled render acceptance → family expansion → packaged local operation and recovery. GUI foundation and independent architecture research can run alongside backend work. Keep the live checklist in `development-roadmap.md`.
