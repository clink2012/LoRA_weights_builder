# LoRA Comfy Combiner / LoRA Weights Builder

> Current owner scope (10 October 2026): [active library and graph drawing](docs/active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

A single-user, Bender-only application for selecting compatible LoRAs, comparing and editing individual block weights, preserving profile history, and copying a complete numeric vector into each ComfyUI loader. It does not merge or train LoRA files.

The first end-to-end target is **FLUX.1**, using a pinned standard FLUX.1 dev architecture and the installed Inspire Pack LoRA Loader (Block Weight) contract. The first owner-rendered [neutral loader comparison](docs/neutral-render-comparison-2026-10-04.md) produced an exact final-pixel match. Structural mapping and parameter measurements do not establish improved image quality; non-uniform balancing and broader release acceptance remain outstanding.

## Current checkpoint

The next active-family foundation is [MiniMax H3 loader characterization](docs/h3-loader-characterization-2026-10-10.md): two pinned candidates have independent behavior fixtures and exact research serialization, including separate main/refiner/other controls for the preferred maintainer route. H3 remains metadata-only until complete target mapping and workflow checks pass. No loader was installed and no family export was enabled.

The graph now supports press-and-hold drawing across bars, including fast strokes. The [10 October scope](docs/active-library-and-graph-drawing-2026-10-10.md) lists the eight live families and retires Pony, SDXL and Illustrious. Absent family folders do not cause scan errors; missing-file history stays intact.

The [main-button role-aware proposal](docs/main-button-role-start-2026-10-09.md) runs source measurement when needed and prepares declared role starting levels followed by measured overlap checks. New stacks show the computed proposal in their graph and full-vector export; original Defaults remain unchanged. Personal recipes and preferred shortcuts now load only by explicit choice. CPU analysis is required for this proposal path; ordinary saved-profile preparation remains available without it.

The [Studio feedback fixes](docs/studio-selection-and-graph-feedback.md) keep selection scrolling stable, hide candidates until their structural check finishes, prevent excluded files from being added through their inspection view, and enable direct editing of ordinary multiplier bars. Graph edits use personal revisions; measured update sizes remain a separate analysis result. The trial selector and adjacent buttons now share a baseline.

Carbon/compact Studio, automatic scans and compatible-library filtering merged in [PR #83](https://github.com/clink2012/LoRA_weights_builder/pull/83), merge `90d56db`, after standard CI37160278570 and actual CPU CI37160278462 passed against source `81b4ca6`. Its [package receipt](docs/compact-studio-and-library-checks.md) records local validation. Header observations and selected-database backups previously merged in PR #82 as `1b7bc0b`.

[PR #80](https://github.com/clink2012/LoRA_weights_builder/pull/80) merged as `4b29d32074e04504a39f6364cb623eb4351b346b`. Its authored source was `6021c313d53d898f107a1520b4e32ffa94853c57`; both [standard CI](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156465208) and the separate [real CPU validation](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156360963) passed on that source.

That checkpoint includes:

- Two switchable Studio themes, real local raster thumbnails, block bars, exact editing, aligned overlays and one export card per loader.
- Full canonical FLUX.1 values: BASE, 19 double blocks and 38 single blocks. The loader input is separately mapped to actual adapter coverage; sparse inputs may require unused padding.
- Immutable Defaults, named personal revisions, explicit roles and A/B bounds, and ordered compositions that restore exact saved values. Copy requires fresh preparation.
- Optional bounded CPU analysis of effective LoRA updates, signed pair measurements, cancellation and source/selection freshness checks. The normal API does not need Torch.
- An explicitly uncalibrated experiment using user-selected Protect/Normal/Flexible priorities. Changed weights are saved as new versions with the full policy and measurement receipt; Default remains unchanged.
- A built loopback launcher and tested recovery of profiles, recipes and experimental provenance into separate database copies.

The **catalogue-refresh package merged in [PR #81](https://github.com/clink2012/LoRA_weights_builder/pull/81) as `1c7f21d`**, following independent review, [standard CI](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499306) and [real CPU CI](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499234). Its copied-data rehearsal found 1,941 current safetensors files, added 420 missing catalogue entries and retained 1,003 missing-file records. All 2,522 existing IDs and historical analysis/profile tables were preserved. The original database remained untouched. See [library refresh](docs/library-refresh.md).

## Start here

| Document | Purpose |
| --- | --- |
| [Intent and completion contract](docs/intent-contract.md) | Product scope, invariants and what counts as finished |
| [Development roadmap](docs/development-roadmap.md) | Delivered packages and remaining work |
| [Studio measurements and experiments](docs/studio-analysis-and-experiments.md) | Saved profiles, measurements, priorities and trial bounds |
| [Exact loader contract](docs/inspire-export-contract.md) | BASE, full architecture values, sparse coverage and loader slots |
| [Effective-update analysis](docs/effective-update-analysis.md) | Optional CPU maths, supported formats and resource limits |
| [Local launcher](docs/local-launcher.md) | Built app, loopback operation and copied-database rehearsals |
| [Data backup and restore](docs/local-state-backup.md) | SQLite snapshots and limits of same-disk backups |
| [GUI design brief](docs/gui-design-brief-2026-10-03.md) | Completed questionnaire and approved Studio direction |
| [Restart prompt](docs/restart-prompt.md) | Continuity for a new management chat |

The [initial restart assessment](docs/restart-2026-10-03.md), older phase documents and Nibbler deployment records are historical evidence. Current behaviour and completion claims follow the intent contract and live roadmap. The old omitted-BASE export defect has a tested correction; do not restore the legacy 57-value CSV path.

## Run locally

On **Bender / PowerShell**, after the project dependencies are installed:

```powershell
Set-Location 'E:\LoRA Project'
.\tools\Launch-LoRA.ps1 -Build
.\tools\Launch-LoRA.ps1 -Database 'E:\LoRA Project\.local\preview-data\Database\lora_master.db'
```

The owner preview serves the built UI and API together at `http://127.0.0.1:5187`, accessible only on Bender. Its restored database copy is durable: preserve and back up `.local/preview-data/Database/lora_master.db`, where owner-created profiles and recipes are saved. Do not recreate it as a test fixture. The original main database remains unchanged. The launcher does not install dependencies or start ComfyUI; a later main-database launch makes a verified local backup before additive schema initialization. See the launcher guide before changing the selected database.

The ordinary app uses the project Python environment. CPU measurements use a separate optional `.venv-analysis` environment pinned by `Database/backend/requirements-analysis.txt`. ComfyUI and model files remain read-only. A background library scan starts once per server startup, with a manual scan, progress and potential file issues. It inventories paths and file details, then checks bounded headers without reading tensor payloads. The legacy global reindex endpoint is retired. After the first selection, Studio hides candidates that do not pass the current FLUX.1 loader checks; excluded and unverified files can be shown with reasons. See [library refresh](docs/library-refresh.md).

## Remaining release work

The existing person/clothing/style sample did not meet the first conservative policy's reduction threshold. The main-button method adds explicit role priors without lowering that threshold. Neither method establishes visually compatible stacking. Controlled renders, owner judgement and calibration are the next quality gate.

FLUX.1 completion does not finish the requested family expansion. LTX-2.3, LTX-2.5, MiniMax H3 and the existing families require separate architecture, adapter, loader and visual validation. Folder recognition alone grants no export capability. A focused companion node can be developed if necessary, with installation into the read-only ComfyUI tree handled separately.

Final release also requires finished-interface acceptance, a reviewed main-data launch and a complete recovery plan covering source, environments, SQLite data and irreplaceable LoRAs/results. Git and a scripts-folder backup alone do not cover all of that state.
