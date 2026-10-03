# Development roadmap

The restart baseline was `411a871` on 3 October 2026. The latest merged application package is PR #81, authored at `e6f0b754c8f234ea0c58b9e8b93c789690694cf1` and merged as `1c7f21dec20e6a7127b5b05ee59f982fe4d43826`. This checklist records delivered behaviour, not a percentage estimate. Completion rules are in `intent-contract.md`.

| Package | Scope | Current state |
| --- | --- | --- |
| R0 | Preserve Bender state; current intent; dependable offline checks and CI | Complete for development baseline. Local checks, snapshot, separate restore rehearsal and hosted CI passed; PR #77 merged as `6912c96`. |
| R1 | Exact FLUX.1 loader adapter, coverage-aware output and one result owner | Foundation implemented and independently reviewed: pinned loader semantics, fresh bounded header coverage and numeric export. Neutral baseline only; balancing and render usefulness remain unproved. |
| R2 | Immutable Defaults, personal revision history and exact saved recipes | Complete foundation: API/UI integration, exact recipe restoration and copied-data backup recovery verified. Independent review and hosted CI passed; PR #78 merged as `d69017e`. |
| R3 | Two-theme Studio connected to real library, block editing, comparison and copy | Named edits, roles, A/B bounds, overlays, history, full-vector copy, local raster thumbnails, preview/main status, CPU measurements and guided experiments merged in PR #80 after independent review and CI. Finished-interface owner acceptance remains pending. |
| Balancing | Effective-update measurements and explained block-weight proposals | CPU maths merged in PR #79 (`e0d191f`); app job control, uncalibrated gentle-reduction preview and atomic experiment history merged in PR #80. Controlled calibration remains pending; neutral Default unchanged. |
| Library | Discover current files and preserve missing-file history | Merged in PR #81 after independent review, standard CI and the actual CPU job passed. Copied-data rehearsal: 1,941 current safetensors files, 420 additions, 1,003 missing records retained and all 2,522 existing IDs preserved. Built browser checks, 360 CPU backend tests, 18 tooling checks and 60 UI tests passed. Path/stat inventory only; no speculative relocation or export claims. See `library-refresh.md`. |
| R4 | Controlled FLUX render comparisons and owner acceptance | Pending. Old workflow remains an example, not a current acceptance fixture. |
| R5 | Existing-family capability truthfulness and LTX-2.3/2.5/MiniMax H3 expansion | LTX-2.5 and MiniMax H3 catalogue entries added without export claims. LTX-2.3 identity and all new family contracts need investigation. |
| R6 | Built local launcher, operating guide, recovery and release | Built single-process loopback launcher implemented; copied-database start/status/stop/restart passed on Bender. Main-database launch and final application/data/environment recovery remain pending. |

The automatic continuation instruction applies between these packages. There is no routine owner approval gate after a passing package. Owner involvement is required for render judgements, materially changed product choices, missing access, and installation changes to the read-only ComfyUI tree.

PR #80 passed [standard CI run 37156465208](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156465208) and [real CPU run 37156360963](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156360963), both against its authored source. The catalogue package separately passed [standard run 37157499306](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499306) and [real CPU run 37157499234](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499234) against `e6f0b754c8f234ea0c58b9e8b93c789690694cf1`. See `studio-analysis-and-experiments.md` for the local and copied-data evidence.

## Baseline preservation

- Local snapshot: `.local/backups/restart-411a871`.
- Separate restore rehearsal: `.local/restore-checks/restart-411a871`.
- SQLite integrity passed and recovered row counts were 2,524 LoRAs, 4,235 block rows, two personal profiles, zero combined profiles and zero role-override rows.
- Six files were captured: the database, external profile JSONs and completed questionnaire state. The manifest retains hashes and the source revision. The live database file hash was unchanged by the operation.
- This is a same-disk development checkpoint, not proof of Leela/Google Drive coverage or full application recovery. See `local-state-backup.md`.
