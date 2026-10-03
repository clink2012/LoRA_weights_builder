# Development roadmap

Current batch begins from `411a871` on 3 October 2026. This checklist records delivered behaviour, not a percentage estimate. Completion rules are in `intent-contract.md`.

| Package | Scope | Current state |
| --- | --- | --- |
| R0 | Preserve Bender state; current intent; dependable offline checks and CI | Complete for development baseline. Local checks, snapshot, separate restore rehearsal and hosted CI passed; PR #77 merged as `6912c96`. |
| R1 | Exact FLUX.1 loader adapter, coverage-aware output and one result owner | Foundation implemented and independently reviewed: pinned loader semantics, fresh bounded header coverage and numeric export. Neutral baseline only; balancing and render usefulness remain unproved. |
| R2 | Immutable Defaults, personal revision history and exact saved recipes | Implemented and locally verified, including API/UI integration, exact recipe restoration and copied-data backup recovery. Package CI/merge follows independent review. |
| R3 | Two-theme Studio connected to real library, block editing, comparison and copy | Named edits, role corrections, personal A/B bounds, overlays, bounded history and full-vector copy work in both themes. Finished-interface owner acceptance remains pending. |
| Balancing | Effective-update measurements and explained block-weight proposals | Optional bounded CPU measurement kernel under development. Automatic proposals and controlled calibration still pending; neutral Default remains unchanged. |
| R4 | Controlled FLUX render comparisons and owner acceptance | Pending. Old workflow remains an example, not a current acceptance fixture. |
| R5 | Existing-family capability truthfulness and LTX-2.3/2.5/MiniMax H3 expansion | LTX-2.5 and MiniMax H3 catalogue entries added without export claims. LTX-2.3 identity and all new family contracts need investigation. |
| R6 | Built local launcher, operating guide, recovery and release | State snapshot/rehearsal started; final application/data/environment recovery still pending. |

The automatic continuation instruction applies between these packages. There is no routine owner approval gate after a passing package. Owner involvement is required for render judgements, materially changed product choices, missing access, and installation changes to the read-only ComfyUI tree.

## Baseline preservation

- Local snapshot: `.local/backups/restart-411a871`.
- Separate restore rehearsal: `.local/restore-checks/restart-411a871`.
- SQLite integrity passed and recovered row counts were 2,524 LoRAs, 4,235 block rows, two personal profiles, zero combined profiles and zero role-override rows.
- Six files were captured: the database, external profile JSONs and completed questionnaire state. The manifest retains hashes and the source revision. The live database file hash was unchanged by the operation.
- This is a same-disk development checkpoint, not proof of Leela/Google Drive coverage or full application recovery. See `local-state-backup.md`.
