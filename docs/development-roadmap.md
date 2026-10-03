# Development roadmap

Current batch begins from `411a871` on 3 October 2026. This checklist records delivered behaviour, not a percentage estimate. Completion rules are in `intent-contract.md`.

| Package | Scope | Current state |
| --- | --- | --- |
| R0 | Preserve Bender state; current intent; dependable offline checks and CI | Local checks, snapshot and separate restore rehearsal passed. Hosted CI awaits the foundation PR. |
| R1 | Exact FLUX.1 loader adapter, coverage-aware output and one result owner | Foundation implemented and independently reviewed: pinned loader semantics, fresh bounded header coverage and numeric export. Neutral baseline only; balancing and render usefulness remain unproved. |
| R2 | Immutable Defaults, personal revision history and exact saved recipes | Individual history store under development; API/UI integration and whole-composition recipes follow the foundation checkpoint. |
| R3 | Two-theme Studio connected to real library, block editing, comparison and copy | Both themes, real library selection and per-loader copy implemented. Editing, overlays and history integration follow R2. |
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
