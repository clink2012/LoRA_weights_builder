# Current library refresh

The library package merged in [PR #81](https://github.com/clink2012/LoRA_weights_builder/pull/81) as `1c7f21dec20e6a7127b5b05ee59f982fe4d43826`, from source `e6f0b754c8f234ea0c58b9e8b93c789690694cf1`. [Standard CI](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499306) and [real CPU CI](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37157499234) passed on that source, including the actual optional CPU job. The original Bender database has not been refreshed or migrated during this work.

The original PR #81 inventory step reads paths and file details only. The subsequent owner-requested startup scanner runs that inventory first, then bounded header checks in the background. Neither step reads tensor payloads or requires Torch. Discovery currently covers `.safetensors` files; it does not claim every possible LoRA file format. Selected FLUX preparation separately validates actual headers against the pinned architecture and loader contract.

## Behaviour and history

- Existing catalogue row IDs and stable IDs are retained. New files receive unused IDs; files outside a recognized family/category receive an `UNK` component. Existing identifiers and history reserve their numbers, including suffixes above 999.
- Missing files retain their catalogue entries and all saved history. **Current**, **Missing** and **All** views separate presence from historical records. Presence describes the last completed inventory; actual export preparation performs fresh file checks.
- Same-name files at new paths are new entries. Refresh never guesses that a filename match proves a relocation or transfers an old profile lineage automatically.
- Family/category recognition comes from folder hints. New LTX, MiniMax and other recognized family rows do not acquire architecture, tensor analysis or loader-export capability. Existing row metadata and analysis caches remain unchanged; observed hints are recorded separately.
- Previously saved roles, priorities, Defaults, personal variants, recipes and experiments are preserved. New folder-derived role hints are labelled as hints.
- Links, Windows junctions, auxiliary `recipes` and `LoRA_Manager_Images` folders, and unsupported file extensions are outside the scan's scope. Their existing records are retained as out of scope rather than falsely marked missing.

The implementation stores presence and scan receipts in additive `lora_catalogue_presence` and `lora_catalogue_scans` tables. It does not replace the old catalogue or remove rows. Before the first completed refresh, the current view is empty with `catalogue_status: not_refreshed`; the All view can still show unchecked historical entries. The owner's later instruction supersedes the original manual-only preference: scan once at server startup, not on every browser refresh, and retain a manual scan action.

## Completeness and concurrency

Two independent no-follow inventories must agree. File and directory identities are checked again inside the database transaction immediately before commit. A changed, inaccessible or incomplete inventory applies no new entries, IDs or missing-file changes. Duplicate existing IDs or normalized paths are reported for resolution, never silently merged.

The initial bounds are 10,000 safetensors files, 10,000 directories, 100,000 directory entries and 60 seconds. A process-shared local lock permits one refresh at a time, and `BEGIN IMMEDIATE` protects ID allocation and the atomic update. This is a checked inventory, not a filesystem snapshot or full-file cryptographic identity. Later file changes are still possible and are handled by fresh preparation.

The root is selected by server configuration (`LORA_ROOT`, normally `E:\models\loras`), and writes use that process's selected database. Browser requests cannot replace either path. No model or ComfyUI file is changed.

## API and retired operation

`POST /api/catalogue/refresh` remains the inventory-only endpoint. The Studio uses `POST /api/library-scan {}` to start the background inventory plus header audit, `GET /api/library-scan` for progress, and `/cancel`, `/resume` and `/issues` for cancellation, remaining checks and observations. Concurrent requests coalesce within the process; another process receives a busy response. CPU measurement jobs remain separate.

The header audit has a 16 MiB per-header ceiling and batch limits of 512 MiB header reads, 90 seconds and 5,000 files. A partial batch explicitly records remaining work and can resume. Persisted observations are attached to the exact catalogue scan and file identity in `lora_library_scan_jobs` and `lora_library_observations`. Potential folder/metadata/key disagreement, incomplete pairs and unreadable headers are visible. Unsupported inspection and unknown families remain explicitly unverified, rather than being called damaged or compatible. No files are renamed, moved or repaired automatically.

`POST /api/catalogue/compatible` uses the first selected stable ID and the pinned target contract to check candidates before pagination. Default browsing hides excluded and unknown candidates, while an optional view shows reasons. Bounded caching is invalidated by catalogue, file identity or loader/source changes, and read access is checked again. The present verified target is FLUX.1; other families remain available for catalogue/observation work without pretending their exports are verified.

`GET /api/catalogue` accepts `presence=current|missing|all`, `base`, `category`, `search`, `limit` and `offset`. Its paginated results retain catalogue fields and add presence, observed metadata provenance and unverified-analysis labels. Current is the default.

The old `POST /api/lora/reindex_all` returns **410 Gone**, directing the caller to Refresh library. It can no longer invoke the legacy tensor indexer or ID assigner. Do not install Torch into the ordinary API to revive that operation. Optional measurements use the separate project CPU worker described in [effective-update-analysis.md](effective-update-analysis.md).

## Copied-data verification

On 3 October 2026, a rehearsal used the real library with a separate copy at `.local/restore-checks/catalogue-refresh-review-01/lora_master.db`:

| Result | Count |
| --- | ---: |
| Current safetensors files | 1,941 |
| Newly catalogued files | 420 |
| Existing blank IDs assigned | 2 |
| Missing entries retained | 1,003 |
| Existing stable IDs preserved exactly | 2,522 |

The refresh completed in approximately 0.98 seconds on this capture. All four legacy analysis/profile tables matched their pre-refresh logical hashes, no duplicate IDs were introduced, and the original database SHA-256 remained unchanged. Timing is an observation, not a guarantee. The private receipt is `.local/restore-checks/catalogue-refresh-review-01/refresh-receipt.json`.

At the backend checkpoint, **28 focused and independent adversarial tests passed**. They cover stable IDs and history, missing/returned files, no filename relocation, unknown categories, new-family metadata, duplicates, interrupted scans, mid-scan changes, budgets, Windows links/junction handling, process-shared locking and retirement of the old indexer. The full pinned CPU suite passed **360 backend tests**, **18 tooling checks** passed, and the UI passed **60 tests, lint and build**, with independent review clear.

The built browser rehearsal at port 5187 confirmed the first refresh counts above and zero additions on the next refresh. An unsaved draft BASE value of `-0.56789` and its name survived filtering and refresh. Copy remained unavailable; the draft was then discarded, and fresh preparation restored export. No named QA revision or recipe was saved in the owner preview. Both themes were checked at four viewport widths without horizontal overflow or browser errors. The private receipt is `.local/studio-qa/catalogue-browser-receipt.json`. This proves the exercised catalogue/UI workflow; finished-interface owner acceptance and rendered-image assessment remain separate.

The owner preview is operational at `http://127.0.0.1:5187` against the durable restored copy `.local/preview-data/Database/lora_master.db`. Preserve that copy and any owner-created versions. It is distinct from the disposable refresh-rehearsal database above; see [local-launcher.md](local-launcher.md) for the exact launch command.

Before eventual main-data use, keep the launcher's verified backup and the separate restore evidence. SQLite recovery includes the new presence/scan tables along with saved versions and recipes; the model library remains a separate backup concern. See [local-state-backup.md](local-state-backup.md).
