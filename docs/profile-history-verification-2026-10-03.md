# Profile and recipe history verification — 3 October 2026

This package extends foundation `20c0ced` / merged `6912c96`. It connects immutable personal versions and ordered composition history to actual numeric export. It does not introduce automatic balancing or claim rendered-image acceptance.

## Backend and preservation

- Full applicable backend suite: **208 passed**, one optional tensor module skipped in the project environment.
- Individual history covers immutable Default capture, append-only revisions/selections, exact values, bounded personal A/B experiments, source/target/loader/engine binding, explicit legacy import, concurrency and reopening persisted records.
- Composition history covers ordered pinned references, exact stored server preparation, digest mismatch rejection, immutable parent/child history, concurrent saves, prohibited client snapshots and revalidation of restored recipes.
- HTTP integration exercises trusted capture, personal export, wrong-LoRA/stale-source rejection, Default rollback, supporting settings, A/B metadata and two-LoRA recipe round-trips.
- Independent cross-review found no backend blockers. A/B scalar metadata was aligned with the stored experiments; numeric exports were already fully resolved.

The real copied-database exercise captured three actual local LoRA Defaults, saved an exact block edit with personal A/B bounds, prepared a three-LoRA composition, saved it and obtained an identical freshly restored preparation. Returning to Default recovered its original canonical values without deleting history. The private receipt is `.local/profile-history-runtime-receipt.json`.

The new history was then snapshotted and restored separately to `.local/restore-checks/profile-history-runtime`. All profile, selection and composition rows matched the captured state; restored immutability triggers still rejected deletion. At that capture the database held four profile versions and one composition version. Later browser exercises may add further QA records to the development copy.

Every row of the five legacy tables in the running development copy was compared with the original database and remained identical: 2,524 LoRAs, 4,235 block rows, two personal profiles, zero combined profiles and zero role overrides. Original database SHA-256 at verification: `9af0501b6aab2d9aabb63e30bb36062b0a76c0eb61bb84f2a136dcf0330a2d35`. No ComfyUI or model file was changed.

## Interface checks

The Studio now edits canonical block values into named drafts, shows prior versions alongside, supports personal A/B bounds and role/model-strength corrections, and saves/restores pinned recipes. Drafts and version changes invalidate Copy until fresh preparation. Legacy profile mutation controls are disabled.

Independent review identified asynchronous cases that could lose a draft during recipe loading or saving. Operation locks and mounted-workspace guards were added; failed HTTP responses cannot supply a recipe digest or copy authority. Deferred-response regressions cover those cases. **27 UI tests, lint and the production build passed.** A browser exercise against the copied database saved a signed BASE value, personal A/B bounds and a recipe, restored Default, reloaded the same LoRA's pinned recipe version, then freshly prepared the original exact 58-slot output. Both themes were inspected; widths 1024, 736, 390 and 320 had no page overflow or JavaScript errors. History scrolling is bounded. Private screenshots remain under `.local/studio-qa`.

Hosted CI must pass for the package commit before merge. These functional and browser checks do not constitute owner acceptance of the finished interface.

See `profile-history.md` for the workflow and API/data boundaries. Effective-update measurements, explained balancing proposals and controlled render evaluation remain separate work.
