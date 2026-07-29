# Phase 8.10a residual reconciliation review

Phase 8.10a converts the completed Phase 8.9 programme into a precise residual worklist. It is read-only and does not repeat the full mounted-library audit.

## Purpose

The Phase 8.9d plan originally identified:

- 311 new insert candidates
- 79 metadata backfill candidates
- 2 mounted rows missing stable IDs
- 3 same-family relocation candidates
- 20 cross-family reclassification candidates
- 668 stale current-family rows held unchanged
- 114 legacy or unmounted rows held unchanged

Phase 8.9e applied the approved metadata-only scope:

- 308 metadata inserts
- 49 metadata backfills
- 2 existing stable-ID assignments

The later targeted FLX phases completed:

- `FLX-PPL-207` with 57 `flux_unet_57` block rows
- `FLX-STL-263` with 56 `flux2_transformer_56` block rows

The remaining scanned candidate, `FLX-BDY-071`, has a damaged safetensors header and remains quarantined.

## Pinned evidence

Phase 8.10a requires:

```text
Phase 8.9d plan SHA-256:
e93d23901e4f05b0f250a0574c8662700be3428854c104f62146058e7ba6c7f2

Phase 8.9m verification SHA-256:
132805103fa858d7954245a309e81bcaa23c4062190d60252fc15a41fb655da7

Current database SHA-256:
6526505261ed62c79c433217161716e6d0bb9b286fb266867f9e6c87b1fa2357
```

Expected current database state:

```text
LoRA rows: 2834
Block rows: 4348
Rows with stable IDs: 2834
Duplicate stable IDs: 0
Orphan block rows: 0
```

## Verification coverage

The review verifies that the current database still contains:

- all 308 approved metadata inserts as metadata-only rows
- all 49 approved metadata backfills at their planned values
- both approved existing-row stable-ID assignments
- `FLX-PPL-207` with 57 block rows
- `FLX-STL-263` with 56 block rows

It verifies the unresolved items remain unchanged:

- one damaged scanned candidate, `FLX-BDY-071`
- 30 stable-ID-prefix metadata conflicts
- 3 same-family path relocation candidates
- 20 cross-family WAN reclassification candidates

For every relocation candidate, the old database path and stable ID must remain unchanged and the proposed destination path must remain absent from the database.

## Evidence limits

Phase 8.10a does not enumerate the LoRA library. Therefore:

- the 668 stale current-family rows are carried forward from Phase 8.9d
- the 114 legacy or unmounted rows are carried forward from Phase 8.9d
- those two counts are not described as freshly observed
- relocation identity remains unproven because Phase 8.9d used exact filename equality only

A fresh filesystem count requires a separately approved library audit.

## Decision baskets

### Damaged scanned target

`FLX-BDY-071` requires a clean replacement file before any new tensor analysis.

### Stable-ID-prefix conflicts

The 30 excluded metadata backfills require a policy decision between:

- preserving the existing stable-ID family prefix and references
- changing folder-derived family metadata, which may make the ID prefix inconsistent
- issuing replacement IDs, which would break stable-ID continuity

No automatic choice is made.

### Same-family relocations

The three same-family candidates preserve model family but still have only filename-based identity evidence. A later targeted evidence phase may inspect destination file size, metadata and available historical evidence. It must not relocate rows merely because filenames match.

### Cross-family reclassifications

The 20 WAN2.2 to WAN2.1 candidates require both identity evidence and an explicit stable-ID policy. Preserving IDs maintains references but leaves a `W22` prefix on a `W21` location. Reissuing IDs breaks continuity.

### Stale and legacy holds

The 668 stale and 114 legacy rows remain untouched until a fresh audit and separate policy discussion.

## Safety boundary

Phase 8.10a:

- opens SQLite with URI `mode=ro`
- enables `PRAGMA query_only = ON`
- does not invoke the indexer
- does not enumerate the library
- does not open safetensors files
- does not create a backup
- does not assign stable IDs
- does not relocate, update or delete rows
- writes only an optional JSON review report

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase810a_residual_reconciliation_review.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase810a_residual_reconciliation_review.py
```

Expected: eight tests pass and compilation is silent.

## Planned Nibbler run

After review and merge:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase810a_residual_reconciliation_review.py \
  --plan /data/phase89d_index_plan.json \
  --phase89m-report /data/phase89m_post_apply_closeout.json \
  --db /data/lora_master.db \
  --json /data/phase810a_residual_reconciliation_review.json
```

The database SHA-256 and backup count must remain unchanged.
