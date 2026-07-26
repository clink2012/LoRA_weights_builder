# Phase 8.9i controlled single Flux apply

Phase 8.9i is the first write-capable step for the single FLX candidate that passed Phase 8.9g diagnostics and was sealed by Phase 8.9h.

It is intentionally limited to:

- stable ID `FLX-PPL-207`
- path `/loras/FLUX/01 - People/ang3l4wh1t3-f1.safetensors`
- one `lora` row
- 57 `lora_block_weights` rows
- layout `flux_unet_57`
- artifact digest `f38b59207dfb0eafc003f92c7a8ad25996994a1a773e589ed8eae870109a0932`

The two blocked FLX candidates remain untouched.

## Default behaviour

The command defaults to dry-run. Dry-run:

- verifies the artifact canonical digest
- verifies the sealed source SHA-256, size and mtime
- checks database integrity and required schema
- confirms the target file path is absent
- confirms the stable ID is unused
- confirms the artifact contains exactly 57 finite normalised weights and 57 finite raw strengths
- prints the exact row counts that would be inserted
- makes no database changes and creates no backup

## Apply gates

Apply mode requires all of the following:

- `--apply`
- `--expected-artifact-sha256` equal to the exact sealed artifact digest
- `--backup-dir`
- source file still matching the sealed SHA-256, size and mtime
- database row counts unchanged since dry-run preview
- target path and stable ID still absent

Before the transaction, the script creates a SQLite backup and verifies:

- `PRAGMA integrity_check = ok`
- backup `lora` row count matches the current database
- backup `lora_block_weights` row count matches the current database

## Transaction behaviour

The write uses:

- `PRAGMA foreign_keys = ON`
- `PRAGMA busy_timeout = 10000`
- `BEGIN IMMEDIATE`

Inside the transaction it:

1. rechecks both table row counts
2. rechecks source SHA-256, size and mtime
3. rechecks target path and stable-ID absence
4. inserts one fully analysed `lora` row
5. inserts 57 contiguous block rows with the sealed stable ID
6. verifies all inserted metadata, block indices, weights and raw strengths
7. verifies table row-count deltas are exactly `+1` and `+57`
8. commits only after all checks pass

Any exception rolls back the transaction. A verified pre-write backup remains available.

## Explicit approval boundary

Merging Phase 8.9i does not authorise a live write.

The live apply command must not be issued until the user explicitly approves:

```text
approve Phase 8.9i apply
```

Approval must be received before the apply command is run.

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89i_controlled_flux_apply.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89i_controlled_flux_apply.py
```

Expected: 6 tests passed and silent compilation.

## Planned Nibbler dry-run

After review and merge, run inside a temporary backend container:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89i_controlled_flux_apply.py \
  --artifact /data/phase89h_flx_ppl_207_artifact.json \
  --db /data/lora_master.db \
  --root /loras \
  --expected-stable-id FLX-PPL-207 \
  --expected-block-count 57 \
  --expected-layout flux_unet_57
```

The database SHA-256 must be identical before and after dry-run.
