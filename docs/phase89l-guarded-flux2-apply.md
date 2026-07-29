# Phase 8.9l guarded Flux 2 apply tooling

Phase 8.9l adds guarded dry-run and apply tooling for the single sealed Flux 2 target `FLX-STL-263`.

Merging this tooling does **not** authorise a live database write.

## Immutable scope

The tool is hard-coded to the following reviewed evidence:

- stable ID: `FLX-STL-263`
- source: `FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors`
- source SHA-256: `c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7`
- Phase 8.9j analysis SHA-256: `7c886a07e87fa36081645d34bd578001420e65a380c6543b17c9b9ee1fb8dc48`
- Phase 8.9k artifact SHA-256: `adad9f9c3eb65bf0b2c0774e3c0c1508c43603ed3774c804f5ef49123d6a48df`
- model family: `Flux 2`
- LoRA type: `Flux 2 (PEFT double+single blocks)`
- rank: 16
- layout: `flux2_transformer_56`
- block rows: 56
- architecture: 8 double blocks and 48 single blocks
- adapted block modules: 128
- block tensors: 256
- global projection modules: 10
- global projection tensors: 20
- total tensor keys: 276

The tool cannot be redirected through CLI arguments to another stable ID, layout, source hash or block count.

## Database representation

The existing schema can store:

- the `Flux 2` model identity
- the exact LoRA type and rank
- the `flux2_transformer_56` layout
- all 56 ordered normalised block weights
- all 56 ordered raw block strengths

The schema has no truthful field for the 20 global projection tensors. They remain preserved in the sealed Phase 8.9k artifact and are explicitly excluded from per-block strengths. Phase 8.9l makes no schema change.

## Dry-run behaviour

Dry-run is the default. It:

1. loads the Phase 8.9k artifact
2. verifies its canonical digest
3. requires the digest to equal the exact reviewed Phase 8.9k digest
4. validates all immutable architecture and global-module evidence
5. validates 56 finite positive weights and 56 non-negative raw strengths
6. verifies the source SHA-256, size and modification time
7. opens SQLite with URI `mode=ro` and `PRAGMA query_only = ON`
8. verifies database integrity and required columns
9. verifies the target path and stable ID are absent
10. verifies `FLX-BDY-071` remains absent
11. verifies there are no duplicate stable IDs
12. prints the exact proposed row counts

Dry-run writes no database rows and creates no backup.

## Apply gates

Apply requires all of the following:

- `--apply`
- `--expected-artifact-sha256 adad9f9c3eb65bf0b2c0774e3c0c1508c43603ed3774c804f5ef49123d6a48df`
- `--backup-dir`

Before the transaction, the tool:

1. revalidates the source
2. creates a SQLite backup named with `phase89l`
3. verifies backup integrity, row counts, schema, stable-ID uniqueness and damaged-target absence

Inside `BEGIN IMMEDIATE`, the tool:

1. rechecks LoRA and block-row counts against the dry-run preview
2. rechecks stable-ID uniqueness
3. rechecks target absence and damaged-target quarantine
4. rechecks the source
5. inserts exactly one `lora` row
6. inserts exactly 56 contiguous `lora_block_weights` rows
7. verifies all inserted metadata, weights and raw strengths
8. verifies the `+1` and `+56` table deltas
9. verifies duplicate stable IDs remain zero
10. commits only after every check passes

Any exception rolls back the transaction. The verified backup remains available, but the tool does not perform an automatic restore.

## Post-commit checks

The tool reopens the database read-only and verifies:

- SQLite integrity
- exactly one row for the target stable ID and file path
- exactly 56 target block rows
- exact inserted metadata and vectors
- duplicate stable IDs remain zero
- `FLX-BDY-071` remains absent

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89l_guarded_flux2_apply.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89l_guarded_flux2_apply.py
```

Expected: 8 tests passed and silent compilation.

## Planned Nibbler dry-run

After review and merge:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89l_guarded_flux2_apply.py \
  --artifact /data/phase89k_flx_stl_263_flux2_artifact.json \
  --db /data/lora_master.db \
  --root /loras
```

The database SHA-256 and backup count must remain unchanged.

## Approval boundary

A clean dry-run does not itself authorise the write. The live apply remains locked until the user explicitly states:

```text
approve Phase 8.9l apply
```
