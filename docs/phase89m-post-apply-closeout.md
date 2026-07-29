# Phase 8.9m post-apply verification and closeout

Phase 8.9m provides a final read-only verification for the authorised Phase 8.9l insertion of `FLX-STL-263`.

It does not perform another apply, scan, reindex, schema migration, relocation or restore.

## Locked production evidence

The verifier is pinned to the production state established on 29 July 2026:

- merged tooling commit: `ba2fe7e21c28fa987c8406c561dc5f3ba276bbd7`
- sealed Phase 8.9k artifact SHA-256: `adad9f9c3eb65bf0b2c0774e3c0c1508c43603ed3774c804f5ef49123d6a48df`
- current database SHA-256: `6526505261ed62c79c433217161716e6d0bb9b286fb266867f9e6c87b1fa2357`
- rollback backup SHA-256: `d732b2739cd2df278b104e325d17481413152d89776d8f1abdc59637bc86c79e`
- rollback backup filename: `lora_master.phase89l.20260729T221823Z.adad9f9c3eb6.db`
- source SHA-256: `c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7`
- stable ID: `FLX-STL-263`
- inserted database row ID: `2834`
- layout: `flux2_transformer_56`
- current totals: 2,834 LoRA rows and 4,348 block rows
- backup totals: 2,833 LoRA rows and 4,292 block rows

## Verification depth

`phase89m_post_apply_closeout.py` verifies more than aggregate totals.

It performs the following read-only checks:

1. validates the exact current database, backup, artifact and source SHA-256 values
2. validates current and backup SQLite integrity
3. verifies the exact current and backup row totals
4. compares every one of the 2,833 pre-existing `lora` rows across every column
5. compares every one of the 4,292 pre-existing `lora_block_weights` rows across every column
6. requires the only new `lora` ID to be row `2834`
7. verifies all Flux 2 metadata for `FLX-STL-263`
8. verifies all 56 contiguous block rows against the sealed artifact weights and raw strengths
9. verifies all 2,834 current rows have stable IDs
10. rejects duplicate stable IDs
11. rejects orphan block rows
12. confirms `FLX-BDY-071` remains absent from both current and backup databases
13. confirms the approved source file remains unchanged
14. emits a canonical verification SHA-256

## Global projection evidence

The 20 global projection tensors remain preserved in the sealed Phase 8.9k artifact.

The existing database schema has no honest storage field for those tensors. Phase 8.9m confirms the artifact still declares them and does not invent a schema extension or fold them into the 56 per-block weights.

## Safety boundary

Phase 8.9m:

- opens both databases with SQLite URI `mode=ro`
- enables `PRAGMA query_only = ON`
- writes no database rows
- creates no backup
- has no apply mode
- does not enumerate the library
- opens only the approved Flux 2 source
- does not touch the damaged source
- writes only the optional JSON verification report

## Bender validation

From the repository root:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89m_post_apply_closeout.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89m_post_apply_closeout.py
```

Expected: 8 tests passed and silent compilation.

## Planned Nibbler run

After review and merge, execute inside a temporary backend container:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89m_post_apply_closeout.py \
  --artifact /data/phase89k_flx_stl_263_flux2_artifact.json \
  --db /data/lora_master.db \
  --backup /data/backups/lora_master.phase89l.20260729T221823Z.adad9f9c3eb6.db \
  --root /loras \
  --json /data/phase89m_post_apply_closeout.json
```

The database SHA-256 and backup count must remain unchanged.

## Completion condition

Phase 8.9 is closed when the live Phase 8.9m report verifies:

- current database integrity `ok`
- backup integrity `ok`
- 2,834 current LoRA rows
- 4,348 current block rows
- 2,833 preserved pre-existing LoRA rows
- 4,292 preserved pre-existing block rows
- one verified `FLX-STL-263` row with 56 verified blocks
- zero duplicate stable IDs
- zero orphan blocks
- `FLX-BDY-071` absent and untouched
- unchanged database SHA during verification
- healthy live API and containers
