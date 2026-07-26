# Phase 8.9f post-apply verification

Phase 8.9f verifies the completed Phase 8.9e metadata reconciliation by comparing three fixed inputs:

1. The reviewed Phase 8.9d JSON plan.
2. The SQLite backup created immediately before Phase 8.9e.
3. The current SQLite database after Phase 8.9e.

The verifier is strictly read-only. Both SQLite files are opened with URI `mode=ro` and `PRAGMA query_only = ON`.

## Live execution being verified

The reported Phase 8.9e execution was:

```text
backup_path: /home/clink/docker/lora_builder/data/backups/lora_master.phase89e.20260726T182850Z.e93d23901e4f.db
metadata_inserts: 308
metadata_backfills: 49
existing_id_assignments: 2
excluded_scanned_inserts: 3
excluded_id_prefix_backfills: 30
excluded_relocations: 23
```

The plan digest was:

```text
e93d23901e4f05b0f250a0574c8662700be3428854c104f62146058e7ba6c7f2
```

## Verification coverage

The verifier confirms:

- `PRAGMA integrity_check` returns `ok` for both backup and current databases.
- Current `lora` row count equals backup count plus the approved metadata inserts.
- Every row ID present in the backup still exists in the current database.
- Every approved metadata insert exists with its planned stable ID, file path, base/category codes and metadata-only state.
- No approved metadata insert has block-weight rows.
- Every approved backfill changed from the exact planned old value to the exact planned new value.
- Every approved existing-row stable-ID assignment was empty in the backup and equals the planned ID in the current database.
- Every excluded ID-prefix backfill remains unchanged across identity and category fields.
- Every same-family and cross-family relocation row remains unchanged across path, family, category and stable-ID fields.
- No duplicate non-empty stable IDs exist.
- `lora_block_weights` row count is unchanged.

The verifier does not modify either database, regenerate the plan, run the indexer or open safetensors files.

## Validation on Bender

From the repository root:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89f_post_apply_verify.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89f_post_apply_verify.py
```

## Live verification on Nibbler

After review and merge:

```bash
cd /home/clink/docker/lora_builder/app/LoRA_weights_builder/Database/backend

python3 phase89f_post_apply_verify.py \
  --plan /home/clink/docker/lora_builder/data/phase89d_index_plan.json \
  --db /home/clink/docker/lora_builder/data/lora_master.db \
  --backup /home/clink/docker/lora_builder/data/backups/lora_master.phase89e.20260726T182850Z.e93d23901e4f.db \
  --db-path-root /loras \
  --json /home/clink/docker/lora_builder/data/phase89f_post_apply_verification.json
```

A successful run reports `status: verified`. Any mismatch raises an error and exits without changing data.
