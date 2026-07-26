# Phase 8.9 read-only model ecosystem audit

This audit compares the mounted LoRA library with the current SQLite index without invoking the indexer, opening safetensors tensors, changing schema, or writing to the database.

It reports:

- top-level mounted folders and `.safetensors` counts
- DB rows by `base_model_name` and `base_model_code`
- `stable_id` coverage
- scanned, fallback-only, metadata-only, and inconsistent block-weight rows
- DB paths whose mounted file is no longer present
- mounted files missing from the DB
- duplicate or unresolvable DB paths
- a per-folder support matrix

`LoRA_Manager_Images` and `recipes` are ignored by default.

## Safety properties

- SQLite is opened with `mode=ro` and `PRAGMA query_only=ON`.
- The script never imports or calls `lora_indexer`.
- The LoRA mount remains read-only.
- File comparison uses canonical paths relative to the library root, so Windows DB paths such as `E:\models\loras\...` can be compared with Nibbler paths under `/loras/...`.

## Run on Nibbler

After pulling the branch and rebuilding the backend image, run inside the backend container:

```bash
cd /home/clink/docker/lora_builder/app/LoRA_weights_builder/deploy/nibbler
sudo docker compose --env-file .env exec -T backend \
  python phase89_audit.py \
  --root /loras \
  --db /data/lora_master.db \
  --json /data/phase89_audit.json
```

The console shows the summary and matrix. The full discrepancy lists are written to:

```text
/home/clink/docker/lora_builder/data/phase89_audit.json
```

This command does not call `/api/lora/reindex_all`, `lora_indexer.py`, or any reindex endpoint.

## Run tests on Bender

From **Bender / VS Code PowerShell**:

```powershell
cd 'E:\LoRA Project'
python -m pytest Database\backend\tests\test_phase89_audit.py -q
```
