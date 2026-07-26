# Phase 8.9d read-only controlled indexing plan

This phase creates an explicit plan for index reconciliation before any database write is permitted.

The planner performs a fresh read-only filesystem/SQLite audit and separates the result into distinct review classes.

## Plan classes

### Same-family relocations

Files whose names uniquely match one stale row and one unindexed mounted path, while remaining in the same top-level model family.

The proposed policy is to preserve the existing database row and stable ID, subject to stronger identity verification before execution.

### Cross-family reclassifications

Files whose names uniquely match but whose top-level family changed, such as WAN2.2 to WAN2.1.

These are never treated as automatic moves. They require a policy decision because:

- preserving the existing stable ID protects external references
- the stable ID may retain an old family prefix such as `W22`
- regenerating the ID to use `W21` would violate stable-ID continuity

### New metadata inserts

Mounted files absent from the DB, excluding relocation candidates, whose registered base model and category can be parsed.

The plan predicts stable IDs using the existing lowest-unused-suffix policy but does not assign them.

### Existing mounted rows missing stable IDs

Current DB rows that still match mounted files and have valid base/category codes but no stable ID.

These are included in the same deterministic suffix plan as new inserts.

### Unparseable missing files

Mounted files for which a registered base code or recognised category code cannot be determined. These remain blocked from automatic insertion.

### Untouched stale and legacy rows

All unmatched stale rows and all legacy/unmounted-family rows remain untouched. This phase does not propose deletion.

## Identity limitation

Relocation matching currently uses unique case-insensitive filename equality only. The old files are no longer mounted, so byte-for-byte or hash identity cannot be established from the current library.

Every relocation remains advisory until a reviewed execution policy is approved.

## Run on Nibbler

After pulling the branch and rebuilding the backend image, run inside the backend container:

```bash
cd /home/clink/docker/lora_builder/app/LoRA_weights_builder/deploy/nibbler
sudo docker compose --env-file .env exec -T backend \
  python phase89d_index_plan.py \
  --root /loras \
  --db /data/lora_master.db \
  --json /data/phase89d_index_plan.json
```

The generated host-side report will be:

```text
/home/clink/docker/lora_builder/data/phase89d_index_plan.json
```

## Safety

The planner:

- opens SQLite using `mode=ro`
- enables `PRAGMA query_only=ON`
- does not invoke the indexer
- does not open safetensors tensors
- does not assign stable IDs
- does not insert or update rows
- does not delete stale rows
- does not change schema or scanner mathematics

## Test on Bender

From the repository root:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89d_index_plan.py `
  -q
```
