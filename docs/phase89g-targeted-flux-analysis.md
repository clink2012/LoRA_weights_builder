# Phase 8.9g targeted Flux analysis plan

Phase 8.9g inspects only the three FLX files that were deliberately excluded from the Phase 8.9e metadata-only reconciliation.

It is a read-only planning phase. It does not contain an apply mode.

## Why this phase exists

The general indexer discovers every `.safetensors` file beneath the configured LoRA root. That is unsuitable for the controlled Phase 8.9 workflow because it could revisit thousands of existing rows.

The three excluded FLX candidates require real tensor-key inspection and the Flux delta-analysis engine. Phase 8.9g reuses those per-file analysis components without invoking the full filesystem indexer.

## Scope controls

The analyser consumes the reviewed Phase 8.9d JSON plan and:

- selects candidates whose `base_model_code` is exactly `FLX`
- requires exactly three candidates by default
- accepts only `source_type = new_metadata_insert`
- opens only the exact plan-listed relative paths
- rejects absolute paths, `..` traversal and paths escaping the approved root
- checks that every target is still absent from the database
- checks that every planned stable ID remains unused
- validates the planned stable-ID family and category prefix
- opens SQLite with `mode=ro` and `PRAGMA query_only = ON`

It does not enumerate the library and never calls `lora_indexer.main()`.

## Analysis output

For each target, the JSON records:

- relative and database paths
- planned stable ID
- SHA-256 and file size
- tensor-key count
- CLIP contributor status and tensor count
- analysed model family, LoRA type and rank
- block count and resolved layout
- block-weight minimum, maximum and mean
- warnings and controlled-apply readiness

A candidate is blocked from a later controlled apply when:

- analysed blocks do not map to a supported block layout
- raw-strength and block-weight counts disagree

No database row or block-weight row is created in this phase.

## Validation

From the Bender repository root:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89g_targeted_flux_analysis.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89g_targeted_flux_analysis.py
```

## Planned Nibbler run

After review and merge:

```bash
cd /home/clink/docker/lora_builder/app/LoRA_weights_builder/Database/backend

python3 phase89g_targeted_flux_analysis.py \
  --plan /home/clink/docker/lora_builder/data/phase89d_index_plan.json \
  --root /home/clink/docker/lora_builder/mounts/loras \
  --db /home/clink/docker/lora_builder/data/lora_master.db \
  --db-path-root /loras \
  --expected-count 3 \
  --json /home/clink/docker/lora_builder/data/phase89g_flux_analysis.json
```

The current database should be hashed before and after the run. Both hashes must match.
