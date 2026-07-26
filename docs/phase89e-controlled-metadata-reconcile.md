# Phase 8.9e controlled metadata reconciliation

This phase introduces a guarded, write-capable metadata reconciler. It does not run automatically and remains dry-run by default.

The reconciler consumes the exact Phase 8.9d JSON plan and applies only the approved metadata scope after explicit operator approval.

## Live scope established on 26 July 2026

The read-only plan reported:

- 311 new metadata insert candidates
- 79 mounted metadata backfill candidates
- 2 existing mounted rows missing stable IDs
- 3 same-family relocation candidates
- 20 cross-family reclassification candidates
- 668 untouched stale current-family rows
- 114 untouched legacy/unmounted rows

The Phase 8.9e executor narrows this to:

- **308 metadata-only inserts**
- **79 metadata backfills**
- **2 existing stable-ID assignments**

It explicitly excludes:

- 3 FLX files, because they require the real Flux scanner rather than metadata-only insertion
- all 23 relocation candidates
- all stale rows
- all legacy/unmounted rows

## Approved insert families

Metadata-only insertion is allowed only for:

- F2K
- ILL
- LTX
- PNY
- SDX
- W22
- ZIM

FLX and FLK are structurally excluded from the metadata insertion path.

## Approved backfill fields

The live plan contains only:

- 32 `base_model_code` corrections
  - 2 F2K
  - 30 ZIM
- 47 `category_name` normalisations for W21

The executor rejects any plan containing a backfill field outside:

- `base_model_code`
- `category_name`

Every update uses a compare-and-swap guard. The row ID, file path and planned old value must still match the database before the new value is written.

## Relocation policy

No relocation is applied in Phase 8.9e.

The three same-family candidates also cross category boundaries:

- Z-Image Action to Body, currently `UNK-ACT-064`
- two WAN2.2 Action to Body rows, currently `W22-ACT-214` and `W22-ACT-215`

Preserving these IDs would retain misleading family/category prefixes. Reassigning the IDs would break stable-ID continuity. They therefore remain a separate manual policy decision alongside the 20 WAN2.2 to WAN2.1 candidates.

## Safety controls

Apply mode requires all of the following:

1. A Phase 8.9d plan with no unresolved relocation, stable-ID exhaustion or existing-ID issues.
2. The exact SHA-256 digest of the canonical plan JSON.
3. An explicit `--apply` flag.
4. An explicit backup directory.
5. A successful SQLite backup before the write transaction begins.
6. The expected `lora` table schema, without creating or changing columns.
7. A `BEGIN IMMEDIATE` transaction.
8. Compare-and-swap validation for every backfill and existing ID assignment.
9. Stable-ID collision checks for every inserted or assigned ID.
10. A rollback on any mismatch or error.

The executor never:

- invokes `lora_indexer`
- opens safetensors tensors
- inserts FLX/FLK records through the metadata path
- changes scanner or orchestration maths
- updates relocation rows
- deletes stale or legacy rows
- modifies the database schema

New metadata rows are inserted with:

- no block layout
- no block-weight rows
- `has_block_weights = 0`
- `clip_tensor_count = -1` to preserve the existing unknown/backfill sentinel
- database file paths rooted at `/loras`

## Dry-run on Nibbler

After Phase 8.9e is reviewed and merged, dry-run only:

```bash
cd /home/clink/docker/lora_builder/app/LoRA_weights_builder/Database/backend

python3 phase89e_metadata_reconcile.py \
  --plan /home/clink/docker/lora_builder/data/phase89d_index_plan.json \
  --root /home/clink/docker/lora_builder/mounts/loras \
  --db /home/clink/docker/lora_builder/data/lora_master.db \
  --db-path-root /loras
```

This prints the plan digest and filtered execution counts. It makes no changes.

## Apply command

The apply command is deliberately documented but must not be run until the dry-run digest and counts are reviewed and explicit approval is given:

```bash
python3 phase89e_metadata_reconcile.py \
  --plan /home/clink/docker/lora_builder/data/phase89d_index_plan.json \
  --root /home/clink/docker/lora_builder/mounts/loras \
  --db /home/clink/docker/lora_builder/data/lora_master.db \
  --db-path-root /loras \
  --backup-dir /home/clink/docker/lora_builder/data/backups \
  --expected-plan-sha256 <REVIEWED_DIGEST> \
  --apply
```

## Validation on Bender

From the repository root:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89e_metadata_reconcile.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89e_metadata_reconcile.py
```
