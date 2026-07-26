# Phase 8.9b indexer registry integration

This slice connects the approved Phase 8.9a model-family registry to the legacy indexer through a narrow, explicit integration seam.

## What changes

- `lora_path_parser.py` preserves the existing folder/category parsing contract while sourcing model-family codes from the canonical registry.
- `model_family_integration.py` applies only the model mapping and parser function to the legacy `lora_indexer` module.
- `lora_api_server_docker.py` applies that integration before importing the FastAPI application.
- `lora_indexer_registered.py` provides an explicit registry-aware CLI entrypoint for future controlled local runs.

## Newly recognised metadata families

| Folder | Code |
|---|---:|
| Flux.2-Klein | F2K |
| LTXV2 | LTX |
| Z-Image | ZIM |

These families remain metadata-only. The integration does not add a block layout, block extraction, role-aware orchestration, or export support for them.

## Compatibility preservation

Existing DB-facing family names remain unchanged:

- `SD`
- `WAN2.1`
- `WAN2.2`

The registry's presentation labels, such as `WAN 2.1`, are not written into index metadata.

WAN mode-folder parsing remains unchanged for T2V, I2V, V2V, T2I, I2I, IMG2VID and IMAGE2VIDEO.

## Safety

This branch does not:

- run a full or partial scan
- modify the current DB
- change schema
- assign stable IDs
- delete stale rows
- change scanner selection
- change block extraction or orchestration maths

The Docker backend will only use the new mapping if a future indexing action is explicitly invoked. Merging or deploying this code does not itself start an index.

## Validation

From **Bender / VS Code PowerShell**:

```powershell
cd 'E:\LoRA Project'

& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_model_family_registry.py `
  Database\backend\tests\test_lora_path_parser.py `
  -q
```
