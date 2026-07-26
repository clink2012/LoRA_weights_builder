# Phase 8.9g targeted Flux diagnostics

This companion diagnostic runner addresses one operational limitation found during the live Phase 8.9g analysis: an unsupported tensor structure previously stopped the whole three-file process.

The diagnostic runner keeps the same read-only scope and path/ID guards, but captures per-file tensor or analysis errors and continues to the remaining plan-listed candidates.

## Live issue observed

The first production run reached the Flux analysis engine and reported that one target contained none of the currently recognised structures:

- `transformer.single_transformer_blocks.<idx>.*`
- `lora_unet_double_blocks_<idx>_*`
- Flux text-encoder-only keys
- UNet-57 keys

The SQLite SHA-256 remained unchanged.

## Diagnostic output

For every one of the three targets, the report records:

- source filename, size, mtime and SHA-256
- planned stable ID and database path
- tensor-key count
- the first 25 sorted tensor keys
- the 20 most common key prefixes
- CLIP contribution metadata
- Flux analysis result, when recognised
- exception type and message, when not recognised
- warnings and controlled-apply readiness

An unsupported structure is reported as a blocked candidate rather than terminating the report.

## Safety boundary

The runner:

- opens SQLite with URI `mode=ro`
- enables `PRAGMA query_only = ON`
- opens only the three paths selected from the reviewed Phase 8.9d plan
- does not enumerate the LoRA library
- does not call `lora_indexer.main()`
- does not assign IDs
- does not insert or update rows
- does not alter block weights
- has no apply mode

## Validation

From Bender:

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89g_targeted_flux_diagnostics.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89g_targeted_flux_diagnostics.py
```

The tests verify that analysis and tensor-inspection failures are isolated per file, all three targets remain represented, and the database bytes remain unchanged.
