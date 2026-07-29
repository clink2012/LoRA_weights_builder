# Phase 8.9k Flux 2 layout support

Phase 8.9k adds a distinct read-only layout for the PEFT-style target `FLX-STL-263` after the Phase 8.9j live analysis proved that it is not a standard Flux 1 architecture.

## Live evidence carried forward

Phase 8.9j established the following immutable inputs:

- stable ID: `FLX-STL-263`
- source: `FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors`
- source SHA-256: `c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7`
- Phase 8.9j analysis SHA-256: `7c886a07e87fa36081645d34bd578001420e65a380c6543b17c9b9ee1fb8dc48`
- tensor keys: 276
- LoRA rank: 16
- double blocks: `0..7`
- single blocks: `0..47`
- adapted block modules: 128
- block tensors: 256
- global projection modules: 10
- global projection tensors: 20
- unmatched tensors: 0

The Phase 8.9j report remains blocked only because the earlier analyser intentionally tested the source against the Flux 1 ranges of 19 double blocks and 38 single blocks.

## New layout

Phase 8.9k registers:

```text
flux2_transformer_56
```

The ordered layout is:

```text
DOUBLE_0 .. DOUBLE_7
SINGLE_0 .. SINGLE_47
```

This is 56 logical block rows. It is separate from the existing `flux_unet_57` layout and does not alter existing Flux 1 block mathematics.

## Sealing behaviour

`phase89k_flux2_layout_support.py`:

1. loads the exact Phase 8.9j JSON report
2. verifies its canonical analysis digest
3. requires the Phase 8.9j observations to match the live 8-double and 48-single architecture
4. requires the only Phase 8.9j blockers to be the old single-block range checks for indices `38..47`
5. verifies the source SHA-256 is unchanged
6. opens SQLite with URI `mode=ro` and `PRAGMA query_only = ON`
7. confirms the file path and stable ID remain absent
8. reopens only the approved source file
9. validates every LoRA A/B pair, tensor shape and rank
10. requires all 8 double blocks and all 48 single blocks
11. requires the exact 128 block modules, 256 block tensors, 10 global modules and 20 global tensors
12. computes and seals all 56 block weights and raw strengths
13. emits a canonical artifact SHA-256

## Safety boundary

Phase 8.9k:

- writes no database rows
- creates no backup
- has no apply mode
- does not run the full indexer
- does not enumerate the library
- does not change `delta_inspector_engine.py`
- does not touch `FLX-BDY-071`
- does not modify existing `flux_unet_57` results

A later database write requires a separate guarded phase and fresh explicit approval.

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89k_flux2_layout_support.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89k_flux2_layout_support.py `
  Database\backend\block_layouts.py
```

Expected: 7 tests passed and silent compilation.

## Planned Nibbler run

After review and merge, run inside a temporary backend container:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89k_flux2_layout_support.py \
  --analysis /data/phase89j_flx_stl_263_analysis.json \
  --root /loras \
  --db /data/lora_master.db \
  --expected-analysis-sha256 7c886a07e87fa36081645d34bd578001420e65a380c6543b17c9b9ee1fb8dc48 \
  --expected-stable-id FLX-STL-263 \
  --expected-source-sha256 c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7 \
  --json /data/phase89k_flx_stl_263_flux2_artifact.json
```

The database SHA-256 and backup count must remain unchanged.
