# Phase 8.9j targeted PEFT Flux analysis

Phase 8.9j adds a read-only analyser for the single remaining valid FLX candidate that Phase 8.9g could not interpret:

- path: `FLUX/02 - Styles/aidmaMJ61Flux.2v0.5.safetensors`
- planned stable ID: `FLX-STL-263`
- source SHA-256: `c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7`
- observed namespace: `base_model.model.double_blocks.*` and `base_model.model.single_blocks.*`

The damaged `FLX-BDY-071` source is explicitly excluded.

## Purpose

The existing Delta Inspector recognises Flux keys such as:

- `lora_unet_double_blocks_<idx>_*`
- `lora_unet_single_blocks_<idx>_*`
- `transformer.single_transformer_blocks.<idx>.*`

`FLX-STL-263` instead uses paired PEFT tensors:

- `base_model.model.double_blocks.<idx>.<module>.lora_A.weight`
- `base_model.model.double_blocks.<idx>.<module>.lora_B.weight`
- `base_model.model.single_blocks.<idx>.<module>.lora_A.weight`
- `base_model.model.single_blocks.<idx>.<module>.lora_B.weight`

Phase 8.9j analyses only that target and does not change the general indexer or database.

## Analysis rules

The analyser:

- requires the exact Phase 8.9g blocked target and stable ID
- requires the exact source SHA-256
- opens SQLite with URI `mode=ro` and `PRAGMA query_only = ON`
- confirms the target path and stable ID remain absent
- requires the tensor-key count to match Phase 8.9g diagnostics
- validates every LoRA A/B pair
- validates two-dimensional tensor shapes and matching ranks
- accepts double-block indices only in `0..18`
- accepts single-block indices only in `0..37`
- maps missing, unadapted blocks to zero strength
- computes ordered `DOUBLE_0..18 + SINGLE_0..37` strengths
- emits layout `flux_unet_57`
- records known global projection tensors separately
- blocks unknown `base_model.model.*` LoRA namespaces
- records a canonical analysis SHA-256

Known global roots are limited to:

- `guidance_in`
- `time_in`
- `double_stream_modulation_img`
- `double_stream_modulation_txt`
- `final_layer`
- `img_in`
- `single_stream_modulation`
- `txt_in`

These tensors are validated but excluded from per-block strengths.

## Safety boundary

- no database writes
- no backup creation
- no full indexer invocation
- no library enumeration
- only the exact diagnostics target is opened
- no stable-ID assignment
- no relocation, stale-row or legacy-row work
- no access to the damaged FLX file

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89j_targeted_peft_flux_analysis.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89j_targeted_peft_flux_analysis.py
```

Expected: 6 tests passed and silent compilation.

## Planned Nibbler run

After review and merge, run inside a temporary backend container:

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89j_targeted_peft_flux_analysis.py \
  --diagnostics /data/phase89g_flux_diagnostics.json \
  --root /loras \
  --db /data/lora_master.db \
  --expected-stable-id FLX-STL-263 \
  --expected-source-sha256 c60c9a5de39da23b3b4f4dca48e3511faa1fe5a4987d4acbb0a04643a9a65be7 \
  --json /data/phase89j_flx_stl_263_analysis.json
```

The database SHA-256 must be identical before and after the run.
