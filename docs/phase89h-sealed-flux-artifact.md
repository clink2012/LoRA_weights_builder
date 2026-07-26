# Phase 8.9h sealed Flux artifact

Phase 8.9h creates a read-only, canonical JSON artifact for the single Phase 8.9g candidate that passed live diagnostics: `FLX-PPL-207`.

It does not write to SQLite and has no apply mode.

## Purpose

The live diagnostics established that one of the three excluded FLX candidates is fully analysable:

- path: `FLUX/01 - People/ang3l4wh1t3-f1.safetensors`
- planned stable ID: `FLX-PPL-207`
- source SHA-256: `9f06e750055f524998cca621e25fca3672a0e6dbf836b8d63c6146348c04cf9d`
- layout: `flux_unet_57`
- block weights: 57
- raw strengths: 57

The other two candidates remain blocked and are not included in this artifact.

## Guard conditions

The generator requires:

- the Phase 8.9d plan digest to match the diagnostics report
- diagnostics phase `8.9g-diagnostics`
- exactly one target marked ready for controlled apply
- stable ID exactly `FLX-PPL-207`
- source SHA unchanged since diagnostics
- current database integrity check `ok`
- target file path still absent from SQLite
- target stable ID still unused
- tensor-key count and CLIP metadata unchanged
- model family, LoRA type and rank unchanged
- exactly 57 block weights and 57 raw strengths
- layout exactly `flux_unet_57`

SQLite is opened with URI `mode=ro` and `PRAGMA query_only = ON`.

## Artifact contents

The output JSON contains:

- plan and diagnostics SHA-256 digests
- source path, size, mtime and SHA-256
- planned family/category metadata and stable ID
- tensor and CLIP metadata
- model family, LoRA type and rank
- all 57 normalised block weights
- all 57 raw strengths
- a canonical artifact SHA-256

A later write phase must require the exact artifact digest before it can insert anything.

## Bender validation

```powershell
& 'C:\Users\clink\miniconda3\python.exe' -m pytest `
  Database\backend\tests\test_phase89h_sealed_flux_artifact.py `
  -q

& 'C:\Users\clink\miniconda3\python.exe' -m py_compile `
  Database\backend\phase89h_sealed_flux_artifact.py
```

## Planned Nibbler run

The generator should run inside a temporary backend container so that `safetensors`, PyTorch and the existing analysis engine are available without modifying Nibbler's host Python environment.

```bash
sudo docker compose --env-file .env run --rm --no-deps -T backend \
  python phase89h_sealed_flux_artifact.py \
  --plan /data/phase89d_index_plan.json \
  --diagnostics /data/phase89g_flux_diagnostics.json \
  --root /loras \
  --db /data/lora_master.db \
  --expected-stable-id FLX-PPL-207 \
  --expected-block-count 57 \
  --expected-layout flux_unet_57 \
  --json /data/phase89h_flx_ppl_207_artifact.json
```

The database SHA-256 must be identical before and after the run.
