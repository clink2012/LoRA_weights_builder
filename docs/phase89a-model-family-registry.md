# Phase 8.9a model-family registry

This slice establishes a canonical, explicit registry for the LoRA model families known to the project.

It does not run an index, modify the database, assign stable IDs, inspect safetensors files, or change scanner mathematics.

## Registered families and codes

| Folder | Code | Display name | Declared capability |
|---|---:|---|---|
| FLUX | FLX | Flux | Mixed scanned / fallback |
| Flux Krea | FLK | Flux Krea | Mixed scanned / fallback |
| Flux.2-Klein | F2K | Flux.2-Klein | Metadata only |
| Illustrious | ILL | Illustrious | Metadata only |
| LTXV2 | LTX | LTXV2 | Metadata only |
| PONY | PNY | Pony | Metadata only |
| SD | SD1 | SD 1.x | Metadata only / legacy mapped family |
| SDXL | SDX | SDXL | Metadata only |
| WAN2.1 | W21 | WAN 2.1 | Metadata only |
| WAN2.2 | W22 | WAN 2.2 | Metadata only |
| Z-Image | ZIM | Z-Image | Metadata only |

All model codes remain three characters to preserve the existing stable-ID format.

## Capability policy

Only `FLX` and `FLK` currently claim:

- block analysis
- fallback layout support
- architecture-specific key normalisation
- role-aware orchestration for scanned rows
- ComfyUI per-LoRA block payload export

Every other family is explicitly declared metadata-only until scanner, layout, export, and architecture-specific tests exist.

## API

The Nibbler Docker application exposes:

```text
GET /api/model-families
```

The response includes the registry schema version and explicit capabilities for each family. This endpoint is intended to become the UI dropdown source in a later Phase 8.9 slice.

## Safety

This branch deliberately does not:

- import the registry into `lora_indexer.py`
- update existing DB rows
- assign new stable IDs
- add block layouts
- run a full or partial rescan
- change block extraction or orchestration maths

The next integration step should replace duplicated family maps in the indexer/catalog and then update the UI to consume the API endpoint. Those changes should be reviewed before any controlled indexing operation.
