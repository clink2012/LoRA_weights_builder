# Inspire FLUX.1 export contract

The R1 adapter separates an analysis vector from a pasteable loader vector. A per-LoRA `loader_export` object is the authority for Copy. `status: ready` means the supplied resolved patch coverage can be represented by the pinned loader; it does **not** mean image quality was validated. The existing `block_weights` analysis array is retained. The misleading per-node `block_weights_csv` field is now null; its historical content is available as `analysis_block_weights_csv` for analysis only. The legacy combined summary remains analysis, never a multi-LoRA loader instruction.

## Pinned behavior

- Target: `LoraLoaderBlockWeight //Inspire`, installed Inspire 1.23 source SHA-256 `4e6188d8d20e2a13c482d2d7fae6ddb570210c426265605b7760b5b96ebb914e`.
- Canonical FLUX.1 order: BASE, DOUBLE 0–18, SINGLE 0–37 (58 values).
- The loader walks **present resolved patch groups**, sorted numerically within double and single groups. Missing blocks do not consume a slot. A full architecture vector pasted into a sparse LoRA can therefore address the wrong blocks.
- The loader retains the last numeric index across group boundaries. DOUBLE 3 followed by SINGLE 3 shares a slot. Conflicting desired values are blocked by this adapter; equal values share an explicitly labelled slot.
- Repeated patches in one block share a numeric slot. Comfy tuple patch keys are classified by their first string member.
- The installed validator requires at least 12 fields. Sparse exports include explicitly labelled trailing unused padding. It also rejects exponent notation, so numeric output uses exact decimal text without one-decimal UI rounding.
- Unknown groups/indices, unconfirmed coverage, changed loader versions and non-FLUX.1 layouts are blocked. No file is loaded or modified by the pure adapter.

`architecture_slot_labels` / `architecture_slot_values` describe the complete model block space. `slot_labels` / `slot_values` and `numeric_csv` describe the actual loader input, including ignored padding where necessary. These are deliberately separate. BASE defaults to 1.0 in the current API path and must be surfaced as a distinct control when export is enabled.

## Evidence

`Database/backend/tests/fixtures/inspire_flux1_contract.json` contains observed outputs captured by executing selected **actual installed** parser and assignment methods against synthetic resolved patch dictionaries. The adjacent capture utility extracts methods using Python AST without importing Comfy/Torch or running module initialization. It records source identity and scope. No upstream GPL source is vendored. Fixtures cover full coverage, sparse misalignment, sparse correct input, boundary collision, repeated patches, short-vector repetition, unknown transformer groups and rejected exponent notation.

This proves numeric assignment semantics only. Tensor shape compatibility, actual Comfy patch acceptance and image quality still require their own evidence.

## Next coverage package

Current stored energy rows do not prove which patches Comfy will load. The legacy combine route therefore remains blocked. A separate header resolver and `/api/lora/prepare-blocks` route now provide a narrow conditional export path. Do not treat nonzero energy positions, a `flux_transformer_57` label, or a raw key-name match as complete proof.

A dependency-light resolver can inspect safetensors headers without reading tensor payloads, then map a narrowly supported key/adapter format against a pinned FLUX.1 target schema. The installed `comfy/lora.py` has SHA-256 `fce6903ca8150611b3f6477c3772bfef0662a2b01747a579ff56b415a44c29bc`; its generic UNet mapping derives `lora_unet_...` aliases from **target model state-dict keys**. `load_lora` then delegates loading to adapter implementations. Both the target key/shape schema and the relevant adapter's pair/shape behavior must be pinned and independently tested. This should initially support complete native Kohya up/down pairs only, explicitly classify BASE patches, detect unmatched/unsupported tensors, and reject uncertain cases. A header-only resolver must not silently claim full runtime patch application or compatibility with every FLUX variant.

Persist coverage evidence with source file identity (size, modification time and header fingerprint), target architecture/key-schema identity, resolver version, Comfy/adapter source identity and unmatched-key diagnostics. Invalidate it on change. Keep expensive tensor analysis and GPU generation out of ordinary app startup. Real image comparisons remain necessary to calibrate the balancing heuristics.

### Implemented conditional target

`contracts/flux1-dev-native-v1.json` records standard FLUX.1 dev parameters, module shapes observed by symbolically executing the installed FLUX constructors, native aliases observed from the installed Comfy mapping function, and pair-selection observations from the actual LoRA adapter. The capture utility is adjacent to the loader fixture utility. Symbolic modules record dimensions without allocating tensors, loading Torch or importing Comfy startup. The native alias branch is covered; unrelated adapter formats are not claimed.

`flux_header_coverage.py` reads at most 16 MiB of safetensors JSON plus its length word. It validates complete up/down pairs, optional one-value alpha shape, known target modules, pair ranks/dimensions, floating dtypes and contiguous tensor offsets matching the file size. Unknown tensors and duplicate target aliases block export. It verifies the pinned local Comfy/Inspire source hashes and detects a changing file or atomic path replacement during reading. It never reads tensor payloads, so numerical tensor contents, alpha values and generated image quality remain unverified.

The Studio route requires `target_contract_id: "flux1-dev-native-v1"`. It reads current FLX catalogue file paths and headers, bypassing stale cached layouts and energies. Proven present blocks start at 1; absent blocks start at 0; BASE starts at 1. This is labelled `structural_baseline_unvalidated`, not a balancing recommendation. `coverage_source` is `statically_resolved_against_pinned_target`; `checkpoint_verified` and `image_quality_verified` are false. `compatible` is false if any requested LoRA is missing or unsupported, while individual successful nodes remain visible. Source files and the catalogue are read-only on this path.

Three current local FLUX LoRAs were inspected as a bounded smoke check: each mapped 304 native pairs covering all 57 transformer blocks, reading approximately 159–168 KB of header from each approximately 19.3 MB file. This is conditional structural evidence only, not a model inference test.

## Isolated runtime

Ordinary catalogue, preparation and profile requests run without Torch. Studio measurements use the separate optional CPU worker; legacy `/inspect` remains a separate historical inspection path. `LORA_DB_PATH` selects an isolated database for the API. The native API includes the model-family registry route once; the Docker wrapper reuses it. The pending lightweight [catalogue-refresh package](library-refresh.md) retires global `/reindex_all` with HTTP 410 and uses path/stat inventory without invoking the old tensor indexer or ID assigner.
