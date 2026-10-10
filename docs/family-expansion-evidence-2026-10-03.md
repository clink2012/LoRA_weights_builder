# Family expansion evidence — 3 October 2026

> Current owner scope (10 October 2026): [active library and graph drawing](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

This is a bounded research and local source inspection, not an implementation or generation acceptance report. ComfyUI and model folders were read only. No installation, checkpoint loading, tensor calculation or inference occurred. The app still has only the pinned FLUX.1 export/measurement path; these families remain metadata-only there.

## Confirmed identities

| Requested family | Primary-source identity | Implication for this app |
| --- | --- | --- |
| LTX-2.3 | Lightricks audio-video diffusion transformer; official 22B dev and distilled checkpoints, plus separate distillation LoRAs. | Distinguish version, checkpoint variant, creative adapters, distillation adapters and conditioning/control adapters. |
| LTX-2.5 | Lightricks synchronized audio-video model; separate transformer, encoder, decoder and related components. Official dev and distilled variants exist. | Treat as its own target contract even where tensor shapes overlap with 2.3. |
| MiniMax H3 | Official H3-Base has FL2VA and Ref2VA task variants generating video and audio. The full official 2K workflow includes hosted components beyond the locally released base model. | Support local H3-Base contracts; do not imply that this local app reproduces the hosted end-to-end system. |

These are verified current model names, not speculative roadmap labels. Sources: [LTX-2.3 official model card](https://huggingface.co/Lightricks/LTX-2.3), [LTX-2.5 official model card](https://huggingface.co/Lightricks/LTX-2.5), [MiniMax H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3). The official MiniMax card differentiates FL2VA first/last-frame generation from Ref2VA multimodal reference generation. Structural compatibility alone does not establish that an adapter was trained for both.

For video families, useful comparison outcomes include identity, clothing, movement, temporal stability and audio preservation. Existing still-image assumptions and block-role labels cannot simply be transferred to these models.

## Local evidence

Filename enumeration found 95 safetensors under `E:\models\loras\LTXV2`, 3 under `LTXV2_5`, and 73 under `MiniMax-H3`. These counts include every safetensors file in each folder, without claiming that all are valid adapters for that family.

Eight adapter headers and three checkpoint headers were sampled. Adapter reads were capped at 2 MiB per header and checkpoint reads at 8 MiB; all selected headers fit. Only the 8-byte header length and JSON header were read. Research header hashes below cover the JSON header bytes alone, **not the length prefix or full file**, and do not verify tensor contents. The production reader's existing `header_sha256` convention instead covers the length prefix plus JSON bytes; these two hash conventions must not be compared directly.

| Sample | Header evidence |
| --- | --- |
| `checkpoints/LTXV2/ltx-2.3-22b-dev-fp8.safetensors` | 48 `transformer_blocks`, indices 0–47; metadata `model_version=2.3.rc1`; video heads 32 × 128, audio heads 32 × 64. Header 1,441,504 bytes, SHA256 `e21e55fbbc308ef8f476041fc298af4f3b22d65811f079f12c63902972be6511`. |
| `diffusion_models/LTX2_5/ltx25FP8Comfyui_v10.safetensors` | 48 `transformer_blocks`, indices 0–47; metadata `model_version=2.5.0`; same sampled head dimensions. Header 679,336 bytes, SHA256 `0ef80965da21a9624538cd4a94f710267639ed7d8196f29bcd04a7ccab137316`. |
| `diffusion_models/H3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors` | 50 main `blocks`, indices 0–49, and 2 `token_refiner.blocks`, indices 0–1; `adaln_t_table` present. Filename indicates Ref2VA/pruned/quantized; header topology alone does not authenticate publisher or task provenance. Header 95,416 bytes, SHA256 `c1a65ff59085cbed8a6ea7c935ae7e58b1d2f870239df54c58a68fb7989e1cd8`. |

Paths in that table are relative to `E:\models`. Additional local filenames show LTX-2.3/2.5 encoders and audio/video VAEs, and both FL2VA- and Ref2VA-labelled H3 variants. Their existence does not prove an operational workflow.

The sampled LTX adapters use `diffusion_model.transformer_blocks.N.*.lora_A/B.weight`. Samples span video attention only and combined video/audio/cross-stream modules. The 2.5 distillation adapter has 3,320 tensor entries, including non-block timestep projections. An adapter's maximum index is not a reliable total model-block count when the adapter is sparse.

The sampled H3 adapters include both native dotted A/B keys and Kohya-style `lora_unet_blocks_N_*` up/down/alpha keys. Main block indices reach 49. Mapping must separately handle token-refiner and other targets rather than dropping them into an undocumented scalar.

**Concrete folder mismatch:** the sampled `happy.safetensors` inside the broad `LTXV2` folder has FLUX double/single-block keys and `ss_base_model_version=flux2_klein_9b`. Its header SHA256 is `bd9c4e3dae6a29b60d978a6d8c6d4a1c0d4aed2495f1d7d8e14129715964c653`. No file was moved or renamed. This is direct evidence that catalogue folder hints must not choose an architecture contract or authorize export.

## Installed source and available loader routes

Installed ComfyUI checkout: `daeb5e53681e2b10a3f0727d9ec5bc90784bee10`. Installed LTXVideo checkout: `3bf3ca62595f1764c47d01c35c8e5dfe47e1a88f`. These are source identities, not proof that every node successfully imports in the running ComfyUI process.

Relevant local files under `E:\ComfyUI_New\ComfyUI`:

- `comfy/model_detection.py:390` detects H3 main/refiner counts and its pruned AdaLN curve form; `:420` detects LTX video versus audio-video and reads model config.
- `comfy/lora.py:383` adds native LTX aliases; `:389` adds native H3 aliases. Generic dotted and Kohya aliases are also defined earlier in that file.
- `comfy/ldm/lightricks/av_model.py:85` separates video, audio and cross-stream attention; `:392` defines the audio-video model.
- `comfy/ldm/minimax/model.py:474` defines the native H3 architecture, with main transformer and token-refiner stacks.

### Inspire: insufficient for these block groups

The inspected `custom_nodes/comfyui-inspire-pack/inspire/lora_block_weight.py:370` groups input/middle/output/double/single blocks. LTX `transformer_blocks` and H3 `blocks`/refiner keys fall through to `others`, which receives BASE handling. A long comma-separated vector cannot create per-block support that this loader lacks. Source SHA256: `4e6188d8d20e2a13c482d2d7fae6ddb570210c426265605b7760b5b96ebb914e`.

### LTX: an installed candidate already exists

KJNodes registers `LTX2BlockLoraSelect`, `DiTBlockLoraLoader`, and `LTX2LoraLoaderAdvanced`. The selector supplies 48 individually configurable entries. The advanced loader also exposes video, audio, video-to-audio, audio-to-video and other controls. Local source agrees with the project's [selector/loader source](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/nodes.py) and [advanced LTX loader source](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/ltxv_nodes.py).

Important local implementation details:

- `nodes/nodes.py:1918` produces `blocks.N.` selectors. Matching is by substring, so these also match `transformer_blocks.N.`. The controls default to zero, not a neutral all-ones vector, and the UI disallows negative block alpha.
- `nodes/nodes.py:1941` and `nodes/ltxv_nodes.py:2201` replace adapter `weights[2]` with the selected number. That field is **alpha**. Comfy's `comfy/weight_adapter/lora.py:249` applies alpha divided by rank; absent alpha otherwise means unit scale.
- For standard additive LoRA, a desired multiplier `w` requires replacement alpha `w × original_alpha`, or `w × rank` when original alpha is absent. A shared per-block value cannot represent this correctly when targets in that block require different replacement alphas (for example, differing original alphas, or differing ranks with absent alpha). Differing ranks alone do not cause this problem when all original alphas are the same non-null value. This formula must not be extended to DoRA without separate characterization. The adapter bridge must detect and reject unrepresentable cases or use a loader that scales each patch directly.
- The advanced loader also substitutes alpha `1` when a modality multiplier changes an adapter whose alpha is absent. For rank `r`, a multiplier `s` then gives `s/r`, rather than the expected `s`; at exactly 1 it leaves absent alpha untouched. This needs a dedicated regression and an explicit compatibility restriction.
- The existing inputs are individual widgets/custom sockets, not an Inspire-compatible paste box. A documented bridge or small purpose-built node would still be needed for the app's simple copy/paste experience.

Local SHA256: `nodes/nodes.py` = `e9e109717c12b8f0c61c3401908fa9d1f07d23c94debb031fb85dfa2b4d9331f`; `nodes/ltxv_nodes.py` = `11286583c56e653dbd57f5e8eda6902c84b759f16b359b3bf0c895b43e6b5364`.

The installed LTXVideo IC-LoRA loader exposes one overall strength and conditioning metadata. It is not itself a per-block substitute. Attention-tuning nodes likewise alter attention execution and must not be confused with adapter block multipliers.

### H3: a newer upstream candidate exists

The maintainer's [FL MiniMax H3 project](https://github.com/filliptm/ComfyUI-FL-MiniMaxH3#fl-minimax-h3-lora-block-loader) documents a LoRA Block Loader with separate main-block, token-refiner and other multipliers, plus text overrides such as `blocks.0-9=0.5` and `refiner.0-1=0.8`. It supports chained adapters and reports matched targets. An independent source review of [its loader at commit 4e63797](https://github.com/filliptm/ComfyUI-FL-MiniMaxH3/blob/4e6379770e7f4fc48651b76d03284a7eee7df8a4/nodes/FL_MiniMaxH3LoraBlockLoader.py) confirmed direct patch-strength scaling, preserving alpha, and main/refiner counts derived from the connected H3 model. It verifies acceptance of mapped nonzero patches; unmatched original adapter keys are only logged upstream, so this is not complete source-key coverage proof. It remains a candidate, not verified integration: no matching installed class was found locally, and no installation or runtime test was performed.

The [H3 PowerLoRAStack maintainer repository](https://github.com/cicalooo/ComfyUI-H3-PowerLoraStack) is a further compatibility lead for pruned/quantized H3. It needs code-level review before selection. A generic loader's successful import is insufficient evidence for AdaLN-curve, convrot or other transformed checkpoints.

## Recommended next implementation packages

1. **Header-only identification and coverage:** preserve current catalogue identities, add evidence-based architecture candidates and explicit disagreement flags. Separate LTX-2.3 from 2.5 through checkpoint contract/metadata evidence; never decide from 48 blocks alone. Model contracts should cover LTX video/audio/cross-stream/other targets and H3 main/refiner/other targets. Keep unsupported formats visible with concrete reasons.
2. **Pinned loader contracts:** characterize the installed KJNodes alpha semantics using tiny synthetic adapters, including missing alpha, differing ranks, signed values, sparse targets, non-LoRA patches and non-block tensors. Assess the H3 candidate's actual patch scaling and transformed-checkpoint compatibility before implementing an exporter. Keep complete app vectors, but serialize separately for each chosen loader's real input format.
3. **Measurement adapters:** generalize the existing low-rank update math through explicit target/key/group mappings; retain tensor-format rejection and bounded worker execution. Video/audio groups should remain distinguishable. Do not transfer FLUX role profiles or claim that parameter alignment predicts video quality.
4. **Controlled workflow trials:** retain pipeline-required distillation/control adapters as a distinct role, protected from ordinary creative balancing until characterized. Verify all-ones equivalence, one-block zeroing, signed changes, chain order and saved rollback before trying role-based recommendations. Then assess short same-seed clips, including temporal and audio behaviour.

No owner clarification is required to start identification and synthetic characterization. Before actual video trials, select a specific existing LTX-2.3, LTX-2.5 and H3 workflow/checkpoint combination; the library contains multiple variants, so there is no justified single default yet. Ask for a current workflow only if existing local workflow inspection cannot identify the owner's intended setup. ComfyUI remains read only until an explicit implementation step authorizes any new node installation or workflow change.

## Header identification foundation

`Database/backend/adapter_identity.py` now provides pure `classify_header` observations and a bounded `identify_file` wrapper. `tools/identify_lora.py` accepts one file and an explicit library root. It reuses the existing structural safetensors reader through an opt-in metadata return; the FLUX reader's default return is unchanged. Length, floating-point dtype, shape, exact offsets, gap/overlap, complete payload length and file-stat consistency checks precede identification. Unsupported tensor dtypes are reported as an inspection failure, not accepted as ordinary LoRA tensors.

The result retains observed indices with unknown total depth, distinguishes LTX-like video/audio/cross-stream keys, H3-like main/refiner keys and ambiguous FLUX double/single topology, reports unsupported/mixed pair formats, and separates selected self-declared metadata from unverified folder hints. Neither matching keys nor metadata establishes exact model version or export support. No API, GUI, loader export or measurement integration was added in this foundation.

Generic video attention and H3-like MLP/refiner names occur in other architectures. They remain explicitly ambiguous candidates. Pair completeness describes tensor-name pairs, not rank compatibility, alpha values or verified target shapes. Only ordinary floating-point tensor descriptors pass this reader; quantized/non-floating descriptors yield an explicit unavailable result rather than partial observations.

The implemented reader was exercised against three real adapters: the misplaced FLUX.2-like sample (224 tensor descriptors, folder disagreement), the LTX-2.5 distillation adapter (3,320 descriptors, 48 observed blocks) and an H3-like adapter (416 descriptors, 50 main plus two refiner blocks). Reads totalled 607,728 bytes including length prefixes; tensor payloads were not read, and every result retained unproven version and false export verification. This validates bounded inspection only, not successful patch loading or generated output.
