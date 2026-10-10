# Native H3 patch characterization and prepared loader trial

This follows the owner's selected DaSiWa Ref2VA Hybrid curve checkpoint. The installed Basic loader cannot consume individual block values. H3 proposal/export capabilities remain disabled pending native quantized runtime evidence; this package does not imply completion of H3 support.

## Native ordinary-pair evidence

`capture_h3_native_patches.py` verifies four source hashes before extracting selected native methods. It invokes ordinary LoRA loading/conversion, `ModelPatcher.add_patches` and `LoRAAdapter.calculate_weight` on small synthetic CPU matrices. It imports no Comfy module startup and opens no owner checkpoint/adapter. Its fixture contains 45 cases across ordinary A/B, up/down and A/B-default naming; absent, positive and negative alpha; positive, negative and zero outer/block strengths.

For those ordinary pairs, effective update is `outer strength × block multiplier × (alpha/rank) × B@A`. With absent alpha the native factor is 1, not automatically `1/rank`. Registration appends each adapter's independent strength/payload to an existing target rather than averaging strengths or changing factors/alpha. Unknown target keys are rejected. The independent tests recompute the observed matrix results using simple arithmetic.

An important negative observation: native registration checks target-key presence, not low-rank dimensions. Native calculation catches a dimension mismatch, logs an error and returns unchanged weights. Therefore a count of registered/accepted patches cannot alone establish correct application. The complete target-shape preflight remains necessary. Native clone behavior, quantized-base calculation and visual acceptance are not proved by these synthetic matrices.

Source hashes beyond the previous alias capture are:

| Source | SHA256 |
| --- | --- |
| `comfy/weight_adapter/lora.py` | `5e30c8b8a22be6459883cb5d758fa76f725cc4688d4048200145b885567c4cec` |
| `comfy/lora_convert.py` | `6c199c3404828838b5745aafc1b0759566a45287f5c3e9cebd31f248cc179245` |
| `comfy/model_patcher.py` | `fc43f038835aeded2e64a0490f9d37795fb9d3c9490d9210ba11f1201aec4fd8` |

The capture still pins `comfy/lora.py` to the native target contract's existing hash. No upstream implementation is vendored in tracked source.

## Prepared installation scope

`tools/prepare_h3_loader_trial.py` stages only the already-characterized [maintainer block loader](https://github.com/filliptm/ComfyUI-FL-MiniMaxH3/blob/4e6379770e7f4fc48651b76d03284a7eee7df8a4/nodes/FL_MiniMaxH3LoraBlockLoader.py), a registration-only wrapper, upstream Apache-2.0 LICENSE/NOTICE and a hash manifest under `.local/h3-loader-trial/lora_builder_h3_loader_trial`. It verifies all upstream input hashes and refuses divergent existing staging content. It performs no download, upstream execution, installation, dependency change, model access or ComfyUI write. Preparation tests cover changed/missing/unbounded inputs, preservation of altered/extra files and exact/idempotent output.

The staged block-loader SHA256 is `3d57625d2fb907bbbb8902d293287df7db5d721b4032a544eaa1a971a82f29d8`; the upstream revision is `4e6379770e7f4fc48651b76d03284a7eee7df8a4`. The loader receives a native model plus a single LoRA, overall/main/refiner/OTHER strengths and multiline per-index overrides. It supplies 50 main and two token-refiner positions from the connected model; OTHER remains a separate widget. No changes to DaSiWa's loader or existing workflow are proposed for installation.

After owner installation authority, the bounded change is addition of this one new folder under `E:\ComfyUI_New\ComfyUI\custom_nodes\lora_builder_h3_loader_trial`, only if that folder is absent. No existing ComfyUI file is overwritten and no dependencies are installed/upgraded. The node needs ComfyUI's current native H3 and node APIs. Its actual registration is unverified until ComfyUI restarts while idle. An existing node with the same ID must block addition rather than be overwritten.

The [intent contract](intent-contract.md) states: **“ComfyUI and model trees remain read-only. A required companion node may be developed here; installation is a separate owner involvement point.”** This is why the prepared addition needs a specific owner decision. Source-only GitHub/Farnsworth CI can verify preparation and synthetic contracts; it cannot verify an uninstalled loader against Bender's quantized checkpoint.

After registration verification, prepare a separate neutral trial copy of the selected workflow, retaining the original and required distillation/control adapters. Compare ordinary model-only loading with all block multipliers at 1, then test a bounded single-block change. Pin checkpoint/header, loader/core source, LoRAs, strengths, seed/prompt/sampler and output stage. Check logged errors as well as patch counts. Generation timing and visual judgement need owner coordination. App copy/proposal support stays disabled until the relevant runtime and result ownership checks pass.
