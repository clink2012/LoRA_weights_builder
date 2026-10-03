# Additional-family loader investigation — 3 October 2026

This continues the required LTX-2.3, LTX-2.5 and MiniMax H3 scope alongside FLUX.1 development. These are candidate interfaces, not enabled exports. No node was installed and no ComfyUI or model file was modified.

## MiniMax H3 candidate

The upstream [Realtime LoRA selective loader](https://github.com/shootthesound/comfyUI-Realtime-Lora/blob/47d5962a651e61a39afcf06c7cf26614454c5fe7/selective_lora_loader.py), pinned to `47d5962a651e61a39afcf06c7cf26614454c5fe7`, exposes 50 main H3 blocks and a separate other-weights multiplier. Its positional parser accepts 50 or 51 numbers; an explicit 51-value output would remove ambiguity about the other weights. This ordering differs from Inspire BASE-first output.

The inspected H3 path scales one LoRA output factor, preserving signed linear strength for supported factor pairs, then calls Comfy's loader. Token-refiner and non-main tensors use the other-weights control. Its returned weights string rounds to two decimals, so that return value must not become this app's saved exact result. Invalid input can fall back to other controls; validate strictly before export. Saving a rewritten adapter is optional and outside this app's contract.

Next: characterise parser, block extraction, factor handling, adapter exceptions and Comfy target mapping independently. The required CLIP input and shared model/CLIP overall strength also need testing against actual H3 workflows. This is a promising existing-node route, not evidence that all local H3 files work.

## LTX candidate and remaining gap

[PlagueKind's loader](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/blob/d58d006a4ea32c25c06499f2ff104f0852a045a6/ltx_lora_loader/ltx_lora_loader.py), pinned to `d58d006a4ea32c25c06499f2ff104f0852a045a6`, offers per-LoRA modality strengths. The LTX path divides recognised transformer keys into audio and video groups. Its inspected interface does not expose arbitrary individual transformer block multipliers. Modality controls therefore do not satisfy this app's block-vector requirement by themselves.

Its H3 handling treats the main packed transformer as joint processing; modality controls affect distinguishable input/output parts. This reinforces the need to keep modality and block dimensions separate in the comparison model.

Continue searching for a proven LTX per-block input route. If none fits, implement a small companion-node adapter in this repository, test the exact effective patch multipliers and preserve unsupported/control-adapter exclusions. Installation into the read-only ComfyUI tree remains an owner involvement point.

## Local evidence and expansion rules

The installed `ComfyUI-H3-Multishot/h3_lora_stack.py` applies one overall strength per LoRA through Comfy's loader. It does not provide the missing per-block interface. The local LTXVideo installation contains separate 2.3 and 2.5 examples, which should inform model-specific acceptance fixtures without modifying them.

For every new family, record separately: detected architecture and version; creative/distillation/control purpose; adapter type and covered target modules; modality and block layout; verified loader contract; and render evaluation. Catalogue recognition alone enables none of the later capabilities. Preserve complete numeric vectors and immutable history across all targets.

Pinned research source copies are in ignored `.local/family-research`. They are reference material only and are not imported by the application.
