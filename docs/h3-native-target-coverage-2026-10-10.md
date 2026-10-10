# Conditional native H3 target coverage

This research foundation follows PR #103's visible H3 adapter inspection. It does not change the UI, register a preparation/export route or enable H3 capabilities.

## Captured target

The capture executes only selected constructors and the native alias function from the local pinned ComfyUI sources against symbolic shape objects. It imports no torch, allocates no model tensors, opens no checkpoint, runs no upstream module startup and changes no ComfyUI file. Source hashes are checked before execution. Upstream code is not vendored.

| Source | SHA256 |
| --- | --- |
| `comfy/ldm/minimax/model.py` | `99bb765eaa8c4fcc5d279d7aa63f54cc92eeffdc9ecb4a9629996dfb214aceb8` |
| `comfy/lora.py` | `fce6903ca8150611b3f6477c3772bfef0662a2b01747a579ff56b415a44c29bc` |

The conditional target is **the native constructor's defaults**, not an identified FL2VA, Ref2VA or owner checkpoint. Those defaults have 50 main blocks, two token-refiner blocks, hidden width 5376, **56 attention heads** of width 128 and time-embedding width 2688. Adaln curves and gate compression are off. The capture contains 266 matrix targets and 798 actual bare/native/Kohya aliases. Non-matrix state entries are retained separately; ordinary low-rank pairs targeting those entries are not supported by this resolver.

For example, the captured qkv target is `[21504, 5376]`, attention output is `[5376, 7168]`, and main adaln projection is `[96768, 2688]`. The defaults must not be substituted for inferred or loaded checkpoint parameters. A sparse adapter's index range cannot select a target. No semantic claims about faces, garments or block importance are made.

## Resolver

The research resolver accounts for every tensor against the captured native aliases and matrix dimensions. It recognizes ordinary A/B and up/down pairs, compatible ranks and scalar alpha shapes. Alpha presence is observed; alpha values and effective contributions are not measured. Unknown targets, unsupported formats, mixed/incomplete/duplicate factors, aliases colliding on one target and dimension mismatches prevent complete coverage.

Only resolved native target keys determine main/refiner/other groups. Generic transformer-prefixed names are not accepted merely because the inspection view recognized their topology. Optional gate-compression targets, alternate attention dimensions and curve-adaln dimensions do not silently fall back to constructor defaults.

A complete match is named `mapped_candidate`, with `source_target_conditional=true`. Checkpoint, architecture, patch application, measurement and export verification remain false. This deliberately separates source-conditional dimensional matching from compatibility with a selected model or successful loader application.

## Local characterisation

The read-only 10 October library characterisation inspected the same 92 current H3 adapters and read 5,232,893 header bytes, with no tensor payload reads. **86 matched every ordinary pair against this conditional target; six need review**: three have main adaln dimension mismatches, two use unsupported LoKr tensors, and one has conflicting aliases/factors for native targets. These are not corruption diagnoses or a claim that 86 adapters work with every H3 checkpoint.

Motion Booster is one of the adaln-mismatch cases. Its inspection graph correctly counts 258 complete ordinary pairs, but that count alone cannot approve all 258 native patches. This is why the released graph remains an observation view and does not offer fabricated H3 weights or export. Detailed library evidence stays local in `.local/h3-native-header-audit.json` and is excluded from source transfer.

## Validation and continuation

The capture script is `Database/backend/tests/fixtures/capture_h3_native_target.py`; recapture with the read-only Comfy root and an output JSON path. It refuses source drift. The manifest is `Database/backend/contracts/h3-native-constructor-defaults-v1.json`. Regression tests cover captured dimensions, actual alias formats, sparse depth, refiner/other mapping, scalar-alpha limits, unsupported/ambiguous factors, alternate shapes and source-pin rejection.

Full local validation and exact-source GitHub/Farnsworth checks are required before merge. This package's source fixtures contain no model payload, personal database or image. Next, identify an explicit checkpoint/task target and account for any supported source-to-target conversion before measurement/export. Investigate maintainer loader application and required control/distillation roles. Node installation and visual render acceptance remain separate owner involvement points.
