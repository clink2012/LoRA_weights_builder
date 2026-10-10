# Selected H3 workflow and curve target

The owner selected `DasiwaMinimaxH3WorkflowsT2VA_cMMH3V23.json`, then explicitly identified the current checkpoint as `DasiwaMinimaxH3_dasiwaREF2VAHybridV1.safetensors`. The saved workflow's older FL2VA/Ref2VA filenames are not the current target. The workflow/model remain read-only and neither is transferred for CI.

The workflow's DaSiWa advanced LoRA loader uses **Basic** mode. Its installed source applies the complete adapter with `master strength × video multiplier` to model and CLIP; the audio multiplier does not provide separate audio weighting in Basic mode. It has no per-block vector input. Existing characterization still prefers the native H3 maintainer block loader for future main/refiner/OTHER values. No loader was installed or workflow edited.

## Evidence and distinction from defaults

The selected checkpoint's header matches all **532 state shapes** from the pinned native constructor with `adaln_curve_grid=1025` and `time_embed_dim=8`. It has 50 main blocks, two token-refiner blocks, hidden width 5376, 56 attention heads of width 128 and no gate compression. There are 264 matrix targets and 792 native aliases. The curve layout omits the time embedder and changes all main/final adaln input dimensions from width 2688 to width 8. Default-shape matching must not select this target implicitly.

The header has 932 tensors: 532 constructor states and 400 observed quantization auxiliary entries (scale/descriptor). There are 200 INT8 matrices. The bounded descriptor reader separately read **14,400 bytes of U8 JSON**: all 200 declare `int8_tensorwise`, `convrot=true`, group size 256. This operation is explicitly **not header-only**. No matrix/scale payload was read or GPU model loaded. Quantization/rotation numerical correctness and patch application are unproved.

The local adapter audit read 92 current H3 headers, totaling 5,232,893 bytes, without adapter tensor payloads. **88** map every ordinary pair dimension to the explicit curve target; **four** need review: one has 50 adaln shape mismatches, two use unsupported LoKr tensors, and one has 200 alias conflicts. These are structural candidates, not render acceptance, semantic compatibility or corruption diagnoses. Personal adapter paths/database records remain in ignored local receipts.

## Implementation and validation

`h3_checkpoint_target.py` adds a separate bounded checkpoint reader for observed INT8/U8 and floating storage. It checks duplicate keys, byte counts, dtype/shape/offset validity, complete non-overlapping payload layout and file identity. The existing LoRA reader still rejects integer factors. The optional descriptor reader checks identity across both openings and caps descriptors at 256 entries of at most 1024 bytes each; matrix/scale payloads are excluded.

`describe_checkpoint` compares every expected state shape, reports auxiliaries separately and refuses unexplained tensors, missing states, incompatible/packed dimensions and unreviewed metadata config overrides. It does not enable task, measurement or export capability. This research path has no API endpoint and does not alter the released observation graph.

Symbolic capture accepts bounded explicit constructor parameters and names the target by their hash. It still executes only hash-pinned constructor/alias source, without module startup or torch. `h3-native-curve8-source-v1.json` contains source shapes, not model payloads. The ordinary-default recapture remains byte-identical. The pair resolver accepts an explicit target while retaining conditional/unverified flags.

Regressions cover the layout difference, integer checkpoint reads without relaxing LoRA factors, malformed/duplicate descriptors, budgets, file replacement, complete state coverage, unknown tensors, packed dimensions, metadata overrides and invalid parameters. Full backend CPU, tooling, UI lint/tests/build and exact-source GitHub/Farnsworth checks are required before merge.

Next: native patch-application characterization for the curve/INT8/convrot layout, preserving earlier patches and required distillation/control adapters. Then connect graphs/proposals/export to a validated block loader; Basic cannot consume individual blocks. ComfyUI installation and controlled render judgement remain separate owner involvement points under the intent contract. This package enables no H3 copy-ready weights.
