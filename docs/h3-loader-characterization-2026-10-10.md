# MiniMax H3 loader characterization

Current owner scope: the [eight active families](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious remain retired. This package follows approval of graph drawing and explicit personal-recipe loading. It changes no UI, database, model file or ComfyUI installation.

## Result and limits

Two pinned H3 candidate loaders now have independently captured behavior fixtures, reproducible source-reading capture tools and strict research serializers. Neither serializer is registered with an API, exporter or capability flag. Both return `characterized_candidate` and `export_verified=false`. MiniMax H3 remains metadata-only in the application.

The preferred route for the next investigation is the native H3 maintainer loader, because it exposes main, token-refiner and other groups separately and scales resolved patch strengths without rewriting factors or alpha. This is a source/synthetic finding, not an installation recommendation or a guarantee for transformed checkpoints. The older selective loader remains a characterized comparison, not the selected runtime contract.

| Candidate | Pinned source | Characterized input |
| --- | --- | --- |
| [Realtime selective loader](https://github.com/shootthesound/comfyUI-Realtime-Lora/blob/47d5962a651e61a39afcf06c7cf26614454c5fe7/selective_lora_loader.py) | Revision `47d5962a651e61a39afcf06c7cf26614454c5fe7`; SHA256 `0a0cac0e3d3de4be0128eddbd1f2803045e2f1542d3219232e4a0c1a6468ed45` | 50 fixed main positions, then explicit OTHER. Refiner shares OTHER. |
| [FL native H3 maintainer loader](https://github.com/filliptm/ComfyUI-FL-MiniMaxH3/blob/4e6379770e7f4fc48651b76d03284a7eee7df8a4/nodes/FL_MiniMaxH3LoraBlockLoader.py) | Revision `4e6379770e7f4fc48651b76d03284a7eee7df8a4`; SHA256 `3d57625d2fb907bbbb8902d293287df7db5d721b4032a544eaa1a971a82f29d8` | Full main/refiner text overrides plus separate OTHER and overall model-strength widgets. Counts derive from the connected native model. |

Pinned public sources were fetched into ignored `.local/family-research`, hashed and executed only through selected AST methods against synthetic inputs. Module startup, file I/O, Comfy target resolution, model patching and saving are stubbed. No upstream source is vendored or imported at application runtime. The older capture additionally uses real CPU float64 A/B matrices to observe signed linear factor updates.

## Observed behavior

The selective loader retains architecture positions for sparse files. It scales only the output/up factor for tested ordinary native A/B and Kohya up/down pairs; alpha and the input/down factor remain unchanged. For these cases the effective update is `outer_strength × block_multiplier × (alpha/rank) × B@A`. A zero removes the entire target pair. The capture exercises main 0–49, negative block and overall strengths, sparse targets, token-refiner, non-main targets and numeric precision.

Important restrictions: 50-number text defaults OTHER to 1; malformed text falls back to preset controls; returned text rounds to two decimals. A main index outside 0–49 is silently dropped; bare `blocks.N` keys route to OTHER. The research serializer therefore requires all 51 numeric values, an explicit OTHER and exact source hash. It must not become an exporter without complete format and target coverage. DoRA, LoHa, LoKr, reshaped/quantized adapters and unfamiliar factor formats are not validated by ordinary A/B tests.

The maintainer loader requires a native `MiniMaxH3Model`, counts both stacks on the connected model, and uses `overall_strength × chosen_multiplier` when adding each resolved patch. Text overrides replace the group multiplier rather than multiply it; later rules win. Tuple patch keys are accepted. Zero groups skip their patches, and overall zero bypasses adapter loading. Empty resolved patches and rejected nonzero patches raise errors. The model clone retains earlier patches; the source model remains unchanged. Synthetic patch objects, including their alpha-bearing payloads, are passed unchanged to the stubbed patcher.

Its capture uses a synthetic 50-main/2-refiner model. This does not prove every H3 target has that topology. The research serializer requires explicit bounded target counts, preserves all slots even for sparse adapters, emits one override per main/refiner slot, and keeps OTHER/overall separate. It never derives total depth from an adapter's largest observed index or returns a FLUX-style BASE-first string. The loader's scalar and override inputs accept finite strengths between -10 and 10, which the serializer enforces.

## Validation and next implementation

The fixtures are `Database/backend/tests/fixtures/h3_selective_contract.json` and `h3_maintainer_contract.json`. Their source hashes are asserted by the tests and capture scripts reject changed source. Recapture with the supplied pinned source and the respective `capture_h3*_contract.py` script; the selective capture requires the optional CPU analysis environment. Normal tests consume the recorded observations offline without downloading source or requiring an H3 checkpoint. These fixtures characterize the source version, not independently reproduce Comfy's resolver.

Required package checks are focused regressions, full backend CPU tests, tooling, UI lint/tests/build and exact-source GitHub/Farnsworth validation. Release evidence is retained locally with source/runtime identities. No UI behavior changed, so promotion uses the existing official launcher and a verified backup of the copied preview database.

Local validation passed: 38 focused H3 checks; 561 full backend tests in the real CPU environment; 146 UI tests; 23 tooling tests; lint, build and whitespace checks. Remote checks and release promotion are recorded against the final source in the local release receipt.

Next: build complete source-key and native target coverage evidence for the preferred H3 contract. Separate FL2VA/Ref2VA and native/pruned/quantized/convrot target identities; classify main/refiner/other without silently dropping unsupported tensors. Preserve creative versus required distillation/control roles. Confirm source-to-resolved mapping before enabling measurements or export. Unmatched original source keys being logged upstream is insufficient for complete coverage. Any node installation or render trial is a later owner involvement point; neither occurred in this package. LTX-2.3/2.5 alpha-replacement characterization remains a separate active-family task.
