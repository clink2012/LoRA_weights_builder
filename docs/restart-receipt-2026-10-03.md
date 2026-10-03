# Restart review receipt — 3 October 2026

## Scope

Read-only application/source/library/SQLite/ComfyUI review, isolated existing tests and synthetic loader checks; new restart documentation and requested Obsidian notes. No app behaviour changes, migrations, library writes, model downloads, generation or deployment.

Initial local HEAD: `06e8cfa92af0e9f688e78a9feab821bf587478c6`.
Live main: `e2b1e9be6b97c491e4c7176a3459dc5a08ac81b3` (PR #73 merge).
Both source trees: `6d84d6685e131e20f43abb0200027a929965e2f6`.

## Validation

| Check | Result | Boundary |
| --- | --- | --- |
| UI smoke | 3 passed | Existing local Vitest tests. |
| UI production build | Passed | Output directed to a temporary directory. |
| UI lint | Failed: 5 unused variables | `App.jsx` lines 205, 396, 637–639 at reviewed source. |
| Backend subset | 121 passed, 1 expected failure | Excludes tensor extractor module; isolated temporary test data. |
| Complete backend suite | Could not collect successfully | Existing environment lacks Torch; a PEFT test inserts a fake Torch module globally, breaking `safetensors.torch` collection (`torch.nn` missing). |
| Installed loader contract experiment | Current export defect reproduced | AST-only synthetic harness; no ComfyUI imports or startup. |
| GitHub baseline | Verified | Remote main contains the inspected tree; no open PRs at capture. |
| Rendered ComfyUI quality | Not tested | Needs pinned workflow and controlled visual comparisons. |
| Hosted CI/Characterisation | Not run | No tracked workflow existed at baseline. |

Existing global npm launcher points to a missing roaming `npm-cli.js`; tests/build/lint were invoked through Node and installed local tool entry points. No global environment repair or package installation was performed. The expected-failure test is stale scaffold that copies arrays and asserts change without invoking the orchestrator; it is not acceptance evidence.

## Loader experiment

Target: installed Inspire Pack 1.23 source at `E:\ComfyUI_New\ComfyUI\custom_nodes\comfyui-inspire-pack\inspire\lora_block_weight.py`.
SHA-256: `4E6188D8D20E2A13C482D2D7FAE6DDB570210C426265605B7760B5B96EBB914E`.

Synthetic full FLUX topology, app block vector `0.01 … 0.57`:

| Loaded patch | Intended block value | Actual value without BASE |
| --- | ---: | ---: |
| Global/base img_in | Separate BASE expected | 0.01 |
| double 0 | 0.01 | 0.02 |
| double 18 | 0.19 | 0.20 |
| single 0 | 0.20 | 0.21 |
| single 36 | 0.56 | 0.57 |
| single 37 | 0.57 | 0.57 |

Prepending BASE=1 corrected this full-topology example. A sparse fixture with double0 and single0 still assigned both the same ratio because the installed grouping loop compares only numeric block indices. A general export fix must characterise actual patch coverage and group transitions, not merely prefix the array.

App serializer: `lora_block_orchestrator.py:666`; UI serializer: `App.jsx:391–393`; installed loader starts at `vector_i=1` near line 412 and reads BASE near line 451. UI rounds output to one decimal while backend CSV uses four, another reason to keep final output under a single owner.

## Data boundaries

Read-only SQLite inspection found 2,524 LoRA rows, 4,235 block rows, 2 user profiles and 0 combined profiles. No database connection through the production API was used, since startup/connection paths can mutate schema/data. This is not a full identity reconciliation or integrity/restore test.

The current library inventory contains 1,935 safetensors files. Only representative LTX/MiniMax headers were inspected for the new-family loader question. File contents and private model filenames were not added to the repository.

The vault was inventoried/content-scanned and relevant notes read in detail. New app notes and narrow cross-links/path/backup corrections preserve the vault's current style, with original edited notes archived and hashes recorded. Archive evidence is retained in the vault; it is not a model/database backup.

## Delivery boundary

This package establishes a reviewed starting point and design choices. It does not certify current copied values, alter the GUI, implement versioned variants or claim the app is finished. FLUX.1 first, Default-preserving manual variants and read-only ComfyUI/models access are owner decisions already captured. GUI preference and visual acceptance remain owner inputs.
