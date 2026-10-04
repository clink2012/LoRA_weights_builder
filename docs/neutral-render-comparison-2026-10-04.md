# First neutral FLUX.1 render comparison

The owner rendered the separate prepared Inspire workflow and supplied `IMG_000164.png` on 4 October 2026. The standard-loader reference is `IMG_000163.png`. The owner considered the side-by-side images visually identical; direct comparison of the original decoded PNG pixels confirms an exact final-image match.

| Check | Result |
| --- | --- |
| Dimensions and decoded mode | Both 3184 x 4096, RGB |
| Compared pixels | 13,041,664 |
| Differing pixels | 0 |
| Maximum/mean absolute channel difference | 0 / 0 |
| Shared decoded pixel SHA256 | `a97d21723034d9d28560c148de49268ba871478bc2cbe4ee936ba081fa2c411c` |
| Standard PNG file SHA256 | `588e1721d6f0fd7f33f0fe33025f4bbab99e1b736040ada4d0c219b9119eed4a` |
| Neutral PNG file SHA256 | `07b286310a1e8124b26ef72025f79f075f63ff8780a39d8c69d7d87b4e10b6b8` |

PNG files differ because their containers include different workflow metadata; the decoded image samples are identical. No screenshot resizing or image conversion was used for the comparison. Originals were only read, and exact copies plus extracted metadata and a machine-readable receipt were preserved in the private project folders.

## Scope of the result

The supplied new PNG's executed prompt contains active Inspire loaders 665,666,667: Cyberpunk Anime Style1.0, aidmaImageUpgrader0.5, fluxlisimo1.0, in that order. Each contains58 numeric ones with bypass and inverse disabled. The original standard loader's trigger text is preserved by a string node. Downstream model/CLIP connections target the new chain. The captured executed seeds, sampler settings, prompts, refinement, upscaling and colour matching are preserved.

This is a passed neutral-loader transition for this particular three-LoRA stack, captured workflow and pair of final outputs. It is not a general proof for every adapter, model family, seed or runtime version. Only final PNG pixels were compared; intermediate tensors/first-pass pixels were not supplied. No new runtime trace or historical full-file LoRA hashes were available, so the receipt does not assert those identities or infer cache behaviour. The assistant did not run ComfyUI or generate either image.

The result does not validate automatic balancing or claim that non-uniform weights improve images. Neutral Default remains unchanged. The next exploratory comparison varies one LoRA's block multipliers while keeping all other generation settings fixed: Cyberpunk double groups0.8/single groups1 versus double groups1/single groups0.8, BASE1 in both. The0.8 setting is a deliberately chosen test level, not an inferred optimum, semantic block assignment or calibrated recommendation. Owner evaluation of those outputs is still required.

## Subsequent group-sensitivity renders

The owner subsequently supplied A.png and B.png. Both are 3184 x 4096 RGB. Independent comparison of all 58 executed nodes confirms that the only changed generation input relative to the neutral run is Cyberpunk node665's block vector. A reduces the 19 DOUBLE slots to 0.8; B reduces the 38 SINGLE slots to 0.8. BASE, other LoRAs, global strengths, prompts, seeds and downstream settings are unchanged. Display titles, a GUI node size and viewport positioning differ without changing generation inputs.

The owner observed differences in signs and lighting, the belt, clothing at the hip, mouth, apparent chest shape and wet-ground reflections, while the broad scene remained recognisable. The assistant also observed sign, neckline/clothing-contour and facial-detail changes. These observations show distributed sensitivity for this specimen; they do not establish a semantic role for either group, the training content of a LoRA, or better balancing. Apparent body differences can involve both contours and shading; no separate anatomical measurement was made.

Relative to neutral, mean absolute RGB channel differences on the 0–255 scale are approximately 8.787 for A and 12.906 for B. These are descriptive pixel differences, not perceptual-quality or semantic scores. The groups differ in size (19 versus 38) and in learned updates, so greater change in B does not establish that single blocks are intrinsically more powerful. Both trials use one captured seed/prompt/stack and a refinement/upscale pipeline; no intermediate images or changed file/runtime identities were independently measured.

Source SHA256: A `b453efdedd2959cb23e2c44725565dea49e68735a771311de0e285e095ef5580`; B `da8cfc74eefbedf20cfccae2b859ff709c845a035c43ddffea3cfd017fc68e85`. Exact copies, extracted metadata and a comparison receipt are retained privately in `.local/render-baselines/sensitivity-A-B-20261004`. The owner has been asked which result to keep overall; no preference is inferred from the differences. Neutral Default and application settings remain unchanged. These results have not been promoted into a recommended profile or automatic balancing rule.

## Evidence locations

## Owner correction: test clash repair

The owner rejected treating these already-coherent A/B images as a preference-optimisation exercise. Their differences demonstrate sensitivity, not the app's intended ability to repair unwanted interactions. No choice among neutral/A/B is required. Keep the neutral result as loader verification and the A/B result as exploratory evidence only. The owner authorised shortlisting a new candidate pair from the current library, and explicitly requested first-pass-only renders with refinement and upscaling bypassed. The current procedure is in `clash-repair-test-plan.md`.

### Private artifacts

- `.local/render-baselines/standard-IMG_000163-20261003`: original reference, extracted metadata and prepared neutral workflow.
- `.local/render-baselines/neutral-IMG_000164-20261004`: exact new PNG copy, extracted metadata and `pixel-comparison.json`.
- `.local/compare-owner-neutral-render.py`: reproducible local comparison using Pillow/NumPy from the existing bundled document runtime; application dependencies were not changed.

These private artifacts are excluded from Git. Include this render-reference folder in the Bender backup plan. ComfyUI, model files, the application database and the saved neutral workflow were not modified by the comparison.
