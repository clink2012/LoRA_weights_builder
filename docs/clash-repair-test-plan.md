# Demonstrate and repair a stacking problem

The owner clarified the acceptance target on 4 October 2026: the app should help with unwanted results when individually useful LoRAs are combined. Variations of a coherent image do not demonstrate that capability. The neutral pixel-equivalence test remains useful loader validation; the subsequent Cyberpunk sensitivity images are not evidence of successful balancing.

The owner asked management to shortlist from the current library and explicitly approved bypassing refinement/upscaling. Prepare three first-pass-only controls for one pair: identity alone, clothing alone, and the ordinary pair at those same strengths. Hold checkpoint, prompt and trigger words, seed, dimensions, sampler and loader order fixed. Save the first decoded image directly; ensure no separate active output still schedules refinement or upscaling. Model and ComfyUI folders remain read-only; prepared workflows live in the project until the owner imports/renders them.

## First candidate

Fresh header/loader inspection found Fictional Model Sabrina (FLX-PPL-113) plus Polka dot tights (FLX-CLT-040) eligible for the pinned FLUX.1 target. Their distinct intended contributions are the fictional person's recognisable appearance and the garment/pattern. Use an explicitly adult, fully clothed fashion scene with face and patterned legs visible. The Sabrina sidecar describes generated SFW training images but does not verify an age; the prompt explicitly specifies a 30-year-old fictional woman. A trial starting strength is not an author recommendation. No clash is claimed until the controls show one.

A second structurally eligible candidate is Rebecca Raven (FLX-PPL-153) plus Brown Mules (FLX-CLT-005), with identity and footwear as distinguishable goals. The sidecars are self-declared descriptions; folder labels alone are not evidence of architecture or compatibility. A pose candidate that failed adapter validation was excluded. Private source details and a bounded shortlist are in `.local/clash-trial-plan`.

## Pass conditions and sequence

1. Check that each solo delivers its intended effect. If a solo fails, correct that baseline first; do not call it a stacking clash.
2. Inspect the ordinary pair for a specific unwanted change or loss compared with the solos: for example loss of identity, displaced garment pattern, malformed clothing or distorted form. The owner identifies what must be retained and what should be fixed.
3. If no meaningful clash appears, record a working pair and move to the next candidate or one justified third LoRA. Do not increase strengths indiscriminately just to manufacture failure.
4. Once a clash is demonstrated, prepare one separately saved block proposal with an explicit hypothesis and bounds, alongside an overall-strength reduction control. Both must use the same first-pass settings. Where supported by measured update norms, match the scalar control's per-LoRA update magnitude to the block proposal; that does not equate visual influence.
5. A useful repair reduces the named unwanted effect while retaining both intended contributions, without introducing another serious defect. Mere difference is not success. If simply lowering overall strength works as well, record that outcome honestly.
6. Confirm a promising repair with another fixed seed before claiming usefulness for that composition. Broader family or automatic-balancing claims require broader evidence.

The current gentle-reduction policy is uncalibrated. Positive parameter alignment, a lower update norm or a successful software test cannot identify semantic conflict. It can propose a bounded hypothesis; it cannot yet certify a visual repair. Do not reduce thresholds merely to force a recommendation, assign semantic roles to whole block groups from one image, or overwrite Default with an exploratory trial.

## First owner controls: clothing solo failed

The owner returned `1.png` (Sabrina only), `2.png` (tights only), and `3.png` (both) on 4 October 2026. Image 2 is severely blurred across the subject; images 1 and 3 are clearer but do not establish the requested polka-dot tights. The owner identified the missing garment effect in the pair. This is a real unsuccessful outcome, but it does not yet demonstrate suppression of an independently working clothing effect: the clothing-only prerequisite failed.

Independent inspection of the embedded generation metadata confirmed that the three graphs differ only in the intended loader bypass flags and captions. All retain FLUX.1 dev, the same prompt and seed 892489797905638, 896 x 1152, 20 steps, dpmpp_2m/sgm_uniform, denoise 1 and guidance 3.5. Active LoRAs use strength 1 and 58 ones. The unused third loader is bypassed, and first-pass decode directly supplies both outputs. Metadata establishes the submitted configuration, not an execution trace or tensor-content validation. Private exact PNG copies, metadata and hashes are retained in `.local/render-baselines/clash-controls-20261004`.

An identical scene with both selected LoRAs bypassed was prepared as `.local/clash-trial-plan/04-no-loras-first-pass.json`. Its use is now on hold because the owner supplied further regular-loader results before handoff. No render has been queued by management.

## Follow-up regular-loader images: promising conflict evidence

The owner then supplied three pasted images, identified as regular-loader polka dots only, polka dots plus Fluxlisimo v4, and all three LoRAs. The first is clear with visible dotted tights; the second appears to weaken the pattern; the third loses the visible tights and changes the requested long-sleeved blouse to a sleeveless top. This supplies a plausible stacking-loss case and evidence that the clothing effect can work in the owner's setup. It supersedes any blanket claim that the clothing LoRA cannot produce a usable solo.

All three clipboard PNGs lack embedded prompt/workflow metadata. Their stated combinations are owner-reported, and their settings have not been matched to each other or to the earlier Inspire controls. The regular-loader solo's sharpness versus the earlier blurred Inspire solo is a separate unresolved loader/setup discrepancy, not proof of an Inspire defect. The next action is to obtain the original saved PNG paths or workflow/settings from the owner, compare actual inputs, and only then choose a matched loader test or a block-repair hypothesis. Do not require a redundant render if the originals already supply the needed control. No arbitrary block suppression, balancing-success claim, Default change or stronger stacking is justified by these images alone.

## Supplied workflows identify unmatched conditions

The owner subsequently supplied the three saved block-loader tabs and the regular-loader workflow, explaining that the latter's images were copied from the first preview while LoRAs were toggled within one tab. These files are preserved privately in `.local/render-baselines/supplied-workflows-20261004`.

The block-loader tabs retain the original effective FLUX.1 dev selection and linked fixed seed 892489797905638. The regular workflow selects `demonCORESFWNSFW_v25HelheimProjectAIO.safetensors` and has seed controller 664 set to -1 (randomize). Its prompt also adds detail/photorealism text and extra trigger words. The adult-fashion main prompt, sampler, scheduler, steps and guidance agree, but the material model, seed and prompt differences prevent attributing the appearance difference to the loader. The regular saved file records all three LoRAs active at model/CLIP strength 1; it cannot establish the historical seed or toggle state of each clipboard image.

Read connected inputs and exposed subgraph widgets, not stale stored defaults: the block tabs' internal UNET default says DemonCORE but the exposed model selection supplies FLUX.1 dev; RandomNoise's stored widget is overridden by connected seed controller 664. Both loader implementations sit after ModelSamplingFlux. The regular first-preview route is decode 8 through the matching Rendered_Image_Out Set/Get nodes, so downstream refinement does not supply that preview.

The owner requested one Queue action that saves every comparison independently, selected Sabrina only / polka dots only / Sabrina plus polka dots, then expanded this to six variants: those three selections through both regular LoraManager and Inspire. Prepare this matrix in `.local/loader-parity-20261004`, preserving the saved regular setup's DemonCORE selection and full prompt, fixing the effective seed to 105362140904133 as a new test seed, and holding trigger text identical independently of enabled adapters. Fluxlisimo stays inactive in every branch; preserved common trigger wording is not an active adapter. Each active adapter uses model/CLIP strength 1 and each Inspire vector contains all 58 ones. The all-ones baseline was explicitly explained before introducing balancing changes.

All six branches start independently from the common pre-LoRA model and original CLIP; do not feed patched models or images from one case into another. Share prompt text, but encode conditioning through each branch's returned CLIP. Give each branch its own guider, sampler, decode and distinct metadata-bearing first-pass PNG save. Remove downstream processing and redundant outputs. This is a controlled loader/selection comparison before block repair, not a recreation of undocumented historical seeds or proof that either loader caused the blur. The earlier no-adapter04 and two-file parity proposals remain held/superseded; do not ask the owner to run them alongside the six-case workflow.

Prepared artifact: `.local/loader-parity-20261004/00-six-comparisons-one-queue.json`, SHA256 `9474582ade0ec0b5a30d46d26e0b34ccd16d4bcd2dba8b8554a767faa0a700ae`. Independent static review passed all 51 nodes, 104 links and three retained subgraphs, branch independence, selection matrices, fixed inputs and six distinct PNG saves to `E:\Pics\output`. Installed loader cloning/widget restoration and save metadata paths were also checked. Six previews reuse the same six samplers. The artifact is prepared for owner import; it has not been rendered or runtime accepted.

## Six returned renders: parity and clothing loss confirmed

The owner returned all six labelled PNGs on 4 October 2026. Independent checks found regular/Inspire pairs 01/04, 02/05 and 03/06 identical across every one of their 1,032,192 RGB pixels; each pair also has identical complete PNG bytes. Embedded metadata confirms the intended active adapters, independent branches, fixed seed 105362140904133, DemonCORE selection, common prompt and first-pass settings. Exact copies, extracted metadata and hashes are in `.local/render-baselines/six-loader-results-20261004`. The previous artifact's unrendered status is superseded by these results.

Dotted tights are clearly visible in the tights-only controls and absent in the combined controls. The owner identifies Sabrina as suppressing the clothing contribution. This supports a controlled stack-loss case at this model/prompt/seed and strengths, with loader parity established for this configuration. It does not locate the responsible blocks, establish a universal incompatibility or explain the earlier unmatched FLUX.1-dev blur.

## First repair diagnostic: selected groups versus overall reduction

Keep the tights adapter unchanged at strength 1 and 58 ones. Test reducing Sabrina's 19 DOUBLE slots to 0.5, then separately its 38 SINGLE slots to 0.5; preserve BASE and all other slots at 1. These are manually bounded exploratory settings, not outputs of the existing gentle policy or semantic assignments of identity/clothing to those groups. Do not change that policy or Default to encode them.

Use two overall-strength controls, each matched to the corresponding group trial's measured Sabrina parameter-update norm. The existing CPU measurement engine reads Sabrina's actual factors and alpha: 304 rank-2 patches with alpha 16, 57 transformer blocks and no mapped BASE or separate CLIP patches. Squared update norms are DOUBLE 24961.31683268871 and SINGLE 60742.61200214275, total 85703.92883483146. For block weights w and per-block squared norms q, the matching overall model strength is sqrt(sum(w*w*q)/sum(q)). This gives 0.8840600001152074 for the DOUBLE-half control and 0.684425245148292 for the SINGLE-half control, with all block slots 1. CLIP strengths remain 1 in all cases. Equal parameter-update norm does not imply equal visual influence or equal combined-stack norm.

Prepare the four cases in one Queue workflow under `.local/clash-repair-trial-20261004`, retaining the verified six-case setup and independent first-pass saves. Reuse the existing solo and combined images as references; do not rerender all six baselines. A useful result restores clear tights while retaining Sabrina's recognisable appearance, without introducing another serious defect. Compare each group change with its overall-strength control; if simple weakening works equally well, record that honestly. Verify any promising outcome on another seed before claiming a reliable repair. Measurement receipts and complete proposal vectors are in `.local/clash-trial-plan/sabrina-effective-update-20261004.json` and `sabrina-block-diagnostic-proposal-20261004.json`.

Prepared four-case artifact: `.local/clash-repair-trial-20261004/00-four-block-diagnostics-one-queue.json`, SHA256 `2d25ccfbe0892ca3886c7c38d7acafb18d7d0fc4009583d10a5ca62eab64956b`. Its 38 nodes and 74 links retain four independent first-pass branches and distinct output prefixes. This artifact is awaiting owner rendering; no repair success is claimed.

## Four diagnostic results: no accepted repair

The owner has now returned all four PNGs. Independent metadata checks confirm the exact planned block values, full-precision scalar coefficients and unchanged generation/tights settings. Evidence is retained in `.local/render-baselines/first-repair-results-20261004`; the previous pending-render status is superseded.

Attachment display order was 01, 04, 02, 03, and the owner's image viewer and Explorer order also differed. Their latest explicit correction supersedes the intermediate assessments: **files 01 and 02 retain the face but have no polka dots; files 03 and 04 have slight polka dots but the face has changed.** The darker tights effect remains incomplete. Thus neither broad SINGLE reduction nor its matched overall-strength control delivers an accepted repair, and no selective-block advantage has been established. Faint visible dots alone are not an accepted garment repair. Earlier private proposal receipts captured an intermediate preference; `current-owner-assessment.md` beside the next workflow records the superseding assessment without rewriting frozen calculation evidence.

## One bounded localisation batch

The next experiment divides Sabrina's 38 SINGLE blocks into three disjoint contiguous ranges. In each selective case, set just one range to 0.25 and keep BASE, DOUBLE and all other SINGLE slots at 1; tights and CLIP settings stay fixed. This is stronger but more localised exploratory suppression, not a validated setting or a semantic map of clothing/identity blocks. Each selective case gets its own overall-strength control, computed from the same measured per-block norms:

| SINGLE range at 0.25 | Canonical slots, including BASE at 0 | Matched overall model strength |
| --- | --- | --- |
| 00-12 | 20-32 | 0.9547599054296829 |
| 13-25 | 33-45 | 0.8870170849064339 |
| 26-37 | 46-57 | 0.7982365622504204 |

The coefficient is sqrt(1 - 0.9375 * range_fraction_of_squared_norm). These ranges have different measured update energies; compare each selected range against its own scalar control rather than ranking raw visual changes as inherent block importance. Reused measurement identities/source pins and independent high-precision arithmetic were checked. Full vectors and calculation receipts are in `.local/clash-localisation-trial-20261004`.

Deliver one Queue workflow with six independently saved first-pass cases, using the unchanged verified DemonCORE/prompt/seed setup. Judge likeness, dot pattern and darker garment appearance together. No Default or application policy is changed. If all six fail, reassess the strategy rather than automatically generating progressively finer block partitions. Any promising result still requires another-seed validation before claiming a reliable repair.

Prepared artifact: `.local/clash-localisation-trial-20261004/00-six-localisation-comparisons-one-queue.json`, SHA256 `3bc4759a4557cf2f09c6a9ba97ce1b9b6dc1ff7a9aa6da6d44eff21e3871b508`. Six independent first-pass outputs use prefixes `LoRA_localise_20261004_CASE01` through `CASE06`; compare adjacent pairs by these filenames, not attachment order. The workflow has not yet been rendered.

## Localisation batch complete: stop subdivisions

The owner returned all six PNGs and explicit filename-based assessments. CASE01 changed the woman substantially with no dots; CASE02 retained facial features with no dots; CASE03 partially compromised features with no dots; CASE04 slightly changed features with no dots; CASE05 moved further from the intended likeness with no dots; CASE06 was closer to CASE02 than CASE04, with no accepted garment recovery. None is accepted as a repair. Independent PNG metadata inspection confirms the prescribed vectors, exact scalar coefficients and unchanged common inputs. Exact copies, metadata and owner assessments are in `.local/render-baselines/localisation-results-20261004`.

The owner also observed similar court shoes in earlier dot-producing images and current CASE05/06. The latter have no accepted polka-dot effect, so footwear is not a sufficient indicator of garment recovery. The prompt already requests black low-heeled closed-toe shoes; without a no-adapter comparison, these examples cannot establish whether the shoe style comes from the clothing adapter, reduced Sabrina influence, the base model or their interaction. Do not infer particular training examples or a shoe-specific block from this association.

The agreed bounded subdivision test is closed. Across four broad and six narrower reduction trials, no accepted repair or selective-block advantage has been demonstrated. This does not prove no useful block vector exists. It does mean that increasingly fine blind partitions are not justified. Measured update norms and signed parameter interactions support structural analysis and experimental controls, not semantic block labels or predicted image success. Do not retune the gentle policy merely to force recommendations or promote these failed settings to Defaults.

## Explicit strategy reassessment: strengthen the missing contribution

Until now the trials reduced Sabrina while keeping tights at 1. The next and final bounded calibration for this pair keeps Sabrina at model strength 1, both full vectors at 58 ones and both CLIP strengths at 1, and tries tights model strengths 1.25 and 1.5. Hold the verified model/prompt/seed and first-pass settings fixed. These strengths are exploratory values, not author recommendations or guaranteed acceptable settings. This is ordinary-strength calibration, not evidence of a successful block-weight repair.

Deliver only these two new independently saved cases in one workflow under `.local/tights-calibration-20261004`, with existing solo/pair renders retained as references. Require intended identity, dotted pattern and dark tights together. If neither succeeds, record this pair as unresolved within tested bounds; do not escalate strengths or launch another grid automatically. A no-adapter render could investigate footwear attribution but is not needed to test the already demonstrated clothing loss. Regional conditioning or inpainting would be a separate ComfyUI strategy and must not silently replace the app's full-vector balancing objective.

The product objective remains unchanged: useful full per-LoRA block values for compatible stacks. Automated role-aware visual repair has not been demonstrated. A subsequent implementation package should make exact comparison inputs, immutable trial variants and explicit owner outcomes first-class experiment records, including legitimate no-improvement outcomes; this supports calibration without presenting today's numerical heuristic as validated semantic repair. Keep broader family work and completion criteria intact.

Prepared and independently reviewed: `.local/tights-calibration-20261004/00-two-tights-calibration-one-queue.json`, SHA256 `9b10078d6079623cebe4ee7acf6df8c6a8ffc36f73620033bc36a8cc37f61fce`. The 22-node/38-link graph has exactly two independent branches, checked against the latest actual render: Sabrina restored to all ones, only tights model strength differs, and both first-pass saves retain metadata and distinct prefixes. Owner rendering remains pending.
