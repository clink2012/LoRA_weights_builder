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
