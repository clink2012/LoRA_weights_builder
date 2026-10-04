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
