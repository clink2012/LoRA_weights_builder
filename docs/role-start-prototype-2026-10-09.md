# Role-informed measured start prototype

This package connects saved profile roles to the existing measured experiment. The Studio starts with **Use saved roles and measurements**, while manual priorities remain available. A role rule supplies a starting priority only when the owner has not overridden it. The backend resolves saved roles and the original metrics itself; browser-supplied tensor measurements, block vectors and role facts remain disallowed.

The policy retains independent multiplier vectors, original and adjusted update magnitudes, graph-ready guidance and the complete explanation in the immutable experiment receipt. Saving still creates new personal variants and a composition atomically. Default and historical experiments are unchanged. This is a first conservative prototype; it deliberately retains the existing 0.5 positive-pressure threshold and 20% attenuation bound. Identity and clothing can both be protected, so their clash can remain unresolved.

Original measurements are not editable settings. A block with measured norm 12, strength 0.8 and multiplier 0.5 has adjusted norm 4.8; the loader still receives multiplier 0.5, not 4.8. Relative measurement curves and colour states are display guidance only. No normalized magnitude is substituted into loader output, no profile is averaged, and no block is labelled as universally face/clothing/footwear.

Scope limits: this package supplies measured data for the upcoming graph, not direct bar editing or the original-line display. Automatic baseline caching/preferred combination variants and library relocation are subsequent work. No controlled owner-rendered repair has passed; software verification is separate from image acceptance.

## Validation

Focused backend checks: 74 passed. Focused UI checks: 9 passed. Full Bender backend checks in the pinned CPU runtime: 464 passed, with no tensor skips. Full UI checks: 86 passed; lint and production build passed. Backup, launcher and questionnaire checks: 23 passed. The first sandboxed attempt was blocked by Windows file/temp permissions; the same checks passed outside those restrictions with a workspace-contained temporary directory. The analysis environment reused existing test libraries through a process-local Python path; no dependencies were installed or changed. Remote checks remain the merge gate.
