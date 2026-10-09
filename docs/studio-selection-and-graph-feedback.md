# Studio selection and graph feedback

This change addresses the owner's four observations: first-selection scrolling,
incompatible candidates, graphical block editing, and the trial selector alignment.

The current-file catalogue is bound to the first selected source. Until that
source's check returns, previous candidates are hidden. Excluded and unverified
files remain hidden by default; their optional inspection view cannot add them
to a stack. This is the pinned FLUX.1 structural loader check, not a guarantee
that two eligible LoRAs produce a good image. Historical catalogue browsing
remains available explicitly.

The library keeps its height while results change, and the main workspace no
longer uses browser scroll anchoring. In the original local build, a controlled
first selection moved the main scroll position from 412.44 to 532.44 after the
compatibility response settled. In the updated synthetic preview it remained
437.78 throughout the delayed response. Browser focus scrolling was measured
separately from selection; these numbers are viewport-specific observations.

After preparing block values, the ordinary multiplier chart supports pointer
clicks and dragging, exact signed inputs above each bar, and keyboard adjustment.
The first edit opens profile history and captures Default if needed. The edit is
a personal draft: saving creates a new revision, and Copy requires preparation
of that saved revision. Defaults and earlier revisions remain unchanged.
Deleting or double-clicking a block, or using the reset buttons, restores captured
Default multipliers. Negative bars retain their sign during dragging; numeric
inputs can change it. The scale stays fixed during drafts and shows an overflow
marker when values exceed the display.

The dashed line in the ordinary graph represents captured Default multipliers.
It does not invent measured update sizes. The measured graph continues to show
original effective update magnitudes and current contributions after CPU analysis.
No balancing mathematics or semantic block interpretation changes here.

The trial selector's grid label no longer adds an external bottom margin. Browser
inspection found the selector and both adjacent button bottoms at 548 pixels,
within rounding tolerance. Pointer editing and dragging were also exercised in
the synthetic browser preview, without accessing model data or a database.

Regression coverage includes hidden stale candidates, blocked excluded selection,
failed preflight, exact graph-to-revision-to-export values, original preservation,
pointer coordinates, negative values, fixed scale, resets and busy-state gating.
External CI receipts and source identity are recorded in the pull request.
