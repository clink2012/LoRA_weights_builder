# Compact Studio and current-library checks

Merged in PR #83 as `90d56db70bd672f500eef435b1960f0e1db8c9e5`, from authored source `81b4ca6afe579938a86b7568f6886350d551387e`. Standard CI37160278570 and actual CPU CI37160278462 passed; the owner preview was updated using its preserved database. The subsequent [neutral render comparison](neutral-render-comparison-2026-10-04.md) passed with an exact final-pixel match.

This package implements the owner's post-preview corrections: Carbon replaces Atelier while Prism remains; all BASE/double/single values share one continuous chart; filters can collapse; and Build plus Compare & experiment preserve their mounted state. Named drafts, Default history and exact backend export remain authoritative. A neutral Default genuinely shows flat ones; the chart never invents variation for decoration. Signed values have exact labels and striped negative bars.

The first selected LoRA now controls structural candidate filtering before pagination. Excluded and unverified candidates are hidden by default, with an optional reason view. Current support uses the pinned FLUX.1 loader/header contract. These checks do not prove visual compatibility or verify an arbitrary checkpoint. A reference that becomes unreadable invalidates the current preparation; unrelated transient candidate-query failures do not silently replace the stack.

A server startup starts one background library scan against the launcher's selected database. Browser refresh reads status instead of starting another scan. Inventory completes atomically before bounded header observations; the app remains usable during header work. Manual scan, stop, remaining-check resume and issues are available. Missing-file history and immutable versions are preserved. Unsupported families/formats are explicitly unverified, never silently marked clean or corrupt. See [library-refresh.md](library-refresh.md) for limits and API contracts.

## Local evidence

Final local checks passed: 429 backend tests using the pinned CPU tensor environment; 81 UI tests; lint and production build; 9 backup, 7 launcher and 7 questionnaire checks. Hosted CI and merge evidence are recorded separately when complete; this local receipt does not itself claim deployment or finished-app acceptance.

Independent review corrected header-read budget accounting after source replacement, late reference-read failures, root/junction changes, nonfatal startup scan contention, resumed-scan summaries and issue pagination. Regression coverage exercises these cases. The API scanner resolves database/root only at startup; disabled test startup does not resolve a real database factory.

The actual startup rehearsal used `.local/scan-validation/Database/lora_master.db`, restored from a freshly verified owner-preview snapshot. It discovered 1,941 current files and retained 1,003 missing entries, with no additions in this pass. All 1,941 headers were attempted within the batch, accounting for 390,531,986 header bytes. There were 130 entries with one or more review/not-checked observations; 1,223 lacked a recognised family pattern. These are checker coverage and observation counts, not counts of broken files. The `happy.safetensors` LTX-folder/FLUX-like-header disagreement was observed explicitly.

Across the full current library, FLUX.1 preflight returned 188 eligible, 1,732 excluded and 21 unknown. Within the Studio's FLX folder filter the same eligible set appeared with 67 excluded and 21 unknown. Actual selected files were Cyberpunk Anime Style, aidmaImageUpgrader-FLUX-v0.3 and fluxlisimo_v4_lora_FLUX. Their exact header/loader checks passed; rendered synergy remains unproved by these checks.

Separate built-browser testing confirmed three fresh backend export vectors, one continuous 58-slot chart, Carbon and Prism, filter collapse, mounted tabs, and an unsaved signed edit surviving manual inventory refresh. Old Copy results were invalidated. Synthetic responsive fixtures covered 1600/1024/736/390/320 pixels without page overflow; browser tests reported no errors. Private receipts/screenshots are under `.local/studio-qa` and `.local/scan-validation`.

The original `Database/lora_master.db` stayed unchanged (SHA256 `9af0501b6aab2d9aabb63e30bb36062b0a76c0eb61bb84f2a136dcf0330a2d35`). The durable owner-preview copy was backed up before this package; its versions must never be replaced by the rehearsal copy. ComfyUI and model files remained read-only.

## Render comparison

The owner supplied `IMG_000163.png`, generated with a standard LoRA loader. Its exact image and embedded workflow/settings are preserved privately under `.local/render-baselines/standard-IMG_000163-20261003`. The actual chain is Cyberpunk1.0, aidma0.5, fluxlisimo1.0, followed by refinement/upscaling/colour matching. It is a coherent standard-stack reference, not evidence of block-weight improvement. The next controlled comparison preserves the executed seeds, prompts, order, strengths and downstream stages while using neutral block multipliers. Compare first-pass output as well as the final refined image before evaluating any separate block-weight proposal.
