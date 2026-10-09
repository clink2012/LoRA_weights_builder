# Main-button role-aware starting proposals

Owner direction on 9 October: preparing a newly selected stack should show a role-aware experimental starting point. Personal recipes require an explicit load. This supersedes automatic preferred-recipe recall from the earlier package.

## Calculation

`managed_role_start_v1` is separate from the conservative `role_measured_start_v1` experiment. It uses immutable original Defaults and their source roles, actual effective-update measurements, and independent vectors for every LoRA. No profiles or tensor magnitudes are averaged or normalized into loader settings.

| Role | Initial multiplier factor | Priority for further overlap reductions |
| --- | --- | --- |
| Character / person | 0.90 | Protect |
| Clothing / coat / footwear | 0.65 | Protect |
| Pose | 0.75 | Protect |
| Style / lighting | 0.60 | Flexible |
| Environment | 0.45 | Normal |
| Utility | 0.35 | Normal |
| Unknown / other | 1.00 | Normal |

These are declared trial preferences. Character, clothing, style, environment and utility levels come from the project's earlier advisory role strengths. Pose is kept nearer full strength, lighting follows style, and unknown intent receives no guessed starting cut. They are not learned from the model, workbook patch counts, or rendered calibration. Historical workbook normalization rules are not silently adopted.

For source `i`, block `b`, original multiplier `m`, model strength `s`, and measured update `D = alpha_scale * B @ A`, the initial coefficient is `c = s*m*role_factor` on measured non-BASE blocks. BASE, zero multipliers, and blocks with zero measured update retain their original settings. Signed combined energy is `E = sum(i,j) c_i*c_j*<D_i,D_j>`.

The existing positive-overlap check runs on those starting coefficients. Pair pressure is `max(2*c_i*c_j*<D_i,D_j>, 0) / (c_i²*||D_i||² + c_j²*||D_j||²)`. Above 0.5, a lower-priority contribution may receive at most a further 20% reduction. Equal priorities are not arbitrarily split. The full signed stack is checked against the original coefficients: if the role cuts weaken cancellation and increase energy, that block returns to its original settings. This is a numerical guard, not proof of visual stability or preserved semantic effects.

The receipt retains role factors, before/after vectors, measured norms, signed interactions and reasons. The graph shows original update sizes as a distinct line, current contribution bars and editable multipliers. Each export is freshly rebuilt against the pinned loader's resolved patch coverage. Unrepresentable sparse double/single boundary collisions fail closed.

## Workflow and persistence

1. Select two to eight eligible LoRAs and press **Prepare block values**. It captures/reuses original Defaults, prepares the exact sources, recovers matching measurements or runs the bounded CPU worker, then computes/reuses a baseline and prepares exports. Progress and cancellation are available. One LoRA alone uses ordinary preparation because it is not a stacking problem.
2. Inspect the role summary, graph and **Copy full vector** cards. All use the same server-owned proposal. Missing analysis dependencies, failed measurement or invalid mapping produce an error. Neutral placeholders are never presented as a successful proposal.
3. The immutable computed baseline is retained automatically, separately from personal profiles and recipes. **Build a fresh proposal** recomputes from originals. Reuse still requires current source, loader, policy and worker validation.
4. **Save recipe version** explicitly commits proposed personal versions, the composition and complete measured-policy receipt atomically. Defaults and earlier recipes remain unchanged. The saved stack loads and needs ordinary fresh preparation before Copy.
5. Editing an unsaved stack proposal keeps every proposed vector as a pending personal draft. Save or discard each draft before preparing; saving one member must not silently reset the others to 1. Saving the proposal as a recipe first is the quickest route when only one member needs an edit.
6. **Load recipe** or **Load preferred recipe** explicitly restores saved settings. Selection alone performs no preferred lookup or automatic personal-version recall. Preparation checks a loaded stack's exact values without applying role factors again. Preferred shortcuts and historical recipes remain stored.

Original folder roles supply a fresh proposal's intent. A deliberately selected personal revision or loaded recipe uses its saved settings through ordinary preparation; **Build a fresh proposal** returns to original Defaults. Changing supporting strength and assigning a role in a personal revision is not secretly reinterpreted as permission to recalculate that revision.

Tests cover four roles, per-block measured adjustments, signed cancellation, original preservation, sparse exports, stale sources, cache/recompute, worker cancellation, exact Copy, explicit save/load and multi-member drafts. GitHub and Farnsworth checks are the publication gate. Controlled owner renders remain the gate for usefulness, including the unresolved Sabrina/tights clash.
