# Studio measurements and experiments

For newly selected stacks, the main button now runs the [managed role starting proposal](main-button-role-start-2026-10-09.md). That workflow automatically measures when needed and displays/copies the resulting values. The separate conservative experiment described below remains available after a recipe or personal profiles are explicitly prepared. Saved recipes load only by explicit choice.

The Studio now separates three things: exact saved block weights, measured parameter updates, and an explicitly uncalibrated balancing experiment. None of these is a claim that an image will look better. Controlled ComfyUI comparisons and owner judgement remain necessary.

This workflow merged in [PR #80](https://github.com/clink2012/LoRA_weights_builder/pull/80) as `4b29d32074e04504a39f6364cb623eb4351b346b`, from authored source `6021c313d53d898f107a1520b4e32ffa94853c57`. The later [lightweight library refresh](library-refresh.md) is a separate package with local verification complete and hosted CI pending at this capture.

## Use the workflow

1. Select compatible FLUX.1 LoRAs. Prepare their block values, then capture any missing Defaults using the measurement panel. Saving a Default records an immutable baseline; it does not change the LoRA file.
2. Prepare again with those saved versions. Choose **Measure** to read the selected native factors using the optional CPU environment. The ordinary app stays responsive and the job can be cancelled. Selection, versions, target and preparation digest must all match.
3. Inspect block norms and signed alignment. These describe effective parameter updates, not semantic influence, image conflict or an image-quality score. Source or worker changes make old measurements stale.
4. For an experiment, explicitly set each LoRA to **Protect**, **Normal** or **Flexible**, then preview. Role labels such as person/clothing/style do not choose priorities automatically.
5. Review changed blocks, exact before/after values and the trial min/max. Save a named experiment only if useful. It creates separate revisions for changed LoRAs plus a pinned composition, with the full measurements and policy recorded together.
6. Load the saved composition and prepare fresh numeric outputs before copying each loader's vector. Returning to Default or an earlier recipe preserves the later experiment.

Existing A/B settings remain in their original parent versions. Changed experimental children use the displayed per-slot policy ranges instead of inheriting potentially inconsistent shared A/B values. Unchanged entries retain their existing version and A/B settings.

## First experimental policy

`gentle_positive_alignment_v1` uses the measured effective-update Gram values and the current saved block/model multipliers. It reduces a lower-priority contribution only when positive pair pressure exceeds 0.5; maximum attenuation is 20%. Both constants are declared trial choices, not calibrated safe limits. BASE and equal-priority contributors remain unchanged. No block is newly activated, no sign is flipped and no individual magnitude is increased.

Negative parameter alignment is shown as cancellation, not automatically suppressed. The policy checks the full signed combined energy after a proposed block change and restores that block's original values if attenuation would increase energy by reducing cancellation elsewhere. Scalar attenuation does not improve cosine alignment itself.

The preview includes sorted numeric bounds for negative as well as positive values. These are policy trial intervals, not architectural limits or universal recommendations. A preview can correctly produce **no qualifying adjustment**. That is not proof of compatibility. In the real three-LoRA sample measured on 3 October, maximum positive pair pressure was approximately 0.00774, below this policy's threshold; no reductions were proposed. The threshold was not lowered to manufacture a recommendation.

## Worker, freshness and recovery

Analysis uses a fixed project-owned CPU worker with one active job, bounded resources and a hard timeout. Windows starts the worker suspended, assigns its whole process tree to a job object, then allows execution. Cancelling, timing out or closing ownership stops both the virtual-environment redirector and its descendants. The API process does not import Torch.

The worker checks the pinned CPU dependency versions. The service verifies saved references, preparation digest, source header/stat identities, target/loader source pins and worker fingerprints before and after analysis and when retrieving a completed result. A changed or interrupted job has no current metrics authority. Failures, cancellation and failed receipt storage cannot expose partial successful measurements.

Saving accepts identifiers, priorities, a name and server proposal digests; browser-supplied vectors or provenance are rejected. Changed profile revisions, the exact recipe and the full experimental receipt commit in one SQLite transaction. Fresh preparation uses that same connection so inserted versions are visible before commit. Any failure rolls back all of them. An unchanged exact retry returns the original historical receipt, with fresh preparation still required for Copy.

The database retains the complete experiment measurements and policy preview, so deleting temporary job files does not destroy saved experimental provenance. Back up the database with the established snapshot procedure; raw unsaved job receipts additionally live in `.local/analysis-jobs`. Model files remain read-only. See [local-state-backup.md](local-state-backup.md) and [local-launcher.md](local-launcher.md).

## Verification boundaries

Independent reviews cover signed maths, precision, transaction rollback, stale result handling and process ownership. Offline suites include dense-update comparisons, fault injection, stale source/selection tests, idempotent/concurrent saves and Windows descendant termination. Browser checks use both themes and narrow layouts.

At the local package checkpoint, **332 backend tests passed** in the pinned CPU environment with no tensor skips, and **54 UI tests plus lint/build passed**. Independent reviewers checked the maths, worker lifecycle, atomic save and asynchronous UI state. Review found and corrected large-integer preservation, exceptional worker cleanup, failed receipt persistence, late cancellation responses and stale-source Copy authority. The later catalogue-refresh package is separate from these counts.

The real three-LoRA measurement path completed in the built local app and left exported values unchanged. A separate, clearly synthetic pair of aligned native LoRA files exercised actual CPU analysis through HTTP, a DOUBLE 0 change from 1 to 0.8, atomic save, exact retry and fresh recipe restoration. Its loader export used 12 slots because the sparse adapter has one mapped group and the pinned loader requires padding; its canonical view still contains all 58 architecture slots. This is workflow evidence, not rendered-image evidence.

A separate SQLite restore recovered all three profile versions, one composition and one full experiment receipt from that synthetic exercise exactly, with the experiment immutability trigger still enforced. The original application database remained unchanged. Private receipts are under `.local/experiment-runtime-receipt.json` and `.local/experiment-restore-receipt.json`.

Hosted [standard CI run 37156465208](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156465208) and the separate [real CPU run 37156360963](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37156360963) both completed successfully against the authored PR #80 source above. These checks cover software behaviour and actual tensor execution, not owner acceptance or controlled renders.
