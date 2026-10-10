# Pause checkpoint: main-button role-aware proposals

> Current owner scope (10 October 2026): [active library and graph drawing](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

Paused at the owner's request on 9 October 2026 so Bender can restart. The commit containing this receipt is a development checkpoint, not a validated release. No push, PR, merge, GitHub check or Farnsworth check has been started for this package.

## Source and scope

- Workspace: `E:\LoRA Project`.
- Base: main commit `4b3881a41b248c771b38c3924c5b234fc8ab4105` (merged PR #98).
- Checkpoint branch: `codex/main-button-role-proposals`.
- Remote: `clink2012/LoRA_weights_builder`.
- Preserve owner models, databases, generated images, secrets and ComfyUI installation. Validation uses synthetic inputs and temporary databases. Only tracked project source and its validation script may be transferred to Farnsworth/GitHub.
- Preserve `.local/preview-data/Database/lora_master.db` and existing saved recipes. Do not write memory without a new explicit request.

## Agreed behavior implemented, awaiting final validation

The main **Prepare block values** button now prepares an experimental role-aware starting proposal for a fresh stack of two to eight LoRAs. It uses original Default profiles, obtains or reuses source-bound tensor measurements, applies declared role priors, then checks measured overlap. A failed or unavailable measurement cannot silently fall back to an all-1.0 export.

Role factors are character 0.90, clothing 0.65, pose 0.75, style/lighting 0.60, environment 0.45, utility 0.35 and unknown 1.00. These are advisory starting preferences, not calibrated optimal weights or semantic block maps. Character, clothing and pose share protected priority; measured overlap can reduce lower-priority contributions further. BASE, unmeasured slots and zeros remain unchanged. A signed-energy guard restores original values at a block if reductions would weaken cancellation and increase its measured stack energy. No averaging of independent LoRA vectors is used.

The server owns proposal provenance and export validation. Computed baselines are cached separately from personal recipes. Saving a proposal is explicit and transactional. Preferred recipes load only by an explicit user action; selecting the same combination does not automatically replace a new proposal with saved values. A fresh-proposal action returns to Defaults and forces proposal recomputation.

The graph and copied Inspire vector show the same proposed settings. Original measured norms remain the graph reference. Editing an unsaved proposal retains every member's proposed values as drafts, preventing an unedited member from silently reverting to 1.0. Selecting a history version also preserves other proposed members as pending drafts. History refresh is available for initially seeded profiles. Ordinary single-LoRA preparation and explicit saved-profile preparation retain their established behavior.

Main implementation files: `managed_start_policy.py`, `experiment_router.py`, `experiment_versions.py`, `lora_api_server.py`, `buildStartingProposal.js`, `App.jsx`, the Studio proposal/history/composition components and their tests. Method details are in `docs/main-button-role-start-2026-10-09.md`.

## Checks completed before pausing

Before the last small review changes:

- Full backend suite: 522 passed in the real CPU tensor environment, with no tensor skips. Log: `.local/managed-backend-full.log`.
- Full UI suite: 141 passed across 21 files. Log: `.local/managed-ui-full.log`.
- UI lint and production build passed.
- Tooling suites: 23 passed (9 backup, 7 launcher, 7 questionnaire).
- Whitespace check passed.

After the last review changes (missing catalogue-row guard, history selection preservation, history refresh button and action-row wrapping):

- Four native proposal API tests passed, temporary directory `.local/test-primary-final`.
- 23 focused UI tests passed across App.starting, App.versions and buildStartingProposal. Log: `.local/managed-ui-review.log`.
- Full suites, lint and build have not been rerun against the final checkpoint. That remains required.

The last test process has exited successfully. There are no active development test workers, synthetic browser servers or delegated agents. No new remote validation is running. The existing owner preview on port 5187 was left untouched; a Bender restart will stop it naturally.

## Runtime state

This package has not been promoted to the owner preview. The existing backend predates the new endpoint. A direct production build changed ignored UI output, but the official launcher build receipt still belongs to the previous release. Do not assume an arbitrary existing preview has matching source and UI. After validation and merge, back up the owner preview database and use the official launcher with `-Build` to rebuild and restart with matching source. Avoid starting the WIP package merely to restore the old preview during the pause.

No new owner renders were performed. The Sabrina/tights clash remains unresolved; passing software checks does not establish a visual repair.

## Resume steps

1. Read this receipt and `docs/main-button-role-start-2026-10-09.md`. Verify branch, HEAD, working tree and any changes made after the checkpoint. Resume the existing work, preserving the agreed algorithm and explicit recipe-loading behavior.
2. Add a focused regression for selecting a history version while peer proposals remain unsaved. Review the Save retry/error reporting and profile-selection state for consistency. Fix only concrete findings within this development scope.
3. Rerun full backend/UI/tooling checks, lint, production build and whitespace checks against the final source. The existing real tensor command is `.venv-analysis/Scripts/python.exe -c "import sys, runpy; sys.path.append('.venv/Lib/site-packages'); sys.argv=['pytest','Database/backend/tests','-q','-ra','--basetemp','.local/test-managed-full']; runpy.run_module('pytest', run_name='__main__')"`. Windows child-process/temp-file checks may require approved execution outside the restricted sandbox.
4. Perform a synthetic browser review of fresh preparation, proposal values/graph/copy, edit/save, explicit recipe load, history refresh and cancellation. Do not use the owner's database for automated writes.
5. Commit completed fixes. Transfer only the filtered tracked source snapshot and validation script to Farnsworth. Existing ignored helpers `.local/prepare-role-start-transfer.py` and `.local/role-start-farnsworth.sh` may be inspected and reused after verifying their source-only filters. Stage under `/home/clink/.cache/lora-validation/<full-source-commit>`.
6. Push the branch, create and attach its PR, and run both normal GitHub CI and the real-tensor workflow plus Farnsworth validation for the exact source commit. Fix failures. Merge only after all required checks pass; verify the merged source and post-merge checks. Standing owner authorization covers this source work and validation, excluding models, owner databases, secrets and images.
7. Back up the owner preview data, then rebuild/relaunch using `tools/Launch-LoRA.ps1 -Build` once the release is validated. Explain where proposed ComfyUI values appear and the explicit recipe-loading behavior. Visual usefulness remains an owner-controlled render trial, not an automatic software claim.

No further development should run until the owner resumes after restarting Bender.
