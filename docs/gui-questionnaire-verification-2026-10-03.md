# Visual questionnaire verification — 3 October 2026

> Current owner scope (10 October 2026): [active library and graph drawing](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

Scope: temporary local design discovery, full-block-vector requirements and the supplied workflow reference. Production application calculations, database, model files and ComfyUI were not changed.

## Delivered

- Six multiple-choice sections and a final review, with initially unanswered choices.
- Four visual directions, three workspace arrangements and three individual-block displays.
- Complete illustrative FLUX.1 vectors: BASE + 19 double + 38 single slots. Default is preserved when trying a personal variant.
- Illustrative A/B bounds, full templates and resolved numeric vectors; no claim that sample values improve generated images.
- Local draft persistence, explicit completed-submission snapshots and saved-status feedback.
- A standard-library Python server bound to `127.0.0.1:5186`, with a hidden-process PowerShell start/stop launcher.

The owner's seven reference images are retained only in ignored local storage. Questionnaire answers are also excluded from Git. Browser QA used port 5187 and a separate data directory; test answers were never written to the owner's port 5186 data.

## Verification

- Seven server tests passed: loopback binding and empty state, durable draft/submission history, invalid-post preservation, foreign-origin rejection, static-path boundaries, corrupt JSON and invalid stored structure.
- JavaScript syntax and PowerShell parsing passed. Launcher start, health check, stop and restart were exercised.
- Browser checks passed for full 58-slot numeric/template output, immutable Default, retained edits across LoRA selection, A/B bounds, preview controls not choosing questionnaire answers, draft save/reload and completed submission/reload.
- Delayed final POST did not show success before acknowledgement. Failed initial GET disabled writes and retried GET; failed POST retained choices and successfully retried saving.
- All nine layout/block-view combinations were exercised. Page widths 1600, 1024, 736, 390 and 320 had no document overflow. Screenshots were inspected, including the narrow preview and settled light theme.
- Independent source review covered domain wording, persistence, launcher and documentation. Findings concerning load-failure overwrite, premature saved confirmation, comparison of edited variants and optional-looking history preservation were corrected before delivery.

The repository has no configured hosted CI workflow at this checkpoint. These checks validate the isolated questionnaire; they do not resolve or replace the production application's baseline test and calculation findings in the restart receipt.

## Next owner action

Open `http://127.0.0.1:5186` on Bender, try the examples, answer the questions and select **Finish & save**. Skipped choices remain open. The saved design brief is input to GUI planning, not automatic approval or implementation of a design.

Use `tools/gui-questionnaire/Launch-Questionnaire.ps1` to reopen after stopping or rebooting. Local responses and reference images remain under `.local/gui-questionnaire/`.
