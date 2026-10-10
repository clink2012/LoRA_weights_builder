# Local visual questionnaire

> Current owner scope (10 October 2026): [active library and graph drawing](../../docs/active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

Temporary design discovery for LoRA Comfy Combiner. It is separate from the application, its database and all model files. The examples show full block vectors; their values and A/B bounds are illustrative, not validated recommendations.

## Open on Bender

In **Bender / PowerShell**:

```powershell
Set-Location 'E:\LoRA Project'
& '.\tools\gui-questionnaire\Launch-Questionnaire.ps1'
```

This starts a hidden Python server bound only to `127.0.0.1:5186` and opens the browser. It uses the project's existing Python environment, or the installed `python` command if that environment is absent. No additional packages, model access, cloud service or application API are required.

To stop it while keeping the answers:

```powershell
Set-Location 'E:\LoRA Project'
& '.\tools\gui-questionnaire\Launch-Questionnaire.ps1' -Stop
```

## Answers and reference images

Choices start unanswered. The sample preview's initial appearance is not a submitted preference. The page reports whether a draft is saved and provides a final review/submission action. Skipped choices stay unanswered.

- Latest saved state: `.local/gui-questionnaire/answers.json`.
- Each completed submission: `.local/gui-questionnaire/submissions/completed-*.json`.
- Owner-supplied style references: `.local/gui-questionnaire/reference-images/1.png` through `7.png`.
- Runtime logs and process identity: `.local/gui-questionnaire/`.

These local files are excluded from Git. The supplied images were copied unchanged from the owner's attachments for this session; they are references, not application assets. Another checkout can still run the questionnaire without them. It will need its own reference copies to show the gallery.

No user responses are committed or sent to external services. The agent can read the saved response file when the owner returns to the chat. This page does not automatically send a chat message or implement any chosen design.

## Checks

Run the persistence tests with `python tools/gui-questionnaire/test_server.py`. Browser verification should use a separate server/data directory, so test answers never become owner answers:

```powershell
Set-Location 'E:\LoRA Project'
& '.\.venv\Scripts\python.exe' '.\tools\gui-questionnaire\server.py' --port 5187 --data-dir '.local\gui-questionnaire-test'
```

Check responsive layout, all choices/review, exact-length numeric and A/B vectors, failed-save recovery, save/reload and retained submission history. The production application test suite is unrelated to this isolated questionnaire; its known baseline failures remain recorded in the restart receipt.
