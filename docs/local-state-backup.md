# Local state snapshots and restore rehearsal

> Current owner scope (10 October 2026): [active library and graph drawing](active-library-and-graph-drawing-2026-10-10.md). Pony, SDXL and Illustrious are retired; older inventories and family plans below are historical and do not authorise further work on those families.

The standalone `tools/local_state_backup.py` uses SQLite's backup API through a read-only source connection. It includes committed WAL state, checks SQLite integrity and records table counts and SHA-256 hashes. It also copies external profile JSONs and saved questionnaire responses. The tool does not start the application or apply schema migrations.

It deliberately refuses to overwrite an existing snapshot or restore destination. A failed capture may leave a partial directory without a complete manifest; retain it for diagnosis and use a new destination after resolving the cause. Restore validates the manifest first and writes only into a newly created directory. Do not replace the live database as part of a rehearsal.

Run in **Bender / PowerShell**, using a fresh destination name for each snapshot:

```powershell
Set-Location 'E:\LoRA Project'
& '.\.venv\Scripts\python.exe' tools/local_state_backup.py snapshot '.local\backups\my-checkpoint'
& '.\.venv\Scripts\python.exe' tools/local_state_backup.py verify '.local\backups\my-checkpoint'
& '.\.venv\Scripts\python.exe' tools/local_state_backup.py restore '.local\backups\my-checkpoint' '.local\restore-checks\my-checkpoint'
```

The restored directory is a data specimen, not a running app. Run future migration tests against its database using the application's supported test configuration. Promotion back to the active app requires stopping the app, preserving the current data and verifying the chosen restored state; the tool does not perform that promotion.

The first rehearsal at source `411a871` recovered both existing personal profiles and all catalogue/block rows. Automated tests exercise committed WAL content, exact recovered values and profile files, corruption rejection, path escape rejection and refusal to overwrite existing state.

PR #80's separate synthetic experiment rehearsal also recovered three profile versions, one composition and the complete measurement/policy receipt exactly, including enforced immutable experiment history. Saved experiment provenance lives in SQLite. Unsaved analysis job receipts additionally live in `.local/analysis-jobs` and are not included by the snapshot tool's external-file list; copy that directory separately if those unsaved measurements are worth retaining. See [studio-analysis-and-experiments.md](studio-analysis-and-experiments.md).

The merged [library refresh](library-refresh.md) adds presence and scan tables to the selected database. A SQLite database backup carries those tables too. Its copied-data rehearsal preserved all original catalogue IDs and legacy profiles; it did not update the main database or establish off-machine backup coverage.

The current owner preview uses `.local/preview-data/Database/lora_master.db`. Treat this as durable user data, not a disposable rehearsal. New owner profiles, recipes and experiments accumulate there. Include this selected database in backups and preserve it before any eventual main-data migration; never overwrite, delete or recreate it to reset a test. The original main database remains separate and unchanged.

To snapshot that owner preview, run this in **Bender / PowerShell**. Choose a new checkpoint name each time; the tool refuses existing destinations:

```powershell
Set-Location 'E:\LoRA Project'
& '.\.venv\Scripts\python.exe' tools/local_state_backup.py snapshot '.local\backups\owner-preview-checkpoint' --database '.local\preview-data\Database\lora_master.db'
& '.\.venv\Scripts\python.exe' tools/local_state_backup.py verify '.local\backups\owner-preview-checkpoint'
```

`--database` selects an existing database within this project; relative paths start at the project root. Links and junctions are rejected. Omitting the option still selects the original `Database/lora_master.db`. The receipt records the selected source, and every snapshot stores its database at `Database/lora_master.db` so the existing separate-directory restore procedure works unchanged. SQLite's read-only backup connection includes committed WAL changes; neither source database is replaced. External profile JSONs and questionnaire responses still come from the project folders described above.

This snapshot does **not** contain the Git repository, Python/Node environment, LoRA/model binaries, ComfyUI installation, browser-local preferences or generated render results. Git protects committed source separately. Final recovery must also include locked environment manifests, any future external app configuration and irreplaceable binaries/results. Store versioned copies on a separate device and off-site; current AOMEI/Leela/Google Drive coverage has not been verified or changed. Same-disk snapshots alone cannot recover from loss of Bender's disk.
