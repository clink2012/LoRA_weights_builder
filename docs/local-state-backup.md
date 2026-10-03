# Local state snapshots and restore rehearsal

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

This snapshot does **not** contain the Git repository, Python/Node environment, LoRA/model binaries, ComfyUI installation, browser-local preferences or generated render results. Git protects committed source separately. Final recovery must also include locked environment manifests, any future external app configuration and irreplaceable binaries/results. Store versioned copies on a separate device and off-site; current AOMEI/Leela/Google Drive coverage has not been verified or changed. Same-disk snapshots alone cannot recover from loss of Bender's disk.
