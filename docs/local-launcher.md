# Local launcher

The launcher serves the built Studio and its native API from one Python process at `http://127.0.0.1:5187`. It binds only to Bender's loopback address. ComfyUI and model files are inspected through the existing read-only header checks; this launcher does not start ComfyUI, install packages, or download models.

The launcher foundation merged in PR #79, the measurement/experiment Studio in PR #80, current-library inventory in PR #81 and header observations/selected-database backup in PR #82. The owner preview uses `http://127.0.0.1:5187` against a durable restored copy. Main-database startup and finished-application acceptance remain pending. See [library refresh](library-refresh.md) for the subsequent startup scan and compatibility filtering.

Run these commands on **Bender / PowerShell** from the project checkout. Build once after changing UI source or its lockfile, then launch the owner preview:

```powershell
Set-Location 'E:\LoRA Project'
.\tools\Launch-LoRA.ps1 -Build
.\tools\Launch-LoRA.ps1 -Database 'E:\LoRA Project\.local\preview-data\Database\lora_master.db'
```

The build uses the existing Node executable and the project's installed Vite directly. This avoids dependence on a working global npm launcher. Missing dependencies produce an error; the launcher never installs them automatically. A content fingerprint checks the UI inputs and every built file. Backend-only commits do not require a UI rebuild. A changed or missing UI build is refused with a rebuild instruction.

**Preserve `.local\preview-data\Database\lora_master.db`.** Owner-created versions, recipes and experiments persist in that copy. Do not overwrite, delete or recreate it as a disposable test fixture. Include it in backups before any eventual transfer to the main database. Launching this preview has left the original main database unchanged.

Without `-Database`, launch uses `Database\lora_master.db`. Before native API startup can initialize its additive history tables, the server creates and verifies a fresh local-state backup under `.local\backups\before-launch-*`. An occupied port is detected before backup or API startup. The selected database must already exist and pass the existing backup verifier's SQLite integrity check with the catalogue tables present.

For a rehearsal, explicitly choose an existing restored database. This launches against that copy, and the main database is not opened by application startup:

```powershell
Set-Location 'E:\LoRA Project'
.\tools\Launch-LoRA.ps1 -Database 'E:\LoRA Project\.local\your-restored-copy\Database\lora_master.db'
```

Replace that example with the actual restored database path. A copied database is identified as `copy` in the launch output and status response. The launcher does not automatically back up copies. All application writes, including saved profile revisions and recipes, go to the selected database for that process.

Optional Studio measurements launch the fixed project `.venv-analysis\Scripts\python.exe` worker when explicitly requested; the main API remains in its lightweight environment. One bounded job is allowed at a time, and Windows process-tree ownership covers cancellation and timeout. Missing or mismatched optional dependencies produce a useful failure; neither the launcher nor the UI installs them into the app or ComfyUI. The library scanner starts once per server startup and can also be run manually. It writes inventory/header observations only to the selected database, reads model paths/stat/header details without Torch, and never reads tensor payloads. Tests can set `LORA_DISABLE_STARTUP_SCAN=1` before startup to disable the scanner factory entirely; this is a test seam, not the normal owner launch setting.

Check or stop the recorded process on **Bender / PowerShell**:

```powershell
Set-Location 'E:\LoRA Project'
.\tools\Launch-LoRA.ps1 -Status
.\tools\Launch-LoRA.ps1 -Stop
```

`-Stop` checks the recorded process ID, Python executable, creation time, script path and unique run identifier before stopping it. It does not stop arbitrary processes that happen to use port 5187. The local process receipt and stdout/stderr logs remain in `.local\runtime`. If a recorded process is alive but has not answered, launch refuses to replace its receipt: check its log or stop it first. An occupied port belonging to another application is left alone.

`-Port` selects another loopback port, and `-NoBrowser` suppresses opening the browser. Use the same port with `-Status` or `-Stop`. A running instance with an older backend/content fingerprint or different selected database must be stopped before reuse. The local status endpoint `/local-app/status` records the database path, main/copy classification, process/run identity, Git revision, dirty-state flag, backend source fingerprint, UI input fingerprint and pre-start backup path. Git identity is informational; it does not substitute for the content checks.

The main app is still under development. A successful launch proves that the local build and API are running; it does not prove image quality or that a selected ComfyUI checkpoint matches the conditional export target. Profile and recipe history retain their existing fresh-file validation requirements. Backups in `.local` are local recovery copies, so include the project data and these backups in the separate machine/off-machine backup plan documented in [local-state-backup.md](local-state-backup.md).

Offline launcher checks run without a listener or production database:

```powershell
Set-Location 'E:\LoRA Project'
.\.venv\Scripts\python.exe tools\test_serve_local.py
```
