[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Build,
    [switch]$NoBrowser,
    [string]$Database,
    [ValidateRange(1024, 65535)][int]$Port = 5187
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$serverPath = Join-Path $PSScriptRoot 'serve_local.py'
$runtimePath = Join-Path $projectRoot '.local\runtime'
$receiptPath = Join-Path $runtimePath "lora-$Port-process.json"
$appUrl = "http://127.0.0.1:$Port"
$explicitDatabase = $PSBoundParameters.ContainsKey('Database')
if (@($Stop, $Status, $Build | Where-Object { $_ }).Count -gt 1) { throw 'Choose only one of -Stop, -Status or -Build.' }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'The project Python environment is missing. Restore it before launching; nothing was installed.' }
if ($Build) {
    & $pythonPath $serverPath build
    if ($LASTEXITCODE -ne 0) { throw 'UI build failed; the app was not started.' }
    return
}
function Read-Receipt {
    if (Test-Path -LiteralPath $receiptPath -PathType Leaf) { return Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json }
    return $null
}
function Assert-OwnedProcess($receipt) {
    if (-not $receipt -or $receipt.app -ne 'lora-comfy-combiner-local' -or $receipt.port -ne $Port) { throw 'No matching app process receipt exists. No process was stopped.' }
    $process = Get-Process -Id $receipt.process_id -ErrorAction SilentlyContinue
    if (-not $process) { return $null }
    $command = Get-CimInstance Win32_Process -Filter "ProcessId = $($receipt.process_id)"
    if ($process.Path -ne $receipt.executable_path -or $process.StartTime.ToUniversalTime().Ticks.ToString() -ne $receipt.start_ticks -or
        $receipt.script -ne $serverPath -or -not $command.CommandLine.Contains($serverPath) -or
        -not $command.CommandLine.Contains("--run-id $($receipt.run_id)")) {
        throw 'The recorded process identity no longer matches. No process was stopped.'
    }
    return $process
}
function Read-Health {
    try { return Invoke-RestMethod "$appUrl/local-app/status" -TimeoutSec 2 }
    catch { return $null }
}
function Assert-Health($health, $receipt) {
    if (-not $health -or $health.app -ne 'lora-comfy-combiner-local' -or $health.run_id -ne $receipt.run_id -or
        $health.pid -ne $receipt.process_id -or $health.port -ne $Port -or $health.host -ne '127.0.0.1' -or
        $health.database_path -ne $receipt.database_path) { throw 'The listener does not match the recorded app and database. No process was stopped.' }
}
$receipt = Read-Receipt
if ($Stop) {
    $owned = Assert-OwnedProcess $receipt
    if ($explicitDatabase -and [IO.Path]::GetFullPath($Database) -ne $receipt.database_path) { throw 'The requested database differs from the recorded app. No process was stopped.' }
    if ($owned) { $owned.Kill(); $owned.WaitForExit(5000) | Out-Null; Write-Output "Stopped the recorded LoRA app on port $Port." }
    else { Write-Output 'The recorded LoRA app is already stopped.' }
    return
}
if ($Status) {
    $owned = Assert-OwnedProcess $receipt
    if (-not $owned) { Write-Output 'The recorded LoRA app is stopped.'; return }
    $health = Read-Health
    Assert-Health $health $receipt
    $health | ConvertTo-Json -Depth 8
    return
}
if (-not $explicitDatabase) { $Database = Join-Path $projectRoot 'Database\lora_master.db' }
if (-not (Test-Path -LiteralPath $Database -PathType Leaf)) { throw 'Choose an existing application database. A new empty database will not be created.' }
$databasePath = (Resolve-Path -LiteralPath $Database).Path
$preflightText = & $pythonPath $serverPath check --database $databasePath
if ($LASTEXITCODE -ne 0) { throw 'Launch checks failed; no app was started.' }
$preflight = $preflightText | ConvertFrom-Json
$existing = Read-Health
if ($existing) {
    $owned = Assert-OwnedProcess $receipt
    Assert-Health $existing $receipt
    if ($existing.database_path -ne $databasePath -or $existing.source.backend_sha256 -ne $preflight.source.backend_sha256 -or
        $existing.build.input_sha256 -ne $preflight.build_input_sha256) { throw 'A different database or older app build is running on this port. Stop the recorded app, then launch again.' }
    Write-Output "LoRA app is already ready: $appUrl"
    if (-not $NoBrowser) { Start-Process $appUrl }
    return
}
if ($receipt) {
    $owned = Assert-OwnedProcess $receipt
    if ($owned) { throw 'The recorded app is still running but has not answered its status check. Check its log or stop that app before retrying.' }
}
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$runId = [guid]::NewGuid().ToString()
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$stdout = Join-Path $runtimePath "lora-$Port-$stamp-out.log"
$stderr = Join-Path $runtimePath "lora-$Port-$stamp-error.log"
$arguments = '"{0}" serve --database "{1}" --port {2} --run-id {3}' -f $serverPath, $databasePath, $Port, $runId
$started = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$receipt = [ordered]@{ app = 'lora-comfy-combiner-local'; process_id = $started.Id; executable_path = $started.Path; start_ticks = $started.StartTime.ToUniversalTime().Ticks.ToString(); run_id = $runId; script = $serverPath; database_path = $databasePath; port = $Port; stdout = $stdout; stderr = $stderr }
$receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    $started.Refresh()
    if ($started.HasExited) { throw "The app stopped during startup. Read $stderr" }
    $health = Read-Health
    if ($health) {
        # Windows venv Python can redirect into a child process. Adopt only
        # the fresh run's responding server, and record its actual identity.
        if ($health.app -ne $receipt.app -or $health.run_id -ne $runId -or $health.database_path -ne $databasePath -or $health.port -ne $Port) { throw 'Another listener answered during startup. Its process was left untouched.' }
        $serverProcess = Get-Process -Id $health.pid
        $serverCommand = Get-CimInstance Win32_Process -Filter "ProcessId = $($health.pid)"
        if (-not $serverCommand.CommandLine.Contains($serverPath) -or -not $serverCommand.CommandLine.Contains("--run-id $runId")) { throw 'The responding process does not match this launch. It was left untouched.' }
        $receipt.process_id = $serverProcess.Id
        $receipt.executable_path = $serverProcess.Path
        $receipt.start_ticks = $serverProcess.StartTime.ToUniversalTime().Ticks.ToString()
        $receipt | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding UTF8
        Assert-Health $health $receipt
        Write-Output "LoRA app ready: $appUrl"
        Write-Output "Database ($($health.database_kind)): $($health.database_path)"
        Write-Output "Source: $($health.source.git_revision) | UI fingerprint: $($health.build.input_sha256)"
        if ($health.pre_start_backup) { Write-Output "Pre-start backup: $($health.pre_start_backup)" }
        if (-not $NoBrowser) { Start-Process $appUrl }
        return
    }
    Start-Sleep -Milliseconds 250
}
throw "The recorded app is still starting or failed to respond. Check -Status and $stderr before retrying."
