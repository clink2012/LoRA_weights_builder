param([switch]$Stop, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$serverPath = Join-Path $PSScriptRoot 'server.py'
$runtimeRoot = Join-Path $projectRoot '.local\gui-questionnaire'
$receiptPath = Join-Path $runtimeRoot 'process.json'
$url = 'http://127.0.0.1:5186'

if ($Stop) {
    if (-not (Test-Path -LiteralPath $receiptPath)) { Write-Output 'No questionnaire process receipt found.'; return }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($receipt.pid)"
    $runningProcess = Get-Process -Id $receipt.pid -ErrorAction SilentlyContinue
    $scriptArgument = '(?:^|\s)"?' + [regex]::Escape($serverPath) + '"?(?=\s|$)'
    if ($processInfo -and $runningProcess -and $receipt.python -and $receipt.process_start_ticks -and
        $processInfo.ExecutablePath -eq $receipt.python -and
        $runningProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $receipt.process_start_ticks -and
        $processInfo.CommandLine -match $scriptArgument) {
        Stop-Process -Id $receipt.pid
        Write-Output 'Questionnaire stopped. Saved answers are retained.'
    } elseif ($processInfo) { throw 'The recorded process is not the questionnaire; it was not stopped.' }
    else { Write-Output 'Questionnaire is already stopped. Saved answers are retained.' }
    return
}

$running = $false
try {
    $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
    if ($health.app -ne 'lora-gui-questionnaire') { throw 'Another service is using the questionnaire port.' }
    $running = $true
} catch {
    $listener = Get-NetTCPConnection -State Listen -LocalPort 5186 -ErrorAction SilentlyContinue
    if ($listener) { throw 'Port 5186 is in use by a different or unresponsive service. Nothing was changed.' }
}
if (-not $running) {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        $pythonPath = (Get-Command python -ErrorAction Stop).Source
    }
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    $child = Start-Process -FilePath $pythonPath -ArgumentList @(('"{0}"' -f $serverPath), '--port', '5186') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeRoot 'server.log') -RedirectStandardError (Join-Path $runtimeRoot 'server-errors.log') -PassThru
    @{ pid = $child.Id; url = $url; server = $serverPath; python = $pythonPath; process_start_ticks = $child.StartTime.ToUniversalTime().Ticks.ToString(); started_at = $child.StartTime.ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $receiptPath -Encoding utf8
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 200
        try { $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 1; if ($health.app -eq 'lora-gui-questionnaire') { $running = $true; break } } catch { }
        if ($child.HasExited) { break }
    }
    if (-not $running) { throw "Questionnaire did not start. See $runtimeRoot\server-errors.log" }
}
Write-Output "Questionnaire ready: $url"
if (-not $NoBrowser) { Start-Process $url }
