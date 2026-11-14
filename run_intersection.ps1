# Start the intersection and its model service. Ctrl+C stops both processes.
param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$demoDirectory = Join-Path $PSScriptRoot 'presentation'
$runtimeDirectory = Join-Path $PSScriptRoot '.runtime'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = "$env:USERPROFILE\.venvs\traffic_lstm\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw 'Python environment missing. Run setup.ps1 to create .venv from requirements.lock first.'
}
$node = (Get-Command node.exe -ErrorAction Stop).Source
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
foreach ($demoPort in @(43125, 43126)) {
    $probe = [System.Net.Sockets.TcpClient]::new()
    try {
        $probe.Connect('127.0.0.1', $demoPort)
        throw "Port $demoPort is already in use. Close the existing demo before starting this copy."
    } catch [System.Net.Sockets.SocketException] {
        # An unused port is expected.
    } finally { $probe.Dispose() }
}
if (-not (Test-Path "$demoDirectory\node_modules\vite\bin\vite.js")) {
    Push-Location $demoDirectory
    try {
        & $npm ci --ignore-scripts --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Demo dependency installation failed.' }
    } finally { Pop-Location }
}
New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
$modelProcess = $null
$webProcess = $null
try {
    $modelProcess = Start-Process -FilePath $python -ArgumentList '-u server/sidecar.py' -WorkingDirectory $demoDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtimeDirectory\intersection-model.log" -RedirectStandardError "$runtimeDirectory\intersection-model-error.log"
    $webProcess = Start-Process -FilePath $node -ArgumentList 'node_modules/vite/bin/vite.js --host 127.0.0.1 --port 43125' -WorkingDirectory $demoDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runtimeDirectory\intersection-web.log" -RedirectStandardError "$runtimeDirectory\intersection-web-error.log"
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($modelProcess.HasExited -or $webProcess.HasExited) {
            throw "Demo startup failed. See logs in $runtimeDirectory."
        }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:43125/api/health' -TimeoutSec 1
            if ($health.ok) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw "Model service did not become ready. See logs in $runtimeDirectory." }
    Write-Host 'Live intersection: http://127.0.0.1:43125/  (Ctrl+C to stop)'
    if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:43125/' }
    while (-not $webProcess.HasExited -and -not $modelProcess.HasExited) {
        Start-Sleep -Seconds 1
    }
    throw "A demo service stopped. See logs in $runtimeDirectory."
} finally {
    foreach ($demoProcess in @($webProcess, $modelProcess)) {
        if ($null -ne $demoProcess -and -not $demoProcess.HasExited) {
            Stop-Process -Id $demoProcess.Id -ErrorAction SilentlyContinue
        }
    }
}
