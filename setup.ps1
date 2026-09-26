# Reproducible local setup. Credentials are initialized separately in your terminal.
$ErrorActionPreference = 'Continue'
if (!(Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv, then rerun setup.ps1' }
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    uv pip sync requirements.lock
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host 'Environment ready. Run ./observatory.ps1 init to create your access token, then ./run_app.ps1.'
} finally { Pop-Location }
