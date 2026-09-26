$ErrorActionPreference = 'Continue'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
& $python -m traffic_lstm.operations @args
exit $LASTEXITCODE
