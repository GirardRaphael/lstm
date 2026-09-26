# New research runs use the causal pipeline. Run name is mandatory.
$ErrorActionPreference = 'Continue' # native stderr warnings are not failed exit codes
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
& $python -m traffic_lstm.research @args
exit $LASTEXITCODE
