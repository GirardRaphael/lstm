$ErrorActionPreference = 'Continue' # TensorFlow writes startup warnings to stderr
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
& $python (Join-Path $PSScriptRoot 'tests/test_pipeline.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m pytest (Join-Path $PSScriptRoot 'tests') -q @args
exit $LASTEXITCODE
