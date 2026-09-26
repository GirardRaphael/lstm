# Open archived educational notebooks; current evidence is reports/REPORT.md.
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
& $python -m jupyterlab (Join-Path $PSScriptRoot 'notebooks') @args
exit $LASTEXITCODE
