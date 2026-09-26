$ErrorActionPreference = 'Continue'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first' }
& $python -m streamlit run (Join-Path $PSScriptRoot 'app/observatory_app.py') --server.address 127.0.0.1 @args
exit $LASTEXITCODE
