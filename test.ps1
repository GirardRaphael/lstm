# Verify the claims: no leakage, correct sequence alignment, exact LSTM replay.
$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$legacyPython = Join-Path $env:USERPROFILE ".venvs\traffic_lstm\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { $pythonExe = $legacyPython }
if (-not (Test-Path $pythonExe)) {
    throw "Python environment not found. Run: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
& $pythonExe "$PSScriptRoot\tests\test_pipeline.py"
exit $LASTEXITCODE
