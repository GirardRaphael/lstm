# Verify the claims: no leakage, correct sequence alignment, exact LSTM replay.
$venv = "$env:USERPROFILE\.venvs\traffic_lstm\Scripts"
& "$venv\python.exe" "$PSScriptRoot\tests\test_pipeline.py"
