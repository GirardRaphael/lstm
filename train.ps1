# Retrain the model, rebuild the figures and regenerate the Obsidian vault.
$venv = "$env:USERPROFILE\.venvs\traffic_lstm\Scripts"
$env:PYTHONPATH = "$PSScriptRoot\src"
& "$venv\python.exe" -m traffic_lstm.train --export-vault @args
