# Open the presentation notebook.
$venv = "$env:USERPROFILE\.venvs\traffic_lstm\Scripts"
$env:PYTHONPATH = "$PSScriptRoot\src"
& "$venv\python.exe" -m jupyterlab "$PSScriptRoot\notebooks"
