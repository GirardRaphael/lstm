# Launch the interactive workbench.
$venv = "$env:USERPROFILE\.venvs\traffic_lstm\Scripts"
& "$venv\python.exe" -m streamlit run "$PSScriptRoot\app\streamlit_app.py"
