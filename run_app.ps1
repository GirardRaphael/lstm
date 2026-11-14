# Open the live intersection by default; retain the existing Streamlit applications.
# Extra arguments are forwarded to Streamlit, e.g. -Images --server.port 8502.
param(
    [switch]$Observatory,
    [switch]$Workbench,
    [switch]$Images,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Continue' # TensorFlow writes startup warnings to stderr.
$selectedModes = @($Observatory, $Workbench, $Images) | Where-Object { $_.IsPresent }
if (@($selectedModes).Count -gt 1) {
    throw 'Choose only one of -Observatory, -Workbench or -Images.'
}
if (@($selectedModes).Count -eq 0) {
    if ($args.Count -gt 0) { throw 'Extra server arguments require a Streamlit mode: -Observatory, -Workbench or -Images.' }
    & (Join-Path $PSScriptRoot 'run_intersection.ps1') -NoBrowser:$NoBrowser
    exit $LASTEXITCODE
}
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create .venv with ./setup.ps1 first.' }
$app = if ($Observatory) { 'app/observatory_app.py' } elseif ($Workbench) { 'app/streamlit_app.py' } else { 'app/intersection_images.py' }
$browserOptions = @()
if ($NoBrowser) { $browserOptions = @('--server.headless', 'true') }
& $python -m streamlit run (Join-Path $PSScriptRoot $app) --server.address 127.0.0.1 @browserOptions @args
exit $LASTEXITCODE
