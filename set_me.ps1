$ErrorActionPreference = 'Stop'

if ($PSScriptRoot) {
    Set-Location $PSScriptRoot
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    throw "Python is not installed or not on PATH."
}

$venvPath = Join-Path (Get-Location) 'venv'

& $pythonCmd.Source -m venv $venvPath

$venvPython = Join-Path $venvPath 'Scripts\python.exe'

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path (Get-Location) 'requirements.txt')

Write-Host "Environment setup complete. Activate with:"
Write-Host "  .\\venv\\Scripts\\Activate.ps1"
