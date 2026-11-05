# Setup script for Windows (PowerShell)
# Usage: .\setup.ps1 [-Launch]
# If scripts are blocked, run:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
param([switch]$Launch)

$ErrorActionPreference = "Stop"
$VenvDir = "venv"
$MinMajor = 3
$MinMinor = 9

Write-Host "==> Detected Windows"

# Find Python (py launcher preferred, then python)
$PyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) { $PyCmd = "py" }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $PyCmd = "python" }
else {
    Write-Host "Python not found. Install it from https://www.python.org/downloads/ or run: winget install Python.Python.3.12"
    exit 1
}

# Check version
& $PyCmd -c "import sys; sys.exit(0 if sys.version_info >= ($MinMajor, $MinMinor) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python $MinMajor.$MinMinor+ is required."
    exit 1
}
Write-Host "==> Using $(& $PyCmd --version)"

# Create virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating virtual environment in .\$VenvDir"
    & $PyCmd -m venv $VenvDir
} else {
    Write-Host "==> Virtual environment already exists, reusing it"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "==> Upgrading pip"
& $VenvPython -m pip install --upgrade pip

Write-Host "==> Installing dependencies"
& $VenvPython -m pip install -r requirements.txt

Write-Host "==> Registering Jupyter kernel"
& $VenvPython -m ipykernel install --user --name machine-learning-models --display-name "Python (ML Models)"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate the environment later with: .\$VenvDir\Scripts\Activate.ps1"

if ($Launch) {
    Write-Host "==> Launching Jupyter Notebook"
    & $VenvPython -m notebook
} else {
    Write-Host "Start notebooks with: jupyter notebook"
}
