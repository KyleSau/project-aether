# Project Aether one-command launcher (Windows PowerShell).
#   .\start.ps1            build (if needed) and serve app + API at http://localhost:8300
#   .\start.ps1 -Rebuild   force frontend rebuild and data precompute
param([switch]$Rebuild, [int]$Port = 8300)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if (-not (Test-Path .venv\Scripts\python.exe)) {
  Write-Host 'Creating Python venv...'
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}
if ($Rebuild -or -not (Test-Path data\thermo.json)) {
  Write-Host 'Running offline precompute (Cantera, plume ROM, reference trajectory)...'
  .\.venv\Scripts\python.exe backend\precompute.py
}
Push-Location frontend
if (-not (Test-Path node_modules)) { npm install }
if ($Rebuild -or -not (Test-Path dist\index.html)) { npm run build }
Pop-Location
Write-Host "Serving Project Aether at http://localhost:$Port"
Start-Process "http://localhost:$Port"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --port $Port
