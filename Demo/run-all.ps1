# Madrid-Barajas Congestion Forecast - PowerShell launcher
# Run from the Demo/ folder:  .\run-all.ps1

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- sanity checks ---
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
  Write-Host "[ERROR] Python is not on PATH. Install Python 3.10+ and re-run." -ForegroundColor Red
  exit 1
}
foreach ($f in @("backend\main.py", "frontend\index.html", "models\hgb_regressor_eurocontrol.pkl")) {
  if (-not (Test-Path (Join-Path $Here $f))) {
    Write-Host "[ERROR] Missing $f" -ForegroundColor Red
    exit 1
  }
}

# --- install deps if needed ---
$check = & python -c "import fastapi, uvicorn, joblib, pandas, numpy, sklearn" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "[INFO] Installing Python dependencies ..." -ForegroundColor Yellow
  & python -m pip install -r (Join-Path $Here "backend\requirements.txt")
  if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] pip install failed." -ForegroundColor Red; exit 1 }
}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " Madrid-Barajas Congestion Forecast Console" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " Backend  : http://127.0.0.1:8000   (FastAPI + trained HGB model)"
Write-Host " Frontend : http://127.0.0.1:5173   (web console)"
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Starting backend ..." -ForegroundColor Yellow
Start-Process -FilePath "cmd" -ArgumentList @(
  "/k", "title Barajas Backend (8000) && cd /d `"$Here\backend`" && python -m uvicorn main:app --port 8000 --host 127.0.0.1"
)

Write-Host "[2/3] Waiting 5s for the model to load ..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "[3/3] Starting frontend ..." -ForegroundColor Yellow
Start-Process -FilePath "cmd" -ArgumentList @(
  "/k", "title Barajas Frontend (5173) && cd /d `"$Here\frontend`" && python -m http.server 5173 --bind 127.0.0.1"
)

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Opening browser ..." -ForegroundColor Green
Start-Process "http://127.0.0.1:5173/"

Write-Host ""
Write-Host "Both servers running in separate windows. Close them to stop."
