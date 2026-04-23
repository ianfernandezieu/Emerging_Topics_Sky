@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Madrid-Barajas Congestion Forecast — one-click launcher.
REM Starts the FastAPI backend (port 8000) and a static-file
REM server for the frontend (port 5173), then opens the browser.
REM
REM Requirements: Python 3.10+ on PATH.
REM Run once before first use:  python -m pip install -r backend\requirements.txt
REM ============================================================

set "HERE=%~dp0"

REM --- sanity checks ----------------------------------------------------------

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not on PATH. Install Python 3.10+ and re-run this script.
  pause
  exit /b 1
)

if not exist "%HERE%backend\main.py" (
  echo [ERROR] Missing backend\main.py at %HERE%backend
  pause
  exit /b 1
)

if not exist "%HERE%frontend\index.html" (
  echo [ERROR] Missing frontend\index.html at %HERE%frontend
  pause
  exit /b 1
)

if not exist "%HERE%models\hgb_regressor_eurocontrol.pkl" (
  echo [ERROR] Missing models\hgb_regressor_eurocontrol.pkl
  pause
  exit /b 1
)

REM --- install deps if needed (fast no-op when already satisfied) -------------

python -c "import fastapi, uvicorn, joblib, pandas, numpy, sklearn" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing Python dependencies from backend\requirements.txt ...
  python -m pip install -r "%HERE%backend\requirements.txt"
  if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
  )
)

echo.
echo =============================================================
echo  Madrid-Barajas Congestion Forecast Console
echo =============================================================
echo  Backend  : http://127.0.0.1:8000   (FastAPI + trained HGB model)
echo  Frontend : http://127.0.0.1:5173   (web console)
echo =============================================================
echo.

echo [1/3] Starting backend in a new window ...
start "Barajas Backend (8000)" cmd /k "cd /d "%HERE%backend" && python -m uvicorn main:app --port 8000 --host 127.0.0.1"

echo [2/3] Waiting 5 seconds for the model to load ...
timeout /t 5 /nobreak >nul

echo [3/3] Starting frontend static server in a new window ...
start "Barajas Frontend (5173)" cmd /k "cd /d "%HERE%frontend" && python -m http.server 5173 --bind 127.0.0.1"

timeout /t 2 /nobreak >nul

echo.
echo Opening browser at http://127.0.0.1:5173/ ...
start "" "http://127.0.0.1:5173/"

echo.
echo Both servers are running in separate windows.
echo If the browser shows BACKEND OFFLINE, wait 5 more seconds and press F5.
echo To stop the demo: close the two cmd windows that opened.
echo.
pause
