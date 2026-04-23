# Madrid-Barajas Congestion Forecast — Live Demo

A live web console backed by a trained HistGradientBoosting model that predicts
daily airport congestion at Adolfo Suárez Madrid-Barajas (LEMD). Every number
shown in the UI is computed on demand against the held-out test set
(**498 days**, 2024-10-19 → 2026-02-28). Nothing is hard-coded.

## What's inside

```
Demo/
├── backend/          FastAPI server (Python)
│   ├── main.py       9 endpoints; loads the trained HGB regressor + classifier
│   └── requirements.txt
├── frontend/         Static React app (no build step, no npm install)
│   ├── index.html
│   ├── src/          api.js + 3 JSX files (pages, components, shell)
│   └── styles/tokens.css
├── models/           Trained model artifacts (.pkl)
├── data/             Eurocontrol model table + feature importance + comparison metrics
├── run-all.bat       One-click launcher (Windows)
├── run-all.ps1       Same, for PowerShell
└── README.md         (this file)
```

## Requirements

- **Python 3.10 or newer** (on `PATH`)
- A modern web browser (Chrome / Firefox / Edge)
- Internet connection on first load (React + Babel are pulled from a CDN)

## Quick start — Windows

1. Open a terminal inside the `Demo/` folder.
2. Install Python dependencies once:

   ```
   python -m pip install -r backend\requirements.txt
   ```

3. Double-click **`run-all.bat`** (or run it from the terminal).
4. A browser window opens at **http://127.0.0.1:5173/**.
5. To stop: close the two `cmd` windows that were opened.

## Quick start — macOS / Linux / manual

Open two terminals inside the `Demo/` folder.

**Terminal 1 — backend (port 8000)**
```
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — frontend static server (port 5173)**
```
cd frontend
python -m http.server 5173 --bind 127.0.0.1
```

Open **http://127.0.0.1:5173/** in your browser.

## What you will see

Four tabs, all driven by the same trained model:

| Tab | What it shows |
|---|---|
| **TODAY** | Current ACPS + 3-day forecast with 95% confidence band |
| **DIAGNOSTICS** | Model showdown (HGB vs. baselines) · 5 residual plots · per-day feature deep-dive |
| **CASE STUDY** | Pick any of the 498 test days → actual vs. predicted with context chart |
| **REPORT CARD** | Held-out metrics (MAE 0.45 · R² 0.96 · Acc 90.2 %) + confusion matrix |

Keyboard shortcuts: `1`–`4` switch tabs · `← →` step days in Diagnostics / Case Study.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Banner says **BACKEND OFFLINE** | Backend still loading | Wait 5 s and press `F5` |
| `ModuleNotFoundError: fastapi` | Dependencies not installed | `python -m pip install -r backend\requirements.txt` |
| `Address already in use :8000` | Something else uses the port | Change `--port` to `8001` in `run-all.bat` (and `API_BASE` in `frontend/src/api.js`) |
| Page is blank | Browser cached an old file | `Ctrl+Shift+R` (hard refresh) |

## Architecture notes

- **Backend**: single-file FastAPI app (`backend/main.py`). Loads both `.pkl` models
  at startup and pre-computes the 498-day residual table and training-set feature
  statistics so every request is O(1) lookups or a single `model.predict()` call.
- **Frontend**: static HTML + in-browser Babel for JSX. No build step, no `npm`,
  no bundler. The whole UI is ~1,000 lines of React across three `.jsx` files.
- **API**: REST, JSON; CORS enabled for `127.0.0.1:5173`. Every endpoint is idempotent.
  Main endpoints: `/api/today`, `/api/metrics`, `/api/test-set/{date}`,
  `/api/model-showdown/{date}`, `/api/residuals`, `/api/local-features/{date}`.

## Data provenance

- **Eurocontrol IFR traffic** (Network Manager counts) — 3,318 daily rows for LEMD, 2017-01-29 → 2026-02-28.
- **Open-Meteo** hourly weather aggregated to daily for Madrid-Barajas coordinates.
- **Nager.Date** Spanish public-holiday calendar (used for `is_holiday`, `is_bridge_day`, etc.).

The composite target — **Airport Congestion Pressure Score (ACPS)** on a 0–100 scale —
is defined as `rescale_0_100(0.6 · z(total_movements) + 0.4 · z(pressure_ratio))`.
