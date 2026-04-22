# Airport Congestion Forecasting - Project CLAUDE.md

## Project Identity
- **Title:** Forecasting Airport Congestion at Madrid-Barajas Using Flight Tracking, Weather, and Calendar Signals
- **Type:** University group project (7 participants, 3 with Claude Code)
- **Airport:** Adolfo Suarez Madrid-Barajas (IATA: MAD, ICAO: LEMD)
- **Primary question:** Can we forecast airport congestion pressure 1h and 3h ahead using flight, weather, and calendar signals?
- **Deadline:** April 1, 2026

## What's Built (Complete Pipeline)
The entire data-to-model pipeline runs in a single notebook: `notebooks/airport_congestion_pipeline.ipynb`

### Pipeline Flow
```
Part 1: Data Collection (cells 1-10)
  → FR24 airport board (1260 arr + 1246 dep, ~36h)
  → Open-Meteo weather (9408 hours, 12 months)
  → Spanish holidays (33 dates)
  → Airport metadata + live aircraft positions

Part 2: Feature Engineering & Baselines (cells 11-22)
  → ACPS target (congestion pressure score)
  → Calendar features (hour, dow, holiday, cyclic encoding)
  → Weather features (wind sin/cos, severe weather, rain flags)
  → Lag/rolling features (1-12h lags, 3h/6h rolling means)
  → Baselines (prev hour, prev day, training mean, hourly avg)
  → SARIMAX with weather exogenous vars

Part 3: ML Models & Evaluation (cells 23-32)
  → HistGradientBoosting regressor (MAE: 0.51, R2: 0.38)
  → HistGradientBoosting classifier (100% on 5 test samples)
  → Feature importance (movements dominant)
  → Geospatial maps (aircraft positions, origin airports)
  → Final model comparison table
```

### Output Files
- `outputs/figures/` - 20+ report-ready PNG figures + interactive HTML map
- `outputs/tables/` - model_comparison.csv, feature_importance.csv, baseline_comparison.csv
- `outputs/models/` - hgb_regressor.pkl, hgb_classifier.pkl
- `data/processed/` - train.parquet, valid.parquet, test.parquet

## Data Sources
1. **FlightRadar24** (unofficial wrapper) - airport board arrivals/departures (PRIMARY flight data)
2. **Open-Meteo** - 12 months hourly weather (CORE)
3. **Nager.Date** - Spanish holidays (CORE)
4. **OurAirports** - airport metadata (CORE)
5. **OpenSky** - live aircraft state vectors (geospatial enrichment)

**Note:** OpenSky historical flights endpoint requires research-tier access (returns 403). FR24 board provides ~36h real flight data. This limitation is documented honestly.

## Team Structure
- **C1 (Ian):** Data Engineering & API Ingestion - DONE
- **C2:** Feature Engineering & Forecasting - DONE (implemented in notebook + src/)
- **C3:** ML, Geospatial & Evaluation - DONE (implemented in notebook + src/)
- **H1-H4:** Research, validation, report, presentation - templates ready in research/, report/, presentation/

## For Teammates Using Claude Code
**Read `ONBOARDING.md` first.** It routes you to your exact tasks based on your role.
**Read `TEAM_GUIDE.md`** for a full explanation of the project aimed at all 7 team members.

## Architecture
```
notebooks/                → Single unified pipeline notebook
src/data/                 → 6 data fetcher modules (all implemented)
src/features/             → 5 feature engineering modules (all implemented)
src/modeling/             → 5 modeling modules (all implemented)
src/visualization/        → 3 visualization modules (stubs - notebook has inline plots)
config/                   → YAML configs (airports, paths, modeling params)
data/raw/                 → Cached API data
data/processed/           → Model-ready train/valid/test splits
outputs/                  → Figures, tables, trained models
research/                 → Templates for H1/H2
report/                   → Report outline for H3
presentation/             → Slide outline for H4
```

## Coding Standards
- Python 3.10, type hints, docstrings on public functions
- Graceful error handling on all API calls
- Config-driven paths and parameters (YAML files)
- `.env` for secrets, `.env.example` committed
- All plots call `plt.close('all')` after `plt.show()` to prevent popups

## Kernel Setup
```bash
python -m venv venv
pip install -r requirements.txt
python -m ipykernel install --user --name airport-congestion --display-name "Airport Congestion (Python 3.10)"
```
