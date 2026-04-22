# Claude Code Team Onboarding Router

> **IMPORTANT FOR CLAUDE CODE:** This file is loaded automatically via CLAUDE.md.
> At the START of every new conversation, you MUST follow the onboarding protocol below
> BEFORE doing any other work. Do not skip this step.

---

## Onboarding Protocol

When a user opens this project in Claude Code for the first time in a session, do the following:

### Step 1: Identify the participant

Ask the user:

```
Welcome to the Airport Congestion Forecasting project!

Before we start, I need to know your role so I can route you to your exact tasks and files.

**Who are you on this team?**

- **C1** - Data Engineering & API Ingestion Lead (Ian)
- **C2** - Feature Engineering & Forecasting Lead
- **C3** - ML, Geospatial & Evaluation Lead

Pick C1, C2, or C3.
```

### Step 2: Check project state

After they answer, immediately:
1. Read `.claude/plans/phase_tracker.md` for current progress and blockers
2. Read `.claude/plans/master_plan.md` for the sprint timeline
3. Check which raw data files exist: `ls data/raw/`
4. Check which intermediate files exist: `ls data/intermediate/`
5. Check which processed files exist: `ls data/processed/`

Report a brief status to the user:
- What data is available
- What blockers exist
- What their next task is based on the plan

### Step 3: Route to role-specific workflow

Follow the role-specific instructions below.

---

## C1 - Data Engineering & API Ingestion Lead

### Your Mission
Build and maintain the entire data collection pipeline. Your code feeds everything else.
C2 and C3 **cannot start** until your raw data is available.

### Your Branch
```bash
git checkout -b feat/data-ingestion  # or switch to it if it exists
```

### Files You Own (DO NOT edit files outside this list)
- `src/data/` (all files)
- `src/config.py`
- `src/utils/` (all files)
- `config/airports.yaml`, `config/paths.yaml`
- `.env.example`
- `notebooks/02_data_collection_opensky.ipynb`
- `notebooks/03_data_collection_weather.ipynb`

### Your Task Queue (in priority order)
1. **Check OpenSky auth** - Run `python -m src.data.fetch_opensky` and verify if historical flights endpoint works (needs credentials in `.env`)
2. **Collect OpenSky data** - Arrivals + departures for LEMD, target 12 months
3. **Collect weather data** - Run `python -m src.data.fetch_open_meteo`
4. **Collect holidays + metadata** - Run `python -m src.data.fetch_holidays` and `python -m src.data.fetch_airport_metadata`
5. **Collect FlightRadarAPI snapshot** - Optional, run `python -m src.data.fetch_flightradar`
6. **Validate everything** - Run `python -m src.data.validate_raw_data`
7. **Update notebooks 02 & 03** - Ensure they document the collection process with outputs visible

### Agent Strategy for C1
When working on C1 tasks, use this agent pattern:
- **Parallel agents** to test multiple API endpoints simultaneously
- **Background agent** to run long data collection while working on validation code
- **Research agent** if OpenSky returns errors: search for alternative endpoints or workarounds

### What C1 Hands Off
When you're done, these must exist and be valid:
- `data/raw/opensky/all_arrivals_LEMD.parquet`
- `data/raw/opensky/all_departures_LEMD.parquet`
- `data/raw/open_meteo/weather_hourly_LEMD.parquet`
- `data/raw/holidays/spain_holidays_*.json`
- `data/raw/ourairports/airport_reference.parquet`

### Acceptance Test
```python
from src.data.validate_raw_data import run_all_validations
report = run_all_validations()
# All sources should show "OK" with zero issues
```

---

## C2 - Feature Engineering & Forecasting Lead

### Your Mission
Transform raw data into features and build the first models (baselines + SARIMAX).
You produce the model-ready table that C3 uses for ML.

### Your Branch
```bash
git checkout -b feat/features-forecasting
```

### Files You Own (DO NOT edit files outside this list)
- `src/features/` (all files - stubs are already there with signatures)
- `src/modeling/baselines.py`
- `src/modeling/sarimax_model.py`
- `notebooks/04_data_cleaning_and_integration.ipynb`
- `notebooks/05_eda_time_series.ipynb`
- `notebooks/07_feature_engineering.ipynb`
- `notebooks/08_baselines_and_sarimax.ipynb`

### Prerequisites (check these first!)
Before you can work, verify C1's data exists:
```python
import pandas as pd
arr = pd.read_parquet("data/raw/opensky/all_arrivals_LEMD.parquet")
dep = pd.read_parquet("data/raw/opensky/all_departures_LEMD.parquet")
weather = pd.read_parquet("data/raw/open_meteo/weather_hourly_LEMD.parquet")
print(f"Arrivals: {len(arr)}, Departures: {len(dep)}, Weather hours: {len(weather)}")
```
If these files don't exist, tell C1 to finish data collection first.

### Your Task Queue (in priority order)
1. **Implement `build_hourly_movements.py`** - Aggregate raw flights into hourly counts. The stub has the function signature ready.
2. **Implement ACPS target** in `build_model_table.py` - Formula: `0.6 * standardized(movements) + 0.4 * standardized(pressure_ratio)`. See `config/modeling.yaml` for weights.
3. **Implement `build_calendar_features.py`** - Use holidays from `data/raw/holidays/`. The stub has `_is_bridge_day` helper ready.
4. **Implement `build_weather_features.py`** - Sin/cos wind encoding, severe weather flags. Stub has helpers.
5. **Implement lag + rolling features** in `build_model_table.py` - Lags: 1,2,3,6,12,24,48,168h. Rolling: mean 3/6/24h, std 24h.
6. **Assemble model table** - `build_model_table.py` merges everything, applies chronological 70/15/15 split.
7. **Implement baselines** in `baselines.py` - Previous hour, same hour prev day, same hour prev week, majority class.
8. **Implement SARIMAX** in `sarimax_model.py` - With weather exogenous variables. Keep it simple, don't over-tune.
9. **Create notebooks 04, 05, 07, 08** - Use the template in `.claude/tools/notebook_template.md`.

### Agent Strategy for C2
- **Research agent** to study the ACPS formula and validate the target makes sense with real data
- **Research agent** to explore SARIMAX parameter choices (use auto_arima if available)
- **Parallel agents** to implement calendar features and weather features simultaneously (they're independent)
- Always check `.claude/tools/data_contracts.md` for exact column schemas before writing outputs

### Data Contracts You Must Follow
Your outputs MUST match these schemas (from `.claude/tools/data_contracts.md`):
- `data/intermediate/hourly_movements.parquet` - columns: timestamp, arrivals, departures, movements, airport_icao
- `data/intermediate/hourly_weather.parquet` - columns: timestamp, temperature_2m, precipitation, wind_speed_10m, wind_dir_sin, wind_dir_cos, surface_pressure, relative_humidity_2m, weather_code, rain_flag, severe_weather_flag
- `data/intermediate/hourly_calendar_features.parquet` - columns: timestamp, hour, day_of_week, is_weekend, month, quarter, is_holiday, is_pre_holiday, is_post_holiday, is_bridge_day
- `data/processed/model_table_hourly.parquet` - all intermediate columns + acps, congestion_class, congestion_binary, lag/rolling features
- `data/processed/train.parquet`, `valid.parquet`, `test.parquet`

### What C2 Hands Off
When you're done, C3 needs:
- `data/processed/train.parquet`, `valid.parquet`, `test.parquet` with all features
- `outputs/tables/baseline_comparison.csv` with baseline metrics
- Feature list documented in notebook 07

### Acceptance Test
```python
import pandas as pd
train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
test = pd.read_parquet("data/processed/test.parquet")
assert "acps" in train.columns
assert "congestion_class" in train.columns
assert len(train) > len(valid) > 0
assert len(test) > 0
print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
print(f"Features: {len(train.columns)} columns")
```

---

## C3 - ML, Geospatial & Evaluation Lead

### Your Mission
Train ML models, build geospatial visualizations, run evaluation, and produce all final outputs.
Your work generates the figures and tables the report/presentation team uses.

### Your Branch
```bash
git checkout -b feat/ml-geo-eval
```

### Files You Own (DO NOT edit files outside this list)
- `src/modeling/tree_models.py`
- `src/modeling/evaluation.py`
- `src/modeling/forecasting_pipeline.py`
- `src/visualization/` (all files)
- `notebooks/06_geospatial_analysis.ipynb`
- `notebooks/09_ml_models.ipynb`
- `notebooks/10_model_evaluation_and_error_analysis.ipynb`
- `notebooks/11_final_visuals_and_tables.ipynb`
- `report/report_outline.md`
- `presentation/slide_outline.md`

### Prerequisites (check these first!)
Before ML work, verify C2's model table exists:
```python
import pandas as pd
train = pd.read_parquet("data/processed/train.parquet")
print(f"Train shape: {train.shape}")
print(f"Target column 'acps' present: {'acps' in train.columns}")
print(f"Features: {[c for c in train.columns if c not in ['timestamp','acps','congestion_class','congestion_binary']]}")
```
For geospatial work, you can start earlier using raw OpenSky data or FlightRadarAPI snapshots.

### Your Task Queue (in priority order)
1. **Implement `evaluation.py`** - `evaluate_regression()`, `evaluate_classification()`, `compare_models()`. Stubs ready.
2. **Implement `tree_models.py`** - HistGradientBoostingRegressor + Classifier. Use params from `config/modeling.yaml`.
3. **Start geospatial (parallel with above)** - `geospatial_plots.py`: airport map with radius bands, flight density hexbin/KDE.
4. **Implement `timeseries_plots.py`** - Daily patterns, seasonal decomposition, ACPS timeseries.
5. **Implement `model_plots.py`** - Model comparison bar charts, feature importance, confusion matrix, error analysis.
6. **Create notebook 06** - Geospatial analysis with folium maps. Use `.claude/reference/airport_params.md` for coordinates/bands.
7. **Create notebook 09** - Train ML models, show training curves, hyperparameter choices.
8. **Create notebook 10** - Full evaluation: comparison table, confusion matrices, error analysis by period.
9. **Create notebook 11** - Export all final figures to `outputs/figures/`, tables to `outputs/tables/`.
10. **Update report outline** - Fill `report/report_outline.md` with actual result references.
11. **Update slide outline** - Fill `presentation/slide_outline.md` with actual figure paths.

### Agent Strategy for C3
- **Research agent** to study geospatial visualization best practices for aviation (folium, geopandas)
- **Research agent** to study SHAP values for HistGradientBoosting interpretation
- **Parallel agents** to build geospatial notebook and ML models simultaneously (independent tracks)
- **Background agent** to generate all output figures while you work on evaluation code

### Key Evaluation Requirements
Regression metrics: MAE, RMSE, R2
Classification metrics: F1 (high-congestion class), precision, recall, balanced accuracy, confusion matrix
Must answer: Which periods are hardest to predict? Does weather matter more in peak vs off-peak?

### What C3 Produces (Final Outputs)
- `outputs/figures/` - All report-ready PNG figures
- `outputs/tables/model_comparison.csv` - Full model comparison
- `outputs/tables/feature_importance.csv` - Top features ranked
- `outputs/models/*.pkl` - Trained model files
- Updated `report/report_outline.md` and `presentation/slide_outline.md`

### Acceptance Test
```python
from pathlib import Path
figures = list(Path("outputs/figures").glob("*.png"))
tables = list(Path("outputs/tables").glob("*.csv"))
models = list(Path("outputs/models").glob("*.pkl"))
print(f"Figures: {len(figures)}, Tables: {len(tables)}, Models: {len(models)}")
assert len(figures) >= 5, "Need at least 5 report-ready figures"
assert len(tables) >= 2, "Need model comparison + feature importance tables"
```

---

## Merge Protocol

When all three participants are done:

1. **C1 merges first** - `feat/data-ingestion` -> main (raw data + configs)
2. **C2 rebases on main, then merges** - `feat/features-forecasting` -> main
3. **C3 rebases on main, then merges** - `feat/ml-geo-eval` -> main
4. **Final cleanup on main** - Run all notebooks top-to-bottom, fix any issues

---

## Universal Rules for All Participants

1. **NEVER edit files you don't own** - Check the file ownership list for your role
2. **Follow data contracts** - Read `.claude/tools/data_contracts.md` before writing any output files
3. **Use src/ modules** - Don't write all code inline in notebooks. Call functions from src/
4. **Leave TODO markers** where human team members need to add interpretation
5. **Test before committing** - Run your notebooks top-to-bottom before pushing
6. **Update the tracker** - After completing a major task, update `.claude/plans/phase_tracker.md`
7. **Check the master plan** - `.claude/plans/master_plan.md` has the 5-day sprint timeline. We have until April 1.
