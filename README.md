# Forecasting Airport Congestion at Madrid-Barajas

**University Group Project** - Predicting next-hour and next-3-hour airport congestion pressure at Adolfo Suarez Madrid-Barajas Airport (MAD/LEMD) using flight tracking, weather, and calendar signals.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your OpenSky credentials (optional but recommended)

# 4. Collect data
make data

# 5. Validate data
make validate
```

## Project Structure

```
├── config/              # YAML configuration files
├── data/
│   ├── raw/             # Raw API outputs
│   ├── intermediate/    # Cleaned hourly tables
│   └── processed/       # Model-ready tables
├── notebooks/           # Numbered Jupyter notebooks (run in order)
│   ├── 01_project_scope_and_data_sources.ipynb
│   ├── 02_data_collection_opensky.ipynb
│   ├── 03_data_collection_weather.ipynb
│   ├── 04_data_cleaning_and_integration.ipynb
│   ├── 05_eda_time_series.ipynb
│   ├── 06_geospatial_analysis.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_baselines_and_sarimax.ipynb
│   ├── 09_ml_models.ipynb
│   ├── 10_model_evaluation_and_error_analysis.ipynb
│   └── 11_final_visuals_and_tables.ipynb
├── src/                 # Python source package
│   ├── config.py        # Central configuration
│   ├── data/            # Data fetchers
│   ├── features/        # Feature engineering
│   ├── modeling/        # Models and evaluation
│   └── visualization/   # Plot generators
├── research/            # Literature and context notes
├── outputs/             # Figures, tables, models
├── report/              # Report outline and drafts
└── presentation/        # Slide outline
```

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| OpenSky Network | Historical API | Arrivals/departures (core training data) |
| Open-Meteo | Historical API | Hourly weather variables |
| Nager.Date | REST API | Spanish public holidays |
| OurAirports | Static CSV | Airport metadata/coordinates |
| FlightRadarAPI | Live wrapper | Optional real-time enrichment |

## Notebook Execution Order

Run notebooks 01-11 in order. Each notebook documents what it does and why.

## Team

7 participants: 3 with Claude Code (data pipeline, features, ML/geo), 4 handling research, validation, report, and presentation.

## License

Academic project - not for commercial use.
