.PHONY: setup data validate features models report-assets clean

# Environment setup
setup:
	python -m venv venv
	./venv/Scripts/pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Setup complete. Edit .env with your credentials."

# Data collection
data:
	python -m src.data.fetch_opensky
	python -m src.data.fetch_open_meteo
	python -m src.data.fetch_holidays
	python -m src.data.fetch_airport_metadata
	python -m src.data.fetch_flightradar || echo "FlightRadarAPI skipped (optional)"

# Validate raw data
validate:
	python -m src.data.validate_raw_data

# Feature engineering (C2 implements)
features:
	python -m src.features.build_hourly_movements
	python -m src.features.build_weather_features
	python -m src.features.build_calendar_features
	python -m src.features.build_model_table

# Model training (C2/C3 implement)
models:
	python -m src.modeling.baselines
	python -m src.modeling.sarimax_model
	python -m src.modeling.tree_models

# Export report assets
report-assets:
	@echo "Exporting figures and tables to outputs/"
	@echo "Run notebook 11 for final exports"

# Clean generated data (use with caution)
clean:
	rm -rf data/intermediate/*
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf outputs/tables/*
	rm -rf outputs/models/*
