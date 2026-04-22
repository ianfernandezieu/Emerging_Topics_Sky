"""Central configuration loader for the airport congestion forecasting project.

Loads settings from YAML config files and environment variables.
All modules should import paths and parameters from here.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root (two levels up from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load all configs
_paths_config = _load_yaml("paths.yaml")
_airport_config = _load_yaml("airports.yaml")
_modeling_config = _load_yaml("modeling.yaml")

# --- Path Configuration ---
PATHS = {key: str(PROJECT_ROOT / value) for key, value in _paths_config.items()}

# --- Airport Configuration ---
AIRPORT = _airport_config["primary_airport"]
AIRPORT_ICAO = AIRPORT["icao"]
AIRPORT_IATA = AIRPORT["iata"]
AIRPORT_LAT = AIRPORT["latitude"]
AIRPORT_LON = AIRPORT["longitude"]
AIRPORT_TZ = AIRPORT["timezone"]
ANALYSIS_BANDS = _airport_config["analysis_bands"]
BOUNDING_BOX = _airport_config["bounding_box"]

# --- Modeling Configuration ---
MODELING = _modeling_config

# --- Environment Variables ---
OPENSKY_USERNAME = os.getenv("OPENSKY_USERNAME", "")
OPENSKY_PASSWORD = os.getenv("OPENSKY_PASSWORD", "")
FLIGHTRADAR_ENABLED = os.getenv("FLIGHTRADAR_ENABLED", "true").lower() == "true"
DATA_START_DATE = os.getenv("DATA_START_DATE", "2025-03-01")
DATA_END_DATE = os.getenv("DATA_END_DATE", "2026-03-27")


def get_path(key: str) -> Path:
    """Get a resolved Path object for a configured path key."""
    if key not in PATHS:
        raise KeyError(f"Unknown path key: {key}. Available: {list(PATHS.keys())}")
    p = Path(PATHS[key])
    p.mkdir(parents=True, exist_ok=True)
    return p
