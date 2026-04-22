"""Open-Meteo historical weather data fetcher.

Fetches hourly weather data for the airport location.
No API key needed - free and open access.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from pathlib import Path

import pandas as pd
import requests

from src.config import (
    AIRPORT_LAT,
    AIRPORT_LON,
    AIRPORT_ICAO,
    DATA_END_DATE,
    DATA_START_DATE,
    get_path,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_parquet

logger = get_logger(__name__)

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly weather variables to fetch
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
]


def fetch_weather(
    latitude: float = AIRPORT_LAT,
    longitude: float = AIRPORT_LON,
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    output_dir: str | Path | None = None,
    airport_icao: str = AIRPORT_ICAO,
) -> pd.DataFrame:
    """Fetch hourly historical weather data from Open-Meteo.

    Args:
        latitude: Airport latitude.
        longitude: Airport longitude.
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
        output_dir: Output directory. Defaults to config path.
        airport_icao: Airport code for filename.

    Returns:
        DataFrame with hourly weather data.
    """
    if output_dir is None:
        output_dir = Path(get_path("raw_weather"))
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / f"weather_hourly_{airport_icao}.parquet"

    # Check cache
    if output_path.exists():
        logger.info(f"Weather data already cached: {output_path}")
        return pd.read_parquet(output_path)

    logger.info(
        f"Fetching weather for ({latitude}, {longitude}): {start_date} to {end_date}"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch weather data: {e}")
        raise

    data = response.json()

    if "hourly" not in data:
        logger.error(f"Unexpected response format: {list(data.keys())}")
        raise ValueError("Open-Meteo response missing 'hourly' key")

    hourly = data["hourly"]

    # Build DataFrame
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"time": "timestamp"})

    # Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    logger.info(
        f"Weather data: {len(df)} hours, "
        f"{df['timestamp'].min()} to {df['timestamp'].max()}, "
        f"{len(df.columns)} variables"
    )

    save_parquet(df, output_path, "hourly weather")

    return df


def load_weather(airport_icao: str = AIRPORT_ICAO) -> pd.DataFrame:
    """Load cached weather data.

    Returns:
        DataFrame with hourly weather data.
    """
    raw_dir = Path(get_path("raw_weather"))
    filepath = raw_dir / f"weather_hourly_{airport_icao}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Weather data not found at {filepath}. Run fetch_weather() first."
        )

    return pd.read_parquet(filepath)


if __name__ == "__main__":
    df = fetch_weather()
    logger.info(f"Done. Shape: {df.shape}")
    print(df.head())
    print(df.dtypes)
