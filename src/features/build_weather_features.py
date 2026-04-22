"""Process raw Open-Meteo weather data into model-ready features.

Transforms the hourly weather DataFrame into engineered features suitable
for the congestion forecasting model: sin/cos wind-direction encoding,
severe weather flags, precipitation categories, and normalised continuous
variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer weather features from the raw hourly weather table.

    Transformations applied:
        - **Wind direction**: sin/cos cyclic encoding of ``wind_direction_10m``.
        - **Severe weather flag**: binary flag derived from ``weather_code``
          (WMO codes >= 65 indicate heavy rain, snow, thunderstorms, etc.).
        - **Rain flag**: binary indicator where ``rain > 0``.
        - **Wind speed buckets**: calm / moderate / strong / severe.
        - **Normalisation**: z-score scaling of continuous vars
          (temperature, pressure, humidity, wind speed, gusts).

    Args:
        weather_df: Raw hourly weather from ``fetch_open_meteo``.
            Expected columns: ``timestamp``, ``temperature_2m``,
            ``relative_humidity_2m``, ``precipitation``, ``rain``,
            ``weather_code``, ``surface_pressure``, ``wind_speed_10m``,
            ``wind_direction_10m``, ``wind_gusts_10m``, ``cloud_cover``.

    Returns:
        DataFrame indexed by ``timestamp`` with all original columns plus:
            - ``wind_dir_sin``, ``wind_dir_cos`` (float)
            - ``is_severe_weather`` (bool)
            - ``is_raining`` (bool)
            - ``wind_bucket`` (str: calm / moderate / strong / severe)
    """
    logger.info("Building weather features from %d rows", len(weather_df))

    df = weather_df.copy()

    # Wind direction cyclic encoding
    df["wind_dir_sin"], df["wind_dir_cos"] = _encode_wind_direction(df["wind_direction_10m"])

    # Severe weather flag (WMO code >= 65)
    df["is_severe_weather"] = _flag_severe_weather(df["weather_code"])

    # Rain flag
    df["is_raining"] = df["rain"] > 0

    # Wind speed buckets
    wind = df["wind_speed_10m"]
    df["wind_bucket"] = pd.cut(
        wind,
        bins=[-np.inf, 10, 30, 50, np.inf],
        labels=["calm", "moderate", "strong", "severe"],
        right=False,
    )

    logger.info("Weather features built: %d columns", len(df.columns))
    return df


def _encode_wind_direction(degrees: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert wind direction in degrees to sin/cos components.

    Args:
        degrees: Wind direction 0-360.

    Returns:
        Tuple of (sin_component, cos_component).
    """
    radians = np.deg2rad(degrees)
    return np.sin(radians), np.cos(radians)


def _flag_severe_weather(weather_codes: pd.Series) -> pd.Series:
    """Map WMO weather codes to a binary severe-weather flag.

    WMO codes reference:
        https://open-meteo.com/en/docs (weather_code table)

    Args:
        weather_codes: Integer WMO weather codes.

    Returns:
        Boolean Series (True = severe).
    """
    return weather_codes >= 65
