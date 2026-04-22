"""Process FlightRadar24 API snapshots into density features per radius band.

Converts raw real-time aircraft position snapshots into aggregated density
metrics at configurable radius bands around the airport. These features
capture live airspace congestion that historical schedule data cannot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import AIRPORT_LAT, AIRPORT_LON, ANALYSIS_BANDS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_live_density_features(
    snapshots: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Compute aircraft density features from FlightRadar snapshots.

    For each snapshot timestamp and each configured radius band, counts
    the number of aircraft and computes summary statistics (mean altitude,
    mean speed, altitude variance).

    Args:
        snapshots: Either a list of snapshot dicts (each with ``timestamp``
            and ``aircraft`` keys) or a DataFrame with one row per observed
            aircraft per snapshot, containing at minimum: ``timestamp``,
            ``latitude``, ``longitude``, ``altitude``, ``ground_speed``.

    Returns:
        DataFrame indexed by ``timestamp`` with columns per band, e.g.:
            - ``density_0_50nm`` (int): Aircraft count within 0-50 NM.
            - ``density_50_100nm`` (int): Aircraft count within 50-100 NM.
            - ``mean_alt_0_50nm`` (float): Mean altitude in band.
            - ``mean_speed_0_50nm`` (float): Mean ground speed in band.
            - ``alt_std_0_50nm`` (float): Altitude std dev in band.
            - ``total_density`` (int): Total aircraft across all bands.
    """
    raise NotImplementedError("C2/C3 to implement")


def _haversine_nm(
    lat1: float,
    lon1: float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance in nautical miles using the Haversine formula.

    Args:
        lat1: Reference latitude (airport) in degrees.
        lon1: Reference longitude (airport) in degrees.
        lat2: Aircraft latitude(s) in degrees.
        lon2: Aircraft longitude(s) in degrees.

    Returns:
        Distance(s) in nautical miles.
    """
    raise NotImplementedError("C2/C3 to implement")


def _assign_band(
    distances_nm: np.ndarray,
    bands: list[dict[str, Any]],
) -> np.ndarray:
    """Assign each distance to a radius band label.

    Args:
        distances_nm: Array of distances in nautical miles.
        bands: List of band dicts from config, each with
            ``inner_nm``, ``outer_nm``, ``label``.

    Returns:
        Array of band label strings (or ``"outside"`` if beyond all bands).
    """
    raise NotImplementedError("C2/C3 to implement")
