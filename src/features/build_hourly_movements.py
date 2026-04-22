"""Aggregate raw OpenSky arrivals and departures into hourly movement counts.

Takes the raw flight-level DataFrames from fetch_opensky and produces a single
hourly time series with columns for arrival count, departure count, and total
movements. This is the primary input signal for the ACPS congestion metric.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_hourly_movements(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate raw flight records into hourly movement counts.

    Adapted for FR24 board data which uses ``scheduled_arrival`` and
    ``scheduled_departure`` Unix-timestamp columns (instead of OpenSky's
    ``firstSeen`` / ``lastSeen``).

    Steps:
        1. Parse Unix timestamps from the relevant column.
        2. Floor each timestamp to the hour.
        3. Count arrivals and departures per hour.
        4. Merge into a single DataFrame indexed by ``timestamp_hour``.
        5. Fill gaps (hours with zero movements) in the date range.

    Args:
        arrivals_df: Raw arrivals from FR24 board (one row per flight).
            Expected column: ``scheduled_arrival`` (Unix timestamp).
        departures_df: Raw departures from FR24 board (one row per flight).
            Expected column: ``scheduled_departure`` (Unix timestamp).

    Returns:
        DataFrame with columns:
            - ``timestamp_hour`` (datetime64[ns, UTC]): Hour bucket.
            - ``arrivals`` (int): Number of arriving flights.
            - ``departures`` (int): Number of departing flights.
            - ``total_movements`` (int): arrivals + departures.
    """
    logger.info("Building hourly movements from FR24 board data")

    arr_counts = _count_by_hour(arrivals_df, "scheduled_arrival", "arrivals")
    dep_counts = _count_by_hour(departures_df, "scheduled_departure", "departures")

    # Merge arrivals and departures on timestamp_hour
    merged = pd.merge(
        arr_counts, dep_counts, left_index=True, right_index=True, how="outer"
    )
    merged = merged.fillna(0).astype(int)

    # Fill gaps: create a complete hourly range and reindex
    full_range = pd.date_range(
        start=merged.index.min(),
        end=merged.index.max(),
        freq="h",
        tz="UTC",
    )
    merged = merged.reindex(full_range, fill_value=0)
    merged.index.name = "timestamp_hour"

    merged["total_movements"] = merged["arrivals"] + merged["departures"]

    logger.info(
        "Hourly movements built: %d hours, %d total flights",
        len(merged),
        merged["total_movements"].sum(),
    )
    return merged.reset_index()


def _count_by_hour(
    df: pd.DataFrame,
    time_col: str,
    label: str,
) -> pd.Series:
    """Count records per floored hour from a Unix-timestamp column.

    Args:
        df: Flight DataFrame.
        time_col: Name of the Unix-timestamp column (e.g. ``scheduled_arrival``).
        label: Name for the resulting count series (e.g. ``arrivals``).

    Returns:
        Series indexed by ``timestamp_hour`` with counts.
    """
    # Convert Unix timestamp (seconds) to UTC datetime and floor to hour
    dt = pd.to_datetime(df[time_col], unit="s", utc=True).dt.floor("h")
    counts = dt.value_counts().sort_index()
    counts.index.name = "timestamp_hour"
    counts.name = label
    return counts.to_frame()
