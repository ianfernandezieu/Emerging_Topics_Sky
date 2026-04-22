"""Build calendar and temporal features for hourly timestamps.

Generates deterministic calendar features that capture recurring temporal
patterns: hour-of-day, day-of-week, weekend indicators, Spanish public
holidays, pre/post-holiday effects, and bridge-day flags.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_calendar_features(
    timestamps: pd.DatetimeIndex | pd.Series,
    holiday_dates: list[date] | set[date],
) -> pd.DataFrame:
    """Create calendar features for each timestamp.

    Features produced:
        - ``hour`` (int 0-23): Hour of day.
        - ``dow`` (int 0-6): Day of week (Monday=0).
        - ``is_weekend`` (bool): Saturday or Sunday.
        - ``is_holiday`` (bool): Date is in ``holiday_dates``.
        - ``is_pre_holiday`` (bool): Day before a holiday.
        - ``is_post_holiday`` (bool): Day after a holiday.
        - ``is_bridge_day`` (bool): Working day between a holiday and a
          weekend (or vice versa) that people commonly take off.
        - ``hour_sin``, ``hour_cos`` (float): Cyclic encoding of hour.
        - ``dow_sin``, ``dow_cos`` (float): Cyclic encoding of day of week.

    Args:
        timestamps: Hourly timestamps to generate features for.
            Must be timezone-aware (UTC).
        holiday_dates: Collection of ``date`` objects representing
            public holidays (e.g., from ``fetch_holidays``).

    Returns:
        DataFrame indexed the same as ``timestamps`` with one column
        per feature listed above.
    """
    logger.info("Building calendar features for %d timestamps", len(timestamps))

    if isinstance(timestamps, pd.DatetimeIndex):
        ts = timestamps
    else:
        ts = pd.DatetimeIndex(timestamps)

    holiday_set = set(holiday_dates)

    df = pd.DataFrame(index=ts)
    df["hour"] = ts.hour
    df["dow"] = ts.dayofweek  # Monday=0
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["month"] = ts.month
    df["quarter"] = ts.quarter

    # Date-level features (computed once per unique date, then mapped)
    dates = pd.Series(ts.date, index=ts)
    df["is_holiday"] = dates.isin(holiday_set)
    df["is_pre_holiday"] = dates.apply(lambda d: (d + timedelta(days=1)) in holiday_set)
    df["is_post_holiday"] = dates.apply(lambda d: (d - timedelta(days=1)) in holiday_set)
    df["is_bridge_day"] = dates.apply(lambda d: _is_bridge_day(d, holiday_set))

    # Cyclic encodings
    df["hour_sin"], df["hour_cos"] = _cyclic_encode(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = _cyclic_encode(df["dow"], 7)

    logger.info("Calendar features built: %d columns", len(df.columns))
    return df


def _is_bridge_day(
    date_val: date,
    holiday_set: set[date],
) -> bool:
    """Determine whether a single date qualifies as a bridge day.

    A bridge day is a non-holiday weekday that sits between a holiday
    and a weekend (or between two holidays), making it a common day
    for workers to take off.

    Args:
        date_val: The date to check.
        holiday_set: Set of known holiday dates.

    Returns:
        True if the date is a bridge day.
    """
    # Must be a weekday and not a holiday itself
    if date_val.weekday() >= 5 or date_val in holiday_set:
        return False

    prev_day = date_val - timedelta(days=1)
    next_day = date_val + timedelta(days=1)

    prev_is_non_working = prev_day in holiday_set or prev_day.weekday() >= 5
    next_is_non_working = next_day in holiday_set or next_day.weekday() >= 5

    # Bridge day: a workday sitting between two non-working days
    # (e.g., holiday on Thursday, Friday is bridge to weekend;
    #  or Monday bridge between weekend and Tuesday holiday)
    return prev_is_non_working and next_is_non_working


def _cyclic_encode(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Encode an integer cyclic variable as sin/cos components.

    Args:
        values: Integer values (e.g. hour 0-23, dow 0-6).
        period: Cycle length (e.g. 24 for hours, 7 for days).

    Returns:
        Tuple of (sin_component, cos_component).
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)
