"""Assemble all feature sources into the final model table.

Joins hourly movements, weather features, calendar features, and live
density features into a single table. Computes the ACPS (Airport Congestion
Pressure Score) target, adds lag/rolling features, and produces
train/validation/test splits.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELING, get_path
from src.features.build_calendar_features import build_calendar_features
from src.features.build_weather_features import build_weather_features
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_model_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the complete model table and split into train/valid/test.

    Pipeline:
        1. Load hourly movements, weather features, calendar features,
           and (optionally) live density features.
        2. Join all on ``timestamp_hour``.
        3. Compute the **ACPS target** (Airport Congestion Pressure Score).
        4. Add lag features (1h, 2h, 3h, 6h, 12h, 24h, 168h).
        5. Add rolling-window features (mean/std over 3h, 6h, 24h).
        6. Drop rows with NaN introduced by lagging.
        7. Split chronologically into train / validation / test sets
           using ratios from ``config/modeling.yaml``.

    Returns:
        Tuple of (train_df, valid_df, test_df), each a DataFrame with
        all feature columns and the target column ``acps``.
    """
    logger.info("Building model table")

    # --- 1. Load hourly movements ---
    movements_path = get_path("intermediate") / "hourly_movements.parquet"
    movements_df = pd.read_parquet(movements_path)
    # Standardise column name: the parquet may use 'timestamp' or 'timestamp_hour'
    if "timestamp" in movements_df.columns and "timestamp_hour" not in movements_df.columns:
        movements_df = movements_df.rename(columns={"timestamp": "timestamp_hour"})
    # Rename 'movements' -> 'total_movements' if needed
    if "movements" in movements_df.columns and "total_movements" not in movements_df.columns:
        movements_df = movements_df.rename(columns={"movements": "total_movements"})
    logger.info("Loaded %d hourly movement rows", len(movements_df))

    # --- 2. Load weather features ---
    weather_path = get_path("raw_weather") / "weather_hourly_LEMD.parquet"
    weather_raw = pd.read_parquet(weather_path)
    weather_df = build_weather_features(weather_raw)
    # Standardise timestamp column name
    if "timestamp" in weather_df.columns and "timestamp_hour" not in weather_df.columns:
        weather_df = weather_df.rename(columns={"timestamp": "timestamp_hour"})
    logger.info("Loaded and engineered weather features: %d rows", len(weather_df))

    # --- 3. Load holidays and build calendar features ---
    holiday_dir = get_path("raw_holidays")
    holiday_dates: set[date] = set()
    for hfile in sorted(holiday_dir.glob("*.json")):
        with open(hfile, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            # Only include global holidays (relevant to Madrid / all of Spain)
            if rec.get("global", True):
                d = pd.Timestamp(rec["date"]).date()
                holiday_dates.add(d)
    logger.info("Loaded %d global holiday dates", len(holiday_dates))

    calendar_df = build_calendar_features(
        movements_df["timestamp_hour"], holiday_dates
    )
    # Reset index so timestamp_hour becomes a regular column (not both index and column)
    calendar_df = calendar_df.reset_index(drop=True)
    calendar_df["timestamp_hour"] = movements_df["timestamp_hour"].values
    logger.info("Calendar features built: %d rows", len(calendar_df))

    # --- 4. Merge all on timestamp_hour ---
    # Ensure merge keys are the same dtype
    movements_df["timestamp_hour"] = pd.to_datetime(movements_df["timestamp_hour"], utc=True)
    weather_df["timestamp_hour"] = pd.to_datetime(weather_df["timestamp_hour"], utc=True)
    calendar_df["timestamp_hour"] = pd.to_datetime(calendar_df["timestamp_hour"], utc=True)

    model_df = movements_df.merge(weather_df, on="timestamp_hour", how="left")
    model_df = model_df.merge(calendar_df, on="timestamp_hour", how="left")
    logger.info("Merged table: %d rows, %d columns", len(model_df), len(model_df.columns))

    # --- 5. Compute ACPS target ---
    model_df["acps"] = compute_acps(model_df)
    logger.info("ACPS target computed")

    # --- 6. Add lag features ---
    lag_hours = MODELING["lag_hours"]
    model_df = _add_lag_features(model_df, "acps", lag_hours)

    # --- 7. Add rolling features ---
    rolling_cfg = MODELING["rolling_windows"]
    # Combine mean and std window sizes into a single set of windows
    mean_windows = rolling_cfg.get("mean", [])
    std_windows = rolling_cfg.get("std", [])
    all_windows = sorted(set(mean_windows + std_windows))
    model_df = _add_rolling_features(model_df, "acps", all_windows,
                                     mean_windows=mean_windows,
                                     std_windows=std_windows)

    # --- 8. Drop NaN rows introduced by lagging ---
    # Only drop based on lag/rolling columns that can actually be filled
    # given the data size. Lags larger than the dataset are left as NaN.
    n_rows = len(model_df)
    feasible_lag_cols = [
        f"acps_lag_{h}h" for h in lag_hours if h < n_rows
    ]
    feasible_rolling_cols = [
        col for col in model_df.columns
        if ("_rmean_" in col or "_rstd_" in col)
    ]
    drop_subset = feasible_lag_cols + feasible_rolling_cols
    # Also include any non-lag/rolling columns that have NaN from the merge
    before_drop = len(model_df)
    if drop_subset:
        model_df = model_df.dropna(subset=drop_subset).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN (lag/rolling), %d remain",
                before_drop - len(model_df), len(model_df))

    # Warn about lags that exceed available data
    infeasible_lags = [h for h in lag_hours if h >= n_rows]
    if infeasible_lags:
        logger.warning(
            "Lags %s exceed data length (%d rows) -- these columns will be all NaN",
            infeasible_lags, n_rows,
        )

    # --- 9. Chronological split ---
    train_ratio = MODELING["split"]["train"]
    valid_ratio = MODELING["split"]["validation"]
    train_df, valid_df, test_df = _split_chronological(model_df, train_ratio, valid_ratio)
    logger.info(
        "Split: train=%d, valid=%d, test=%d",
        len(train_df), len(valid_df), len(test_df),
    )

    # --- 10. Save to processed ---
    processed_dir = get_path("processed")
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    valid_df.to_parquet(processed_dir / "valid.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    model_df.to_parquet(processed_dir / "model_table.parquet", index=False)
    logger.info("Saved splits to %s", processed_dir)

    return train_df, valid_df, test_df


def compute_acps(movements_df: pd.DataFrame) -> pd.Series:
    """Compute the Airport Congestion Pressure Score for each hour.

    ACPS is a normalised composite metric derived from total movements
    relative to the airport's declared hourly capacity and weighted by
    the arrival/departure balance.

    Args:
        movements_df: DataFrame with ``arrivals``, ``departures``, and
            ``total_movements`` columns.

    Returns:
        Series of ACPS values (float, 0-100 scale) aligned to the input index.
    """
    weights = MODELING["target"]
    w_movement = weights["acps_movement_weight"]   # 0.6
    w_pressure = weights["acps_pressure_weight"]    # 0.4

    total = movements_df["total_movements"].astype(float)

    # Compute pressure_ratio: movements / max(1, median for same hour+dow)
    # Build median lookup by (hour, dow)
    temp = movements_df[["total_movements"]].copy()
    if "hour" in movements_df.columns:
        temp["hour"] = movements_df["hour"]
        temp["dow"] = movements_df["dow"]
    else:
        # Derive from timestamp_hour
        ts = pd.to_datetime(movements_df["timestamp_hour"], utc=True)
        temp["hour"] = ts.dt.hour
        temp["dow"] = ts.dt.dayofweek

    medians = temp.groupby(["hour", "dow"])["total_movements"].transform("median")
    pressure_ratio = total / np.maximum(1.0, medians)

    # Standardise both components (z-score)
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    z_movements = _zscore(total)
    z_pressure = _zscore(pressure_ratio)

    # Weighted combination
    acps_raw = w_movement * z_movements + w_pressure * z_pressure

    # Rescale to 0-100 using min-max
    acps_min = acps_raw.min()
    acps_max = acps_raw.max()
    if acps_max == acps_min:
        acps = pd.Series(50.0, index=movements_df.index)
    else:
        acps = (acps_raw - acps_min) / (acps_max - acps_min) * 100.0

    return acps


def _add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of the target column.

    Args:
        df: Model table sorted by timestamp.
        target_col: Column to lag (e.g. ``acps``).
        lags: List of lag periods in hours (e.g. [1, 2, 3, 6, 12, 24, 168]).

    Returns:
        DataFrame with new columns ``{target_col}_lag_{h}h`` for each lag.
    """
    df = df.copy()
    for h in lags:
        df[f"{target_col}_lag_{h}h"] = df[target_col].shift(h)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    mean_windows: list[int] | None = None,
    std_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std features for the target column.

    Args:
        df: Model table sorted by timestamp.
        target_col: Column to compute rolling stats on.
        windows: List of all window sizes in hours (union of mean and std).
        mean_windows: Windows for which to compute rolling mean.
            Defaults to all windows if None.
        std_windows: Windows for which to compute rolling std.
            Defaults to all windows if None.

    Returns:
        DataFrame with new columns ``{target_col}_rmean_{w}h`` and
        ``{target_col}_rstd_{w}h`` for applicable windows.
    """
    if mean_windows is None:
        mean_windows = windows
    if std_windows is None:
        std_windows = windows

    df = df.copy()
    for w in windows:
        rolling = df[target_col].rolling(window=w, min_periods=1)
        if w in mean_windows:
            df[f"{target_col}_rmean_{w}h"] = rolling.mean()
        if w in std_windows:
            df[f"{target_col}_rstd_{w}h"] = rolling.std()
    return df


def _split_chronological(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/valid/test sets.

    Args:
        df: Full model table, sorted by time.
        train_ratio: Fraction for training (e.g. 0.7).
        valid_ratio: Fraction for validation (e.g. 0.15).
            Test gets the remainder.

    Returns:
        Tuple of (train, valid, test) DataFrames.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train = df.iloc[:train_end].reset_index(drop=True)
    valid = df.iloc[train_end:valid_end].reset_index(drop=True)
    test = df.iloc[valid_end:].reset_index(drop=True)

    return train, valid, test
