"""Baseline models for the airport congestion forecasting task.

Implements simple heuristic forecasters to establish a performance floor.
All real models must beat these baselines to justify their complexity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "acps",
    timestamp_col: str = "timestamp_hour",
) -> pd.DataFrame:
    """Run all baseline models and return a comparison table.

    Baselines implemented:
        - **prev_hour**: Predict using the value from 1 hour ago.
        - **prev_day_same_hour**: Predict using same hour yesterday.
        - **prev_week_same_hour**: Predict using same hour 7 days ago.
        - **majority_class**: Predict the most common ACPS class from training
          (for the classification variant).

    Args:
        train: Training set DataFrame with target and timestamp columns.
        test: Test set DataFrame with the same schema.
        target_col: Name of the target column.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with one row per baseline, columns:
            - ``model`` (str): Baseline name.
            - ``mae`` (float): Mean Absolute Error on test.
            - ``rmse`` (float): Root Mean Squared Error on test.
    """
    y_true = test[target_col].values

    baselines: dict[str, pd.Series] = {
        "prev_hour": _baseline_prev_hour(train, test, target_col),
        "prev_day_same_hour": _baseline_same_hour_offset(
            train, test, target_col, timestamp_col, offset_hours=24,
        ),
        "prev_week_same_hour": _baseline_same_hour_offset(
            train, test, target_col, timestamp_col, offset_hours=168,
        ),
        "majority_class": _baseline_majority_class(train, target_col, len(test)),
    }

    rows: list[dict[str, object]] = []
    for name, preds in baselines.items():
        preds_arr = preds.values
        mae = mean_absolute_error(y_true, preds_arr)
        rmse = root_mean_squared_error(y_true, preds_arr)
        logger.info("Baseline %-25s | MAE=%.4f  RMSE=%.4f", name, mae, rmse)
        rows.append({"model": name, "mae": mae, "rmse": rmse})

    return pd.DataFrame(rows)


def _baseline_prev_hour(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> pd.Series:
    """Naive forecast: use the previous hour's value.

    Args:
        train: Training data (used to fill first test prediction).
        test: Test data.
        target_col: Target column name.

    Returns:
        Series of predictions aligned to the test index.
    """
    # Shift test target by 1: each prediction = the previous row's actual value.
    # The first prediction has no "previous test row", so use the last train value.
    last_train_value = train[target_col].iloc[-1]
    shifted = test[target_col].shift(1)
    shifted.iloc[0] = last_train_value
    return shifted.reset_index(drop=True)


def _baseline_same_hour_offset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    offset_hours: int,
) -> pd.Series:
    """Forecast using the value from ``offset_hours`` hours ago.

    Args:
        train: Training data.
        test: Test data.
        target_col: Target column name.
        timestamp_col: Timestamp column name.
        offset_hours: How many hours back to look (24 or 168).

    Returns:
        Series of predictions aligned to the test index.
    """
    combined = pd.concat([train, test], ignore_index=True)
    shifted = combined[target_col].shift(offset_hours)

    # Take only the test portion (last len(test) rows)
    test_preds = shifted.iloc[len(train):].reset_index(drop=True)

    # Fill any NaN (where offset reaches before available data) with train mean
    train_mean = train[target_col].mean()
    test_preds = test_preds.fillna(train_mean)

    return test_preds


def _baseline_majority_class(
    train: pd.DataFrame,
    target_col: str,
    n_test: int,
) -> pd.Series:
    """Predict the most frequent target class from training data.

    Args:
        train: Training data.
        target_col: Target column name.
        n_test: Number of test samples.

    Returns:
        Constant Series of length ``n_test`` with the majority class value.
    """
    majority_value = train[target_col].mode().iloc[0]
    return pd.Series([majority_value] * n_test)
