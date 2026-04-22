"""Evaluation metrics and model comparison utilities.

Provides standardised metric computation for both regression and classification
tasks, plus a comparison function to rank all models in a single table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute regression metrics for ACPS prediction.

    Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error, guarded against division by zero)
        - R-squared

    Args:
        y_true: Ground-truth continuous target values.
        y_pred: Predicted continuous target values.

    Returns:
        Dict mapping metric names to float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Guard against empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays passed to evaluate_regression")
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }

    logger.info("Regression metrics - MAE: %.4f, RMSE: %.4f, R²: %.4f", mae, rmse, r2)
    return metrics


def evaluate_classification(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute classification metrics for congestion-level prediction.

    Metrics:
        - Accuracy
        - Weighted F1 score
        - Per-class precision, recall, F1 (as a dict)
        - Confusion matrix (as a 2D list)

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Dict with keys ``accuracy``, ``f1_weighted``, ``classification_report``
        (dict), and ``confusion_matrix`` (list of lists).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Guard against empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays passed to evaluate_classification")
        return {
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "classification_report": "",
            "confusion_matrix": np.array([]),
        }

    acc = accuracy_score(y_true, y_pred)

    # Guard against single-class predictions: use zero_division to avoid warnings
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_str = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "classification_report": report_str,
        "confusion_matrix": cm,
    }

    logger.info(
        "Classification metrics - Accuracy: %.4f, F1 macro: %.4f", acc, f1_macro
    )
    return metrics


def compare_models(results_dict: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from multiple model evaluation results.

    Args:
        results_dict: Mapping of model name to its evaluation metrics dict.
            Example::

                {
                    "SARIMAX": {"mae": 3.2, "rmse": 4.1, "r2": 0.78},
                    "HistGBR": {"mae": 2.8, "rmse": 3.5, "r2": 0.84},
                    "prev_hour": {"mae": 5.0, "rmse": 6.2, "r2": 0.55},
                }

    Returns:
        DataFrame with one row per model, sorted by MAE ascending.
        Columns are the union of all metric keys across models.
    """
    if not results_dict:
        logger.warning("Empty results_dict passed to compare_models")
        return pd.DataFrame()

    rows = []
    for model_name, metrics in results_dict.items():
        row = {"model": model_name}
        # Only include scalar metrics (skip arrays, strings, etc.)
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by MAE ascending if the column exists
    if "mae" in df.columns:
        df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    logger.info("Model comparison table built with %d models", len(df))
    return df
