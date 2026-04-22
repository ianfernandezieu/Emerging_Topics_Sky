"""HistGradientBoosting regressor and classifier for congestion forecasting.

Uses scikit-learn's HistGradientBoosting models which natively handle
missing values and are fast on medium-sized tabular datasets. Provides
both regression (continuous ACPS) and classification (congestion level)
variants.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_tree_models(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "acps",
    n_bins: int = 5,
) -> dict[str, Any]:
    """Train HistGradientBoosting regressor and classifier.

    The regressor predicts continuous ACPS. The classifier predicts a
    discretised congestion level (``target_col`` binned into ``n_bins``
    quantile-based classes).

    Args:
        train: Training DataFrame.
        valid: Validation DataFrame (used for early stopping).
        feature_cols: List of feature column names.
        target_col: Continuous target column name.
        n_bins: Number of quantile bins for the classification target.

    Returns:
        Dict with keys:
            - ``regressor``: Fitted HistGradientBoostingRegressor.
            - ``classifier``: Fitted HistGradientBoostingClassifier.
            - ``feature_cols``: Feature column list (for inference).
            - ``bin_edges``: Quantile bin edges used for classification.
            - ``reg_valid_score``: Regressor R^2 on validation set.
            - ``clf_valid_score``: Classifier accuracy on validation set.
    """
    import numpy as np

    results: dict[str, Any] = {
        "regressor": None,
        "classifier": None,
        "feature_cols": feature_cols,
        "bin_edges": None,
        "reg_valid_score": None,
        "clf_valid_score": None,
        "reg_predictions": None,
        "cls_predictions": None,
    }

    X_train = train[feature_cols]
    y_train_reg = train[target_col]
    X_valid = valid[feature_cols]
    y_valid_reg = valid[target_col]

    # --- Regressor ---
    try:
        regressor = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            min_samples_leaf=2,
        )
        regressor.fit(X_train, y_train_reg)
        results["regressor"] = regressor
        results["reg_valid_score"] = regressor.score(X_valid, y_valid_reg)
        results["reg_predictions"] = pd.Series(
            regressor.predict(X_valid), index=valid.index
        )
        logger.info(
            "Regressor trained. Validation R²: %.4f", results["reg_valid_score"]
        )
    except Exception as e:
        logger.error("Failed to train regressor: %s", e)

    # --- Classifier ---
    try:
        # Determine classification labels
        if "congestion_class" in train.columns:
            y_train_cls = train["congestion_class"].astype(str)
            y_valid_cls = valid["congestion_class"].astype(str)
            results["bin_edges"] = None  # Using pre-defined classes
        else:
            # Bin the target into n_bins quantile-based classes
            y_train_cls, bin_edges = pd.qcut(
                y_train_reg, q=n_bins, labels=False, retbins=True, duplicates="drop"
            )
            y_valid_cls = pd.cut(
                y_valid_reg, bins=bin_edges, labels=False, include_lowest=True
            )
            results["bin_edges"] = bin_edges

        classifier = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            min_samples_leaf=2,
        )
        classifier.fit(X_train, y_train_cls)
        results["classifier"] = classifier
        results["clf_valid_score"] = classifier.score(X_valid, y_valid_cls)
        results["cls_predictions"] = pd.Series(
            classifier.predict(X_valid), index=valid.index
        )
        logger.info(
            "Classifier trained. Validation accuracy: %.4f",
            results["clf_valid_score"],
        )
    except Exception as e:
        logger.error("Failed to train classifier: %s", e)

    return results


def predict_regression(
    model: HistGradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Generate continuous ACPS predictions.

    Args:
        model: Fitted regressor.
        df: DataFrame with feature columns.
        feature_cols: Ordered list of feature column names.

    Returns:
        Series of predicted ACPS values.
    """
    preds = model.predict(df[feature_cols])
    return pd.Series(preds, index=df.index, name="acps_pred")


def predict_classification(
    model: HistGradientBoostingClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Generate congestion-level class predictions.

    Args:
        model: Fitted classifier.
        df: DataFrame with feature columns.
        feature_cols: Ordered list of feature column names.

    Returns:
        Series of predicted congestion-level classes (int 0..n_bins-1).
    """
    preds = model.predict(df[feature_cols])
    return pd.Series(preds, index=df.index, name="congestion_pred")


def get_feature_importance(
    model: HistGradientBoostingRegressor | HistGradientBoostingClassifier,
    feature_cols: list[str],
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    """Extract and sort feature importances from a fitted tree model.

    Uses ``feature_importances_`` if available, otherwise falls back to
    permutation importance (requires X and y).

    Args:
        model: Fitted HistGradientBoosting model.
        feature_cols: Feature column names matching the training order.
        X: Feature DataFrame for permutation importance fallback.
        y: Target series for permutation importance fallback.

    Returns:
        DataFrame with columns ``feature`` and ``importance``, sorted
        descending by importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X[feature_cols], y,
            n_repeats=10, random_state=42, n_jobs=-1,
        )
        importances = result.importances_mean
    else:
        # Last resort: uniform importance (all features equally unknown)
        logger.warning(
            "No feature_importances_ available and no X/y for permutation importance. "
            "Returning uniform importances."
        )
        importances = [1.0 / len(feature_cols)] * len(feature_cols)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    return importance_df
