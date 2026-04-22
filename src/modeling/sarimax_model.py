"""SARIMAX model for airport congestion forecasting with weather exogenous variables.

Fits a Seasonal ARIMA with eXogenous regressors using statsmodels.
Designed for hourly ACPS prediction with weather features as external drivers.
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default SARIMAX order — to be tuned via grid search or auto_arima
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 24)  # 24-hour seasonality


def train_sarimax(
    train: pd.DataFrame,
    exog_cols: list[str],
    target_col: str = "acps",
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
) -> SARIMAXResultsWrapper | None:
    """Fit a SARIMAX model on the training data.

    Args:
        train: Training DataFrame sorted by time, containing the target
            and exogenous columns.
        exog_cols: List of column names to use as exogenous regressors
            (e.g. weather features: ``temperature_2m``, ``wind_speed_10m``,
            ``is_severe_weather``).
        target_col: Name of the endogenous target column.
        order: ARIMA (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order. ``s=24`` for hourly data.

    Returns:
        Fitted SARIMAX results object, or None if fitting fails entirely.
    """
    endog = train[target_col]
    exog = train[exog_cols] if exog_cols else None

    # Attempt 1: full SARIMAX with seasonal component
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False, maxiter=200)
            logger.info(
                "SARIMAX%s x %s fitted — AIC=%.2f",
                order, seasonal_order, results.aic,
            )
            return results
    except Exception as exc:
        logger.warning(
            "SARIMAX seasonal fit failed (%s). Falling back to non-seasonal ARIMA%s.",
            exc, order,
        )

    # Attempt 2: fall back to ARIMA without seasonal component
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False, maxiter=200)
            logger.info(
                "ARIMA%s (non-seasonal fallback) fitted — AIC=%.2f",
                order, results.aic,
            )
            return results
    except Exception as exc:
        logger.error("ARIMA fallback also failed: %s. Returning None.", exc)
        return None


def forecast_sarimax(
    results: SARIMAXResultsWrapper,
    steps: int,
    exog_future: pd.DataFrame | None = None,
) -> pd.Series:
    """Generate out-of-sample forecasts from a fitted SARIMAX model.

    Args:
        results: Fitted SARIMAX results from ``train_sarimax``.
        steps: Number of hours to forecast.
        exog_future: Future exogenous values for the forecast horizon.
            Must have ``steps`` rows and the same columns used in training.

    Returns:
        Series of forecasted ACPS values.
    """
    forecast = results.forecast(steps=steps, exog=exog_future)
    return pd.Series(forecast.values, name="forecast")


def grid_search_sarimax(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    exog_cols: list[str],
    target_col: str = "acps",
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any] | None:
    """Grid search over SARIMAX hyperparameters using validation performance.

    Args:
        train: Training set.
        valid: Validation set.
        exog_cols: Exogenous column names.
        target_col: Target column name.
        param_grid: Dict mapping parameter names to lists of values to try.
            Defaults to a small grid of common (p,d,q) combinations.

    Returns:
        Dict with keys ``best_order``, ``best_seasonal_order``, ``best_aic``,
        ``best_results``, and ``all_results`` (list of dicts).
        Returns None if every combination fails.
    """
    if param_grid is None:
        param_grid = {
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1, 2],
        }

    p_values = param_grid.get("p", [1])
    d_values = param_grid.get("d", [1])
    q_values = param_grid.get("q", [1])

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_results = None
    all_results: list[dict[str, Any]] = []

    total_combos = len(p_values) * len(d_values) * len(q_values)
    logger.info("Grid search: testing %d (p,d,q) combinations.", total_combos)

    for p, d, q in product(p_values, d_values, q_values):
        order = (p, d, q)

        # Try with seasonal component first, then without
        for s_order in [DEFAULT_SEASONAL_ORDER, (0, 0, 0, 0)]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = train_sarimax(
                        train,
                        exog_cols,
                        target_col=target_col,
                        order=order,
                        seasonal_order=s_order,
                    )

                    if results is None:
                        continue

                    aic = results.aic
                    entry = {
                        "order": order,
                        "seasonal_order": s_order,
                        "aic": aic,
                    }
                    all_results.append(entry)

                    logger.info(
                        "  order=%s seasonal=%s => AIC=%.2f", order, s_order, aic,
                    )

                    if aic < best_aic:
                        best_aic = aic
                        best_order = order
                        best_seasonal_order = s_order
                        best_results = results

                    # If seasonal succeeded, no need to try non-seasonal for this order
                    break

            except Exception as exc:
                logger.warning(
                    "  order=%s seasonal=%s failed: %s", order, s_order, exc,
                )
                continue

    if best_results is None:
        logger.error(
            "Grid search: all %d configurations failed. Returning None.",
            total_combos,
        )
        return None

    logger.info(
        "Grid search best: order=%s seasonal=%s AIC=%.2f",
        best_order, best_seasonal_order, best_aic,
    )

    return {
        "best_order": best_order,
        "best_seasonal_order": best_seasonal_order,
        "best_aic": best_aic,
        "best_results": best_results,
        "all_results": all_results,
    }
