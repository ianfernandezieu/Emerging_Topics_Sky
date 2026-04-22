"""End-to-end forecasting pipeline for airport congestion prediction.

Orchestrates the full workflow: feature assembly, model training, evaluation,
and comparison. Designed to be run as a single entry point for experiments.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_path
from src.modeling.evaluation import compare_models, evaluate_regression
from src.modeling.tree_models import predict_regression, train_tree_models
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Columns that are NOT features (metadata / targets)
_META_COLS = {
    "timestamp",
    "airport_icao",
    "acps",
    "congestion_class",
    "congestion_binary",
}


def _get_feature_cols(df: pd.DataFrame, target_col: str = "acps") -> list[str]:
    """Derive numeric feature columns from a DataFrame, excluding meta/target cols."""
    exclude = _META_COLS | {target_col}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols


def run_pipeline(
    retrain: bool = True,
    include_baselines: bool = True,
    include_sarimax: bool = True,
    include_tree: bool = True,
) -> dict[str, Any]:
    """Execute the full forecasting pipeline.

    Steps:
        1. Build the model table (calls ``build_model_table``).
        2. Run baseline models on train/test split.
        3. Fit SARIMAX with weather exogenous variables.
        4. Train HistGradientBoosting regressor and classifier.
        5. Evaluate all models and produce comparison table.
        6. Save results and artefacts to the configured output directory.

    Args:
        retrain: If True, retrain all models from scratch. If False,
            attempt to load cached model artefacts.
        include_baselines: Whether to run baseline models.
        include_sarimax: Whether to fit the SARIMAX model.
        include_tree: Whether to train tree-based models.

    Returns:
        Dict with keys:
            - ``comparison_table`` (pd.DataFrame): Side-by-side model metrics.
            - ``predictions`` (dict[str, pd.Series]): Model name -> predictions.
            - ``models`` (dict[str, Any]): Model name -> fitted model object.
            - ``train`` (pd.DataFrame): Training split used.
            - ``valid`` (pd.DataFrame): Validation split used.
            - ``test`` (pd.DataFrame): Test split used.
    """
    target_col = "acps"
    processed_dir = get_path("processed")
    model_dir = get_path("models")

    # --- 1. Load data splits ---
    logger.info("Loading train/valid/test splits from %s", processed_dir)
    try:
        train = pd.read_parquet(processed_dir / "train.parquet")
        valid = pd.read_parquet(processed_dir / "valid.parquet")
        test = pd.read_parquet(processed_dir / "test.parquet")
    except FileNotFoundError as e:
        logger.error("Data splits not found: %s", e)
        raise

    logger.info(
        "Data loaded - train: %d, valid: %d, test: %d rows",
        len(train),
        len(valid),
        len(test),
    )

    feature_cols = _get_feature_cols(train, target_col)
    logger.info("Using %d feature columns", len(feature_cols))

    all_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, pd.Series] = {}
    models: dict[str, Any] = {}

    # --- 2. Load baseline results if available ---
    if include_baselines:
        try:
            tables_dir = get_path("tables")
            baseline_csv = tables_dir / "baseline_comparison.csv"
            if baseline_csv.exists():
                baseline_df = pd.read_csv(baseline_csv)
                for _, row in baseline_df.iterrows():
                    name = row["model"]
                    metrics = {}
                    if "mae" in row and pd.notna(row["mae"]):
                        metrics["mae"] = float(row["mae"])
                    if "rmse" in row and pd.notna(row["rmse"]):
                        metrics["rmse"] = float(row["rmse"])
                    if "r2" in row and pd.notna(row["r2"]):
                        metrics["r2"] = float(row["r2"])
                    if metrics:
                        all_metrics[name] = metrics
                logger.info("Loaded %d baseline models from CSV", len(all_metrics))
            else:
                logger.warning("Baseline comparison CSV not found at %s", baseline_csv)
        except Exception as e:
            logger.warning("Could not load baseline results: %s", e)

    # --- 3. Try loading cached tree models ---
    cached = None
    if not retrain:
        cached = _load_cached_models(str(model_dir))

    # --- 4. Train tree-based models ---
    if include_tree:
        if cached is not None:
            logger.info("Using cached tree models")
            models.update(cached)
            # Still need to evaluate on test
            if "regressor" in cached and cached["regressor"] is not None:
                reg_pred = predict_regression(
                    cached["regressor"], test, feature_cols
                )
                predictions["HistGBR"] = reg_pred
                reg_metrics = evaluate_regression(test[target_col], reg_pred)
                all_metrics["HistGBR"] = reg_metrics
        else:
            try:
                tree_results = train_tree_models(
                    train, valid, feature_cols, target_col
                )
                models["tree_results"] = tree_results

                # Evaluate regressor on test set
                if tree_results["regressor"] is not None:
                    reg_pred = predict_regression(
                        tree_results["regressor"], test, feature_cols
                    )
                    predictions["HistGBR"] = reg_pred
                    reg_metrics = evaluate_regression(test[target_col], reg_pred)
                    all_metrics["HistGBR"] = reg_metrics
                    logger.info("HistGBR test MAE: %.4f", reg_metrics["mae"])

            except Exception as e:
                logger.error("Tree model training failed: %s", e)

    # --- 5. Compare all models ---
    comparison_table = compare_models(all_metrics)
    logger.info("Final model comparison:\n%s", comparison_table.to_string())

    results: dict[str, Any] = {
        "comparison_table": comparison_table,
        "predictions": predictions,
        "models": models,
        "all_metrics": all_metrics,
        "feature_cols": feature_cols,
        "train": train,
        "valid": valid,
        "test": test,
    }

    # --- 6. Save results ---
    try:
        _save_results(results)
    except Exception as e:
        logger.error("Failed to save results: %s", e)

    return results


def _save_results(
    results: dict[str, Any],
    output_dir: str | None = None,
) -> None:
    """Persist pipeline results to disk.

    Saves:
        - Comparison table as CSV.
        - Predictions as parquet.
        - Fitted models as pickle.

    Args:
        results: Output dict from ``run_pipeline``.
        output_dir: Directory to write outputs. Defaults to config path.
    """
    if output_dir is not None:
        tables_path = Path(output_dir) / "tables"
        models_path = Path(output_dir) / "models"
    else:
        tables_path = get_path("tables")
        models_path = get_path("models")

    tables_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison = results.get("comparison_table")
    if comparison is not None and not comparison.empty:
        csv_path = tables_path / "model_comparison.csv"
        comparison.to_csv(csv_path, index=False)
        logger.info("Saved model comparison to %s", csv_path)

    # Save models as pickle
    tree_results = results.get("models", {}).get("tree_results")
    if tree_results is not None:
        if tree_results.get("regressor") is not None:
            reg_path = models_path / "hist_gbr_regressor.pkl"
            with open(reg_path, "wb") as f:
                pickle.dump(tree_results["regressor"], f)
            logger.info("Saved regressor to %s", reg_path)

        if tree_results.get("classifier") is not None:
            cls_path = models_path / "hist_gbc_classifier.pkl"
            with open(cls_path, "wb") as f:
                pickle.dump(tree_results["classifier"], f)
            logger.info("Saved classifier to %s", cls_path)

    # Save predictions
    predictions = results.get("predictions", {})
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_path = tables_path / "predictions.parquet"
        pred_df.to_parquet(pred_path)
        logger.info("Saved predictions to %s", pred_path)


def _load_cached_models(
    model_dir: str | None = None,
) -> dict[str, Any] | None:
    """Attempt to load previously trained models from disk.

    Args:
        model_dir: Directory containing cached model artefacts.

    Returns:
        Dict of model name -> model object, or None if cache is missing.
    """
    if model_dir is None:
        model_dir = str(get_path("models"))

    model_path = Path(model_dir)
    reg_path = model_path / "hist_gbr_regressor.pkl"
    cls_path = model_path / "hist_gbc_classifier.pkl"

    if not reg_path.exists() and not cls_path.exists():
        logger.info("No cached models found in %s", model_dir)
        return None

    cached: dict[str, Any] = {}

    if reg_path.exists():
        try:
            with open(reg_path, "rb") as f:
                cached["regressor"] = pickle.load(f)
            logger.info("Loaded cached regressor from %s", reg_path)
        except Exception as e:
            logger.warning("Failed to load cached regressor: %s", e)

    if cls_path.exists():
        try:
            with open(cls_path, "rb") as f:
                cached["classifier"] = pickle.load(f)
            logger.info("Loaded cached classifier from %s", cls_path)
        except Exception as e:
            logger.warning("Failed to load cached classifier: %s", e)

    return cached if cached else None
