"""Model evaluation and diagnostic visualisation functions.

Produces charts for model comparison, feature importance, confusion matrices,
and residual/error analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing all models across selected metrics.

    Args:
        comparison_df: Output from ``evaluation.compare_models()``,
            with one row per model and metric columns.
        metrics: List of metric column names to plot (e.g.
            ``["mae", "rmse", "r2"]``). Defaults to all numeric columns.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns

    if metrics is None:
        # Use all numeric columns except 'model'
        metrics = [
            c for c in comparison_df.columns
            if c != "model" and pd.api.types.is_numeric_dtype(comparison_df[c])
        ]

    if not metrics:
        logger.warning("No metrics to plot in model comparison.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", transform=ax.transAxes)
        plt.close("all")
        return fig

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = comparison_df[["model", metric]].dropna()
        bars = ax.bar(
            range(len(data)),
            data[metric].values,
            color="steelblue",
            alpha=0.8,
        )
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["model"].values, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, data[metric].values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Model Comparison", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved model comparison plot to %s", save_path)

    plt.close("all")
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top feature importances.

    Args:
        importance_df: DataFrame with ``feature`` and ``importance`` columns,
            as returned by ``tree_models.get_feature_importance()``.
        top_n: Number of top features to display.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    top = importance_df.head(top_n).copy()
    # Reverse for horizontal bar (top feature at the top)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top))))

    ax.barh(
        top["feature"],
        top["importance"],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(importance_df))} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)

    plt.close("all")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    class_labels: list[str] | None = None,
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap confusion matrix for the classification model.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        class_labels: Display names for each class (e.g.
            ``["Low", "Medium-Low", "Medium", "Medium-High", "High"]``).
        normalize: If True, show proportions instead of raw counts.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as compute_cm

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = compute_cm(y_true, y_pred)

    if normalize and cm.sum() > 0:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    if class_labels is None:
        # Derive labels from unique values in y_true and y_pred
        all_labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])), key=str)
        class_labels = [str(lbl) for lbl in all_labels]

    fig, ax = plt.subplots(figsize=(max(6, len(class_labels) * 1.2), max(5, len(class_labels))))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to %s", save_path)

    plt.close("all")
    return fig


def plot_error_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    timestamps: pd.DatetimeIndex | pd.Series | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Multi-panel error analysis plot.

    Panels:
        1. **Actual vs Predicted** scatter with identity line.
        2. **Residuals over time** (if timestamps provided).
        3. **Residual distribution** histogram with normal overlay.
        4. **Residuals by hour-of-day** boxplot to detect systematic bias.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        timestamps: Optional timestamps for temporal residual analysis.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure with 4 subplots.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    has_timestamps = timestamps is not None and len(timestamps) == len(y_true)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, color="steelblue", alpha=0.7, edgecolors="navy", linewidth=0.5)
    # Identity line
    all_vals = np.concatenate([y_true, y_pred])
    if len(all_vals) > 0:
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.05 if vmax > vmin else 0.5
        ax1.plot(
            [vmin - margin, vmax + margin],
            [vmin - margin, vmax + margin],
            "r--", alpha=0.6, label="Identity line",
        )
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: Residuals over time
    ax2 = axes[0, 1]
    if has_timestamps:
        ts = pd.to_datetime(timestamps)
        ax2.scatter(ts, residuals, color="steelblue", alpha=0.7, s=30)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Time")
    else:
        ax2.scatter(range(len(residuals)), residuals, color="steelblue", alpha=0.7, s=30)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Observation Index")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals Over Time")
    ax2.grid(alpha=0.3)

    # Panel 3: Residual distribution histogram
    ax3 = axes[1, 0]
    n_bins = max(5, min(20, len(residuals) // 2))
    ax3.hist(residuals, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="navy", density=True)
    # Normal overlay if enough data
    if len(residuals) >= 3:
        from scipy import stats
        mu, sigma = residuals.mean(), residuals.std()
        if sigma > 0:
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            ax3.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), "r-", linewidth=2, label="Normal fit")
            ax3.legend(fontsize=8)
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Density")
    ax3.set_title("Residual Distribution")
    ax3.grid(alpha=0.3)

    # Panel 4: Residuals by hour-of-day
    ax4 = axes[1, 1]
    if has_timestamps:
        ts = pd.to_datetime(timestamps)
        hours = ts.hour if hasattr(ts, "hour") else pd.Series(ts).dt.hour
        resid_df = pd.DataFrame({"hour": hours.values, "residual": residuals})
        if resid_df["hour"].nunique() > 1:
            resid_df.boxplot(column="residual", by="hour", ax=ax4)
            ax4.set_title("Residuals by Hour of Day")
        else:
            ax4.bar(resid_df["hour"].unique(), [residuals.mean()], color="steelblue")
            ax4.set_title("Residuals by Hour (single hour)")
        ax4.set_xlabel("Hour of Day")
        ax4.set_ylabel("Residual")
        # Remove automatic suptitle from boxplot
        fig.suptitle("")
    else:
        ax4.text(
            0.5, 0.5,
            "No timestamps\navailable",
            ha="center", va="center",
            transform=ax4.transAxes, fontsize=12,
        )
        ax4.set_title("Residuals by Hour (N/A)")
    ax4.grid(alpha=0.3)

    fig.suptitle("Error Analysis", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved error analysis plot to %s", save_path)

    plt.close("all")
    return fig
