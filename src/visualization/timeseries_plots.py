"""Time series visualisation functions for exploratory analysis.

Produces publication-quality charts for daily traffic patterns, seasonal
decomposition, and ACPS time series with annotations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def plot_daily_patterns(
    movements_df: pd.DataFrame,
    target_col: str = "total_movements",
    timestamp_col: str = "timestamp_hour",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot average hourly traffic patterns by day of week.

    Creates a faceted line chart showing the mean value of ``target_col``
    for each hour (0-23), with separate lines or panels per day of week.

    Args:
        movements_df: Hourly model table with timestamp and target columns.
        target_col: Column to aggregate (e.g. ``total_movements`` or ``acps``).
        timestamp_col: Datetime column name.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    import numpy as np

    df = movements_df.copy()

    # Ensure timestamp column exists and is datetime
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df["hour_of_day"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.day_name()
    elif "hour" in df.columns:
        df["hour_of_day"] = df["hour"]
        if "dow" in df.columns:
            day_names = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            df["day_of_week"] = df["dow"].map(
                lambda x: day_names[x] if 0 <= x < 7 else "Unknown"
            )
        else:
            df["day_of_week"] = "All"
    else:
        logger.warning("No timestamp or hour column found. Plotting raw index.")
        df["hour_of_day"] = np.arange(len(df)) % 24
        df["day_of_week"] = "All"

    # Compute average target by hour (and optionally by day of week)
    unique_days = df["day_of_week"].nunique()

    fig, ax = plt.subplots(figsize=(12, 6))

    if unique_days > 1 and unique_days <= 7:
        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        for day in day_order:
            day_data = df[df["day_of_week"] == day]
            if len(day_data) == 0:
                continue
            hourly_avg = day_data.groupby("hour_of_day")[target_col].mean()
            ax.plot(hourly_avg.index, hourly_avg.values, marker="o", label=day, alpha=0.8)
        ax.legend(fontsize=8)
    else:
        hourly_avg = df.groupby("hour_of_day")[target_col].mean()
        ax.bar(hourly_avg.index, hourly_avg.values, color="steelblue", alpha=0.8)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(f"Average {target_col}")
    ax.set_title(f"Average Hourly {target_col} Pattern")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved daily patterns plot to %s", save_path)

    plt.close("all")
    return fig


def plot_seasonal_decomposition(
    series: pd.Series,
    period: int = 24,
    model: str = "additive",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot STL / classical seasonal decomposition of the ACPS time series.

    Shows trend, seasonal, and residual components in a stacked subplot layout.

    Args:
        series: Univariate time series (e.g. hourly ACPS), datetime-indexed.
        period: Seasonal period in observations (24 = daily cycle for hourly data).
        model: Decomposition model type (``"additive"`` or ``"multiplicative"``).
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    import numpy as np

    series = series.dropna()

    # Need at least 2 full periods for seasonal decomposition
    can_decompose = len(series) >= 2 * period

    if can_decompose:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            result = seasonal_decompose(series, model=model, period=period)

            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            result.observed.plot(ax=axes[0], title="Observed")
            result.trend.plot(ax=axes[1], title="Trend")
            result.seasonal.plot(ax=axes[2], title="Seasonal")
            result.resid.plot(ax=axes[3], title="Residual")

            for ax in axes:
                ax.grid(alpha=0.3)

            fig.suptitle("Seasonal Decomposition of ACPS", fontsize=14, y=1.01)
            fig.tight_layout()

        except Exception as e:
            logger.warning(
                "Seasonal decomposition failed (%s). Falling back to raw + rolling mean.",
                e,
            )
            can_decompose = False

    if not can_decompose:
        # Fallback: raw series + rolling mean
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(series.index, series.values, alpha=0.6, label="Raw ACPS")

        # Rolling mean window: min of 3 or half the data length
        window = min(3, max(1, len(series) // 2))
        if window >= 2:
            rolling = series.rolling(window=window, center=True).mean()
            ax.plot(
                rolling.index, rolling.values,
                color="red", linewidth=2, label=f"Rolling Mean (w={window})"
            )

        ax.set_title("ACPS Time Series (too few observations for decomposition)")
        ax.set_xlabel("Time")
        ax.set_ylabel("ACPS")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved seasonal decomposition plot to %s", save_path)

    plt.close("all")
    return fig


def plot_acps_timeseries(
    df: pd.DataFrame,
    target_col: str = "acps",
    timestamp_col: str = "timestamp_hour",
    highlight_severe_weather: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the full ACPS time series with optional weather overlay.

    Draws the ACPS line with shaded regions for severe weather events and
    horizontal reference lines for congestion thresholds.

    Args:
        df: Model table with timestamp, ACPS, and optionally
            ``is_severe_weather`` columns.
        target_col: ACPS column name.
        timestamp_col: Datetime column name.
        highlight_severe_weather: If True and ``is_severe_weather`` column
            exists, shade severe weather periods in red.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    plot_df = df.copy()

    # Resolve timestamp column (might be 'timestamp' instead of 'timestamp_hour')
    ts_col = None
    for candidate in [timestamp_col, "timestamp", "timestamp_hour"]:
        if candidate in plot_df.columns:
            ts_col = candidate
            break

    if ts_col is not None:
        plot_df[ts_col] = pd.to_datetime(plot_df[ts_col], utc=True)
        x_vals = plot_df[ts_col]
    else:
        x_vals = plot_df.index

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color by congestion class if available
    if "congestion_class" in plot_df.columns:
        class_colors = {"Low": "green", "Medium": "orange", "High": "red"}
        classes = plot_df["congestion_class"].astype(str)
        for cls_name, color in class_colors.items():
            mask = classes == cls_name
            if mask.any():
                ax.scatter(
                    x_vals[mask], plot_df.loc[mask, target_col],
                    color=color, label=cls_name, s=40, zorder=3,
                )
        ax.plot(x_vals, plot_df[target_col], color="gray", alpha=0.4, linewidth=1, zorder=1)
        ax.legend(title="Congestion Class")
    else:
        ax.plot(x_vals, plot_df[target_col], color="steelblue", linewidth=1.5)

    # Highlight severe weather
    if (
        highlight_severe_weather
        and "is_severe_weather" in plot_df.columns
        and plot_df["is_severe_weather"].any()
    ):
        severe_mask = plot_df["is_severe_weather"].astype(bool)
        y_min, y_max = ax.get_ylim()
        for idx in plot_df[severe_mask].index:
            x_val = x_vals.iloc[plot_df.index.get_loc(idx)] if ts_col else idx
            ax.axvspan(x_val, x_val, color="red", alpha=0.15)
        # Draw one patch for legend
        ax.axvspan(
            x_vals.iloc[0], x_vals.iloc[0],
            color="red", alpha=0.15, label="Severe Weather",
        )

    # Congestion thresholds as horizontal lines
    if target_col in plot_df.columns and len(plot_df) > 0:
        acps_vals = plot_df[target_col].dropna()
        if len(acps_vals) > 0:
            p60 = acps_vals.quantile(0.6)
            p85 = acps_vals.quantile(0.85)
            ax.axhline(y=p60, color="orange", linestyle="--", alpha=0.5, label=f"60th pctl ({p60:.2f})")
            ax.axhline(y=p85, color="red", linestyle="--", alpha=0.5, label=f"85th pctl ({p85:.2f})")

    ax.set_title("ACPS Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("ACPS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ACPS timeseries plot to %s", save_path)

    plt.close("all")
    return fig
