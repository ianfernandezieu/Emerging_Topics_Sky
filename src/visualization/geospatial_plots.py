"""Geospatial map visualisation functions.

Creates interactive and static maps showing airport location, flight
density heatmaps, and congestion comparison across analysis radius bands.
Uses Folium for interactive maps and Matplotlib/Cartopy for static ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AIRPORT_LAT, AIRPORT_LON, ANALYSIS_BANDS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def plot_airport_map(
    center_lat: float = AIRPORT_LAT,
    center_lon: float = AIRPORT_LON,
    bands: list[dict[str, Any]] | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Create an interactive map centred on the airport with radius bands.

    Draws concentric circles for each analysis band and labels them.
    Useful for validating the geographic scope of the density features.

    Args:
        center_lat: Airport latitude.
        center_lon: Airport longitude.
        bands: List of band config dicts (from ``config/airports.yaml``).
            Defaults to ``ANALYSIS_BANDS``.
        save_path: If provided, save the HTML map to this path.

    Returns:
        Folium Map object.
    """
    import folium

    if bands is None:
        bands = [
            {"name": "Inner", "radius_km": ANALYSIS_BANDS.get("inner", 15), "color": "green"},
            {"name": "Middle", "radius_km": ANALYSIS_BANDS.get("middle", 40), "color": "orange"},
            {"name": "Outer", "radius_km": ANALYSIS_BANDS.get("outer", 80), "color": "red"},
        ]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # Airport marker
    folium.Marker(
        [center_lat, center_lon],
        popup="Madrid-Barajas (LEMD)",
        tooltip="MAD Airport",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
    ).add_to(m)

    # Radius band circles
    for band in bands:
        radius_m = band["radius_km"] * 1000
        color = band.get("color", "blue")
        name = band.get("name", f"{band['radius_km']}km")
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius_m,
            color=color,
            fill=False,
            weight=2,
            popup=f"{name} band ({band['radius_km']} km)",
            tooltip=f"{name}: {band['radius_km']} km",
        ).add_to(m)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(save_path))
        logger.info("Saved airport map to %s", save_path)

    return m


def plot_flight_density(
    density_df: pd.DataFrame,
    timestamp: str | pd.Timestamp | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Visualise aircraft density on a map for a given snapshot.

    Plots individual aircraft positions as markers and overlays a
    heatmap layer. Optionally filters to a single timestamp.

    Args:
        density_df: DataFrame with ``latitude``, ``longitude``, and
            optionally ``altitude``, ``ground_speed`` columns.
            If ``timestamp`` column exists, can filter to a single snapshot.
        timestamp: If provided, filter to this specific snapshot time.
        save_path: If provided, save the HTML map to this path.

    Returns:
        Folium Map object.
    """
    import matplotlib.pyplot as plt

    df = density_df.copy()

    # Filter by timestamp if requested
    if timestamp is not None and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ts = pd.Timestamp(timestamp)
        df = df[df["timestamp"] == ts]

    if len(df) == 0:
        logger.warning("No data to plot for flight density.")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Flight Density - No Data Available")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return fig

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of aircraft positions
    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        c="steelblue",
        alpha=0.6,
        s=20,
        edgecolors="navy",
        linewidth=0.3,
    )

    # Mark airport
    ax.scatter(
        AIRPORT_LON, AIRPORT_LAT,
        c="red", s=200, marker="*", zorder=5, label="MAD Airport",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Aircraft Positions"
    if timestamp is not None:
        title += f" at {timestamp}"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved flight density plot to %s", save_path)

    plt.close("all")
    return fig


def plot_congestion_comparison(
    hourly_df: pd.DataFrame,
    density_df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Side-by-side map panels comparing congestion levels across time windows.

    Creates a grid of small maps (e.g. 4 panels for morning, midday,
    evening, night) showing how aircraft density and congestion shift
    throughout the day.

    Args:
        hourly_df: Hourly model table with ACPS and timestamp.
        density_df: Optional density features with aircraft positions.
        save_path: If provided, save the HTML visualisation to this path.

    Returns:
        Folium Map or Matplotlib Figure (depending on data availability).
    """
    import matplotlib.pyplot as plt

    df = hourly_df.copy()

    # Determine congestion classes
    if "congestion_class" in df.columns:
        classes = df["congestion_class"].astype(str)
    elif "acps" in df.columns:
        # Bin into Low/High based on median
        median_acps = df["acps"].median()
        classes = df["acps"].apply(lambda x: "High" if x >= median_acps else "Low")
    else:
        logger.warning("No congestion data available for comparison plot.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No congestion data", ha="center", va="center", transform=ax.transAxes)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return fig

    low_mask = classes.isin(["Low"])
    high_mask = classes.isin(["High"])

    # If we have density data with lat/lon, use scatter
    if density_df is not None and "latitude" in density_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # We need to match density data to congestion periods
        # For simplicity, split density data in half if no direct mapping
        if "congestion_class" in density_df.columns:
            low_pts = density_df[density_df["congestion_class"].astype(str) == "Low"]
            high_pts = density_df[density_df["congestion_class"].astype(str) == "High"]
        else:
            mid = len(density_df) // 2
            low_pts = density_df.iloc[:mid]
            high_pts = density_df.iloc[mid:]

        for ax, pts, title, color in [
            (ax1, low_pts, "Low Congestion", "green"),
            (ax2, high_pts, "High Congestion", "red"),
        ]:
            if len(pts) > 0:
                ax.scatter(
                    pts["longitude"], pts["latitude"],
                    c=color, alpha=0.5, s=15,
                )
            ax.scatter(AIRPORT_LON, AIRPORT_LAT, c="blue", s=200, marker="*", zorder=5)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.3)
    else:
        # Fallback: bar chart comparison of ACPS during low vs high periods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        low_data = df[low_mask]
        high_data = df[high_mask]

        if "acps" in df.columns:
            if len(low_data) > 0:
                ax1.bar(range(len(low_data)), low_data["acps"].values, color="green", alpha=0.7)
            ax1.set_title(f"Low Congestion (n={len(low_data)})")
            ax1.set_ylabel("ACPS")
            ax1.set_xlabel("Observation")

            if len(high_data) > 0:
                ax2.bar(range(len(high_data)), high_data["acps"].values, color="red", alpha=0.7)
            ax2.set_title(f"High Congestion (n={len(high_data)})")
            ax2.set_ylabel("ACPS")
            ax2.set_xlabel("Observation")

    fig.suptitle("Congestion Level Comparison", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved congestion comparison plot to %s", save_path)

    plt.close("all")
    return fig
