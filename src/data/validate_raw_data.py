"""Raw data validation utilities.

Checks completeness, schema conformity, and data quality for all raw sources.
Run after data collection to identify gaps before proceeding to feature engineering.
"""

from pathlib import Path

import pandas as pd

from src.config import AIRPORT_ICAO, DATA_START_DATE, DATA_END_DATE, get_path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_opensky_data(airport: str = AIRPORT_ICAO) -> dict:
    """Validate OpenSky arrivals and departures data.

    Returns:
        Dict with validation results.
    """
    raw_dir = Path(get_path("raw_opensky"))
    results = {"source": "opensky", "airport": airport, "issues": []}

    for flight_type in ["arrivals", "departures"]:
        combined_path = raw_dir / f"all_{flight_type}_{airport}.parquet"

        if not combined_path.exists():
            results["issues"].append(f"Missing combined {flight_type} file")
            continue

        df = pd.read_parquet(combined_path)
        results[f"{flight_type}_rows"] = len(df)

        if df.empty:
            results["issues"].append(f"Empty {flight_type} data")
            continue

        # Check for required columns
        expected_cols = ["icao24", "firstSeen", "lastSeen"]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            results["issues"].append(f"{flight_type} missing columns: {missing_cols}")

        # Check time coverage
        if "firstSeen" in df.columns:
            ts_col = "firstSeen" if flight_type == "departures" else "lastSeen"
            if ts_col in df.columns:
                min_ts = pd.to_datetime(df[ts_col], unit="s", utc=True)
                results[f"{flight_type}_date_range"] = f"{min_ts.min()} to {min_ts.max()}"

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            results["issues"].append(f"{flight_type}: {dup_count} duplicate rows")

    return results


def validate_weather_data(airport: str = AIRPORT_ICAO) -> dict:
    """Validate weather data.

    Returns:
        Dict with validation results.
    """
    raw_dir = Path(get_path("raw_weather"))
    results = {"source": "weather", "issues": []}

    filepath = raw_dir / f"weather_hourly_{airport}.parquet"

    if not filepath.exists():
        results["issues"].append("Missing weather file")
        return results

    df = pd.read_parquet(filepath)
    results["rows"] = len(df)
    results["columns"] = list(df.columns)

    if "timestamp" in df.columns:
        results["date_range"] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"

        # Check for gaps in hourly data
        expected_hours = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq="h",
        )
        missing_hours = len(expected_hours) - len(df)
        if missing_hours > 0:
            results["issues"].append(f"{missing_hours} missing hours in weather data")
            results["missing_hours"] = missing_hours

    # Check for null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        results["null_columns"] = null_cols.to_dict()

    return results


def validate_holidays_data() -> dict:
    """Validate holiday data.

    Returns:
        Dict with validation results.
    """
    raw_dir = Path(get_path("raw_holidays"))
    results = {"source": "holidays", "issues": []}

    json_files = list(raw_dir.glob("spain_holidays_*.json"))
    results["files_found"] = len(json_files)

    if not json_files:
        results["issues"].append("No holiday files found")

    return results


def validate_airport_metadata(airport: str = AIRPORT_ICAO) -> dict:
    """Validate airport metadata.

    Returns:
        Dict with validation results.
    """
    raw_dir = Path(get_path("raw_airports"))
    results = {"source": "airport_metadata", "issues": []}

    filepath = raw_dir / "airport_reference.parquet"

    if not filepath.exists():
        results["issues"].append("Missing airport reference file")
        return results

    df = pd.read_parquet(filepath)
    results["rows"] = len(df)

    if "ident" in df.columns:
        if airport not in df["ident"].values:
            results["issues"].append(f"Airport {airport} not found in metadata")

    return results


def run_all_validations() -> dict:
    """Run all data validations and return combined report.

    Returns:
        Dict with results for each data source.
    """
    logger.info("Running data validation...")

    report = {
        "opensky": validate_opensky_data(),
        "weather": validate_weather_data(),
        "holidays": validate_holidays_data(),
        "airport_metadata": validate_airport_metadata(),
    }

    # Summary
    total_issues = sum(len(r.get("issues", [])) for r in report.values())
    logger.info(f"Validation complete. {total_issues} issue(s) found.")

    for source, results in report.items():
        issues = results.get("issues", [])
        if issues:
            for issue in issues:
                logger.warning(f"  [{source}] {issue}")
        else:
            logger.info(f"  [{source}] OK")

    return report


if __name__ == "__main__":
    report = run_all_validations()
    print("\n=== Validation Report ===")
    for source, results in report.items():
        print(f"\n{source.upper()}:")
        for key, value in results.items():
            if key != "issues":
                print(f"  {key}: {value}")
        issues = results.get("issues", [])
        if issues:
            print(f"  ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  Status: OK")
