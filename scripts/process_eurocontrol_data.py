"""Process Eurocontrol Airport_Traffic.xlsx into model-ready data and retrain.

Reads the Eurocontrol IFR daily traffic dataset, filters Madrid-Barajas (LEMD),
fetches matching weather data from Open-Meteo and holidays from Nager.Date,
engineers all features (calendar, weather, ACPS, lags, rolling windows),
produces train/valid/test splits, and retrains HistGradientBoosting models.

Usage:
    python scripts/process_eurocontrol_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXCEL_PATH = PROJECT_ROOT / "Airport_Traffic.xlsx"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

with open(CONFIG_DIR / "modeling.yaml", "r") as f:
    MODELING = yaml.safe_load(f)

with open(CONFIG_DIR / "airports.yaml", "r") as f:
    AIRPORT_CFG = yaml.safe_load(f)

AIRPORT = AIRPORT_CFG["primary_airport"]
AIRPORT_LAT = AIRPORT["latitude"]
AIRPORT_LON = AIRPORT["longitude"]

# Ensure directories exist
for d in [
    DATA_DIR / "processed",
    DATA_DIR / "raw" / "holidays",
    DATA_DIR / "raw" / "weather",
    OUTPUT_DIR / "models",
    OUTPUT_DIR / "tables",
    OUTPUT_DIR / "figures",
]:
    d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# STEP 1: Load and filter Eurocontrol data
# ===========================================================================

def load_madrid_flights() -> pd.DataFrame:
    """Load Airport_Traffic.xlsx DATA sheet and filter for LEMD."""
    print("[1/7] Loading Eurocontrol data...")
    df = pd.read_excel(EXCEL_PATH, sheet_name="DATA")
    mad = df[df["APT_ICAO"] == "LEMD"].copy()
    mad["FLT_DATE"] = pd.to_datetime(mad["FLT_DATE"])
    mad = mad.sort_values("FLT_DATE").reset_index(drop=True)

    # Use NM columns (zero nulls) and rename to match pipeline conventions
    mad = mad.rename(columns={
        "FLT_DATE": "date",
        "FLT_DEP_1": "departures",
        "FLT_ARR_1": "arrivals",
        "FLT_TOT_1": "total_movements",
    })

    # Keep only what we need
    mad = mad[["date", "arrivals", "departures", "total_movements"]].copy()
    mad["airport_icao"] = "LEMD"

    print(f"    {len(mad)} daily records: {mad.date.min().date()} to {mad.date.max().date()}")
    return mad


# ===========================================================================
# STEP 2: Fetch weather data from Open-Meteo
# ===========================================================================

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
]


def fetch_daily_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo and aggregate to daily.

    Open-Meteo archive API has a limit of ~1 year per request,
    so we chunk the date range into yearly segments.
    """
    print("[2/7] Fetching weather data from Open-Meteo...")
    cache_path = DATA_DIR / "raw" / "weather" / "weather_daily_LEMD_eurocontrol.parquet"
    if cache_path.exists():
        print("    Using cached weather data")
        return pd.read_parquet(cache_path)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    all_frames = []
    # Chunk into 365-day segments (Open-Meteo limit)
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=364), end)
        s_str = chunk_start.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        print(f"    Fetching weather: {s_str} to {e_str}...")

        params = {
            "latitude": AIRPORT_LAT,
            "longitude": AIRPORT_LON,
            "start_date": s_str,
            "end_date": e_str,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
        }

        try:
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            if "hourly" not in data:
                print(f"    WARNING: No hourly data in response for {s_str} to {e_str}")
                chunk_start = chunk_end + pd.Timedelta(days=1)
                continue

            chunk_df = pd.DataFrame(data["hourly"])
            chunk_df["time"] = pd.to_datetime(chunk_df["time"], utc=True)
            all_frames.append(chunk_df)
        except requests.exceptions.RequestException as e:
            print(f"    WARNING: Weather fetch failed for {s_str}-{e_str}: {e}")

        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not all_frames:
        raise RuntimeError("Failed to fetch any weather data")

    hourly = pd.concat(all_frames, ignore_index=True)
    hourly = hourly.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    hourly["date"] = hourly["time"].dt.date

    # Aggregate hourly -> daily
    daily_weather = hourly.groupby("date").agg(
        temperature_2m=("temperature_2m", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        precipitation=("precipitation", "sum"),
        rain_total=("rain", "sum"),
        weather_code_max=("weather_code", "max"),
        surface_pressure=("surface_pressure", "mean"),
        wind_speed_10m=("wind_speed_10m", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),
        wind_direction_10m=("wind_direction_10m", "mean"),
        wind_gusts_10m=("wind_gusts_10m", "max"),
        cloud_cover=("cloud_cover", "mean"),
    ).reset_index()

    daily_weather["date"] = pd.to_datetime(daily_weather["date"])

    # Engineered weather features
    rad = np.deg2rad(daily_weather["wind_direction_10m"])
    daily_weather["wind_dir_sin"] = np.sin(rad)
    daily_weather["wind_dir_cos"] = np.cos(rad)
    daily_weather["is_severe_weather"] = (daily_weather["weather_code_max"] >= 65).astype(int)
    daily_weather["is_raining"] = (daily_weather["rain_total"] > 0).astype(int)

    # Save cache
    daily_weather.to_parquet(cache_path, index=False)
    print(f"    Weather data: {len(daily_weather)} daily records cached")
    return daily_weather


# ===========================================================================
# STEP 3: Fetch holidays
# ===========================================================================

def fetch_all_holidays(years: list[int]) -> set[date]:
    """Fetch Spanish public holidays for all years from Nager.Date API."""
    print("[3/7] Fetching Spanish holidays...")
    holiday_dir = DATA_DIR / "raw" / "holidays"
    all_holidays: set[date] = set()

    for year in years:
        cache_file = holiday_dir / f"spain_holidays_{year}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/ES"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                records = resp.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
                print(f"    Fetched and cached {year} holidays")
            except requests.exceptions.RequestException as e:
                print(f"    WARNING: Could not fetch holidays for {year}: {e}")
                continue

        for rec in records:
            if rec.get("global", True):
                d = pd.Timestamp(rec["date"]).date()
                all_holidays.add(d)

    print(f"    Total holiday dates loaded: {len(all_holidays)}")
    return all_holidays


# ===========================================================================
# STEP 4: Build calendar features (adapted for daily data)
# ===========================================================================

def build_daily_calendar_features(
    dates: pd.Series,
    holiday_dates: set[date],
) -> pd.DataFrame:
    """Build calendar features for daily timestamps."""
    print("[4/7] Building calendar features...")
    ts = pd.to_datetime(dates)

    df = pd.DataFrame()
    df["dow"] = ts.dt.dayofweek  # Monday=0
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["day_of_year"] = ts.dt.dayofyear

    # Holiday features
    date_vals = ts.dt.date
    df["is_holiday"] = date_vals.isin(holiday_dates).astype(int)
    df["is_pre_holiday"] = date_vals.apply(
        lambda d: (d + timedelta(days=1)) in holiday_dates
    ).astype(int)
    df["is_post_holiday"] = date_vals.apply(
        lambda d: (d - timedelta(days=1)) in holiday_dates
    ).astype(int)

    def _is_bridge(d: date) -> bool:
        if d.weekday() >= 5 or d in holiday_dates:
            return False
        prev = d - timedelta(days=1)
        nxt = d + timedelta(days=1)
        prev_off = prev in holiday_dates or prev.weekday() >= 5
        next_off = nxt in holiday_dates or nxt.weekday() >= 5
        return prev_off and next_off

    df["is_bridge_day"] = date_vals.apply(_is_bridge).astype(int)

    # Cyclic encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    print(f"    Calendar features: {len(df.columns)} columns")
    return df


# ===========================================================================
# STEP 5: Compute ACPS and build model table
# ===========================================================================

def compute_acps(df: pd.DataFrame) -> pd.Series:
    """Compute Airport Congestion Pressure Score (daily version).

    Same formula as the existing pipeline:
      ACPS = rescale_to_0_100(w_mov * z(movements) + w_prs * z(pressure_ratio))
    where pressure_ratio = movements / median(movements for same dow).
    """
    weights = MODELING["target"]
    w_movement = weights["acps_movement_weight"]  # 0.6
    w_pressure = weights["acps_pressure_weight"]   # 0.4

    total = df["total_movements"].astype(float)

    # Pressure ratio: movements / median for same day-of-week
    temp = df[["total_movements", "dow"]].copy()
    medians = temp.groupby("dow")["total_movements"].transform("median")
    pressure_ratio = total / np.maximum(1.0, medians)

    # Z-scores
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    z_movements = _zscore(total)
    z_pressure = _zscore(pressure_ratio)

    acps_raw = w_movement * z_movements + w_pressure * z_pressure

    # Min-max rescale to 0-100
    acps_min, acps_max = acps_raw.min(), acps_raw.max()
    if acps_max == acps_min:
        return pd.Series(50.0, index=df.index)
    return (acps_raw - acps_min) / (acps_max - acps_min) * 100.0


def build_model_table(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all features into the final model table."""
    print("[5/7] Building model table...")

    # Merge flights + weather on date
    model_df = flights.copy()
    model_df = model_df.merge(weather, on="date", how="left")

    # Add calendar features (aligned by index)
    calendar = calendar.reset_index(drop=True)
    for col in calendar.columns:
        model_df[col] = calendar[col].values

    # Arrival/departure imbalance
    model_df["arr_dep_imbalance"] = (
        model_df["arrivals"] - model_df["departures"]
    ) / model_df["total_movements"].clip(lower=1)

    # ACPS target
    model_df["acps"] = compute_acps(model_df)

    # Congestion class
    low_thr = MODELING["target"]["low_threshold"]   # 60th percentile
    high_thr = MODELING["target"]["high_threshold"]  # 85th percentile
    p_low = np.percentile(model_df["acps"].dropna(), low_thr)
    p_high = np.percentile(model_df["acps"].dropna(), high_thr)
    model_df["congestion_class"] = pd.cut(
        model_df["acps"],
        bins=[-np.inf, p_low, p_high, np.inf],
        labels=["Low", "Medium", "High"],
    )
    model_df["congestion_binary"] = (model_df["acps"] >= p_low).astype(int)

    # --- Lag features (daily lags) ---
    # Adapt lag config: original is hourly, we use day equivalents
    # 1d, 2d, 3d, 7d (weekly), 14d, 28d (monthly), 365d (yearly)
    daily_lags = [1, 2, 3, 7, 14, 28, 365]
    for lag in daily_lags:
        model_df[f"acps_lag_{lag}d"] = model_df["acps"].shift(lag)
        model_df[f"movements_lag_{lag}d"] = model_df["total_movements"].shift(lag)

    # --- Rolling features ---
    # 7d, 14d, 28d rolling means; 28d rolling std
    for w in [7, 14, 28]:
        model_df[f"acps_rmean_{w}d"] = model_df["acps"].rolling(w, min_periods=1).mean()
        model_df[f"movements_rmean_{w}d"] = model_df["total_movements"].rolling(w, min_periods=1).mean()
    model_df["acps_rstd_28d"] = model_df["acps"].rolling(28, min_periods=1).std()

    # Year-over-year change
    model_df["movements_yoy_change"] = (
        model_df["total_movements"] - model_df["total_movements"].shift(365)
    )

    print(f"    Model table: {len(model_df)} rows, {len(model_df.columns)} columns")

    # Drop rows where key lag features are NaN (first 28 rows)
    feasible_lag_cols = [c for c in model_df.columns if "_lag_" in c and "365" not in c]
    feasible_rolling_cols = [c for c in model_df.columns if "_rmean_" in c or "_rstd_" in c]
    drop_cols = feasible_lag_cols + feasible_rolling_cols
    before = len(model_df)
    model_df = model_df.dropna(subset=drop_cols).reset_index(drop=True)
    print(f"    Dropped {before - len(model_df)} rows with NaN from lags/rolling, {len(model_df)} remain")

    return model_df


# ===========================================================================
# STEP 6: Split data
# ===========================================================================

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/valid/test split."""
    print("[6/7] Splitting data...")
    train_ratio = MODELING["split"]["train"]
    valid_ratio = MODELING["split"]["validation"]

    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train = df.iloc[:train_end].reset_index(drop=True)
    valid = df.iloc[train_end:valid_end].reset_index(drop=True)
    test = df.iloc[valid_end:].reset_index(drop=True)

    print(f"    Train: {len(train)} rows ({train.date.min().date()} to {train.date.max().date()})")
    print(f"    Valid: {len(valid)} rows ({valid.date.min().date()} to {valid.date.max().date()})")
    print(f"    Test:  {len(test)} rows ({test.date.min().date()} to {test.date.max().date()})")

    return train, valid, test


# ===========================================================================
# STEP 7: Train models and evaluate
# ===========================================================================

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Determine feature columns by excluding non-feature columns."""
    exclude = {
        "date", "airport_icao", "acps", "congestion_class", "congestion_binary",
        "rain_total", "weather_code_max", "day_of_year",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, float, int]
    ]
    return feature_cols


def train_and_evaluate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Train HistGradientBoosting models and evaluate on test set."""
    print("[7/7] Training models and evaluating...")

    feature_cols = get_feature_cols(train)
    print(f"    Using {len(feature_cols)} features")

    # Handle NaN in features (lag_365d, movements_yoy_change may have NaN)
    for df in [train, valid, test]:
        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    X_train = train[feature_cols]
    y_train = train["acps"]
    X_valid = valid[feature_cols]
    y_valid = valid["acps"]
    X_test = test[feature_cols]
    y_test = test["acps"]

    # --- Regressor ---
    hgb_params = MODELING["models"]["hist_gradient_boosting"]
    reg = HistGradientBoostingRegressor(
        max_iter=hgb_params["max_iter"],
        max_depth=hgb_params["max_depth"],
        learning_rate=hgb_params["learning_rate"],
        random_state=hgb_params["random_state"],
        min_samples_leaf=20,
    )
    reg.fit(X_train, y_train)

    # Evaluate regressor
    for split_name, X, y in [("Valid", X_valid, y_valid), ("Test", X_test, y_test)]:
        preds = reg.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        print(f"    Regressor [{split_name}]  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")

    # --- Classifier ---
    y_train_cls = train["congestion_class"].astype(str)
    y_valid_cls = valid["congestion_class"].astype(str)
    y_test_cls = test["congestion_class"].astype(str)

    clf = HistGradientBoostingClassifier(
        max_iter=hgb_params["max_iter"],
        max_depth=hgb_params["max_depth"],
        learning_rate=hgb_params["learning_rate"],
        random_state=hgb_params["random_state"],
        min_samples_leaf=20,
    )
    clf.fit(X_train, y_train_cls)

    for split_name, X, y_cls in [("Valid", X_valid, y_valid_cls), ("Test", X_test, y_test_cls)]:
        preds_cls = clf.predict(X)
        acc = accuracy_score(y_cls, preds_cls)
        f1 = f1_score(y_cls, preds_cls, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_cls, preds_cls)
        print(f"    Classifier [{split_name}]  Accuracy: {acc:.4f}  F1: {f1:.4f}  Balanced Acc: {bal_acc:.4f}")

    # Test set detailed classification report
    print("\n    Classification Report (Test Set):")
    print(classification_report(y_test_cls, clf.predict(X_test), zero_division=0))

    # --- Feature importance (permutation-based) ---
    perm_result = permutation_importance(
        reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": perm_result.importances_mean,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n    Top 15 features by importance:")
    print(importance_df.head(15).to_string(index=False))

    # --- Save everything ---
    # Models
    joblib.dump(reg, OUTPUT_DIR / "models" / "hgb_regressor_eurocontrol.pkl")
    joblib.dump(clf, OUTPUT_DIR / "models" / "hgb_classifier_eurocontrol.pkl")
    print(f"\n    Models saved to {OUTPUT_DIR / 'models'}")

    # Feature importance
    importance_df.to_csv(OUTPUT_DIR / "tables" / "feature_importance_eurocontrol.csv", index=False)

    # Model comparison table
    test_preds_reg = reg.predict(X_test)
    test_preds_cls = clf.predict(X_test)
    comparison = pd.DataFrame({
        "Model": ["HGB Regressor", "HGB Classifier"],
        "MAE": [
            mean_absolute_error(y_test, test_preds_reg),
            np.nan,
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, test_preds_reg)),
            np.nan,
        ],
        "R2": [
            r2_score(y_test, test_preds_reg),
            np.nan,
        ],
        "Accuracy": [
            np.nan,
            accuracy_score(y_test_cls, test_preds_cls),
        ],
        "F1_macro": [
            np.nan,
            f1_score(y_test_cls, test_preds_cls, average="macro", zero_division=0),
        ],
        "Balanced_Accuracy": [
            np.nan,
            balanced_accuracy_score(y_test_cls, test_preds_cls),
        ],
    })
    comparison.to_csv(OUTPUT_DIR / "tables" / "model_comparison_eurocontrol.csv", index=False)

    # Predictions on test set for analysis
    test_results = test[["date", "acps", "congestion_class", "total_movements"]].copy()
    test_results["acps_predicted"] = test_preds_reg
    test_results["congestion_predicted"] = test_preds_cls
    test_results.to_csv(OUTPUT_DIR / "tables" / "test_predictions_eurocontrol.csv", index=False)

    print(f"    Tables saved to {OUTPUT_DIR / 'tables'}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run the full Eurocontrol data processing and model retraining pipeline."""
    print("=" * 70)
    print("EUROCONTROL AIRPORT TRAFFIC - MADRID BARAJAS PROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: Load flights
    flights = load_madrid_flights()

    # Step 2: Fetch weather
    start_date = flights.date.min().strftime("%Y-%m-%d")
    end_date = flights.date.max().strftime("%Y-%m-%d")
    weather = fetch_daily_weather(start_date, end_date)

    # Step 3: Fetch holidays
    years = sorted(flights.date.dt.year.unique())
    holidays = fetch_all_holidays(years)

    # Step 4: Calendar features
    calendar = build_daily_calendar_features(flights["date"], holidays)

    # Step 5: Build model table
    model_df = build_model_table(flights, weather, calendar)

    # Save full processed dataset
    processed_path = DATA_DIR / "processed" / "eurocontrol_model_table.parquet"
    model_df.to_parquet(processed_path, index=False)
    print(f"\n    Full model table saved: {processed_path}")

    # Also save as CSV for teammates without parquet support
    csv_path = DATA_DIR / "processed" / "eurocontrol_model_table.csv"
    model_df.to_csv(csv_path, index=False)
    print(f"    CSV copy saved: {csv_path}")

    # Step 6: Split
    train, valid, test = split_data(model_df)

    # Save splits
    train.to_parquet(DATA_DIR / "processed" / "train_eurocontrol.parquet", index=False)
    valid.to_parquet(DATA_DIR / "processed" / "valid_eurocontrol.parquet", index=False)
    test.to_parquet(DATA_DIR / "processed" / "test_eurocontrol.parquet", index=False)
    print("    Splits saved to data/processed/")

    # Step 7: Train and evaluate
    train_and_evaluate(train, valid, test)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    # Summary stats
    print(f"\nDataset: {len(model_df)} daily records")
    print(f"Date range: {model_df.date.min().date()} to {model_df.date.max().date()}")
    print(f"Features: {len(get_feature_cols(model_df))}")
    print(f"Train/Valid/Test: {len(train)}/{len(valid)}/{len(test)}")
    print(f"\nOutputs:")
    print(f"  - data/processed/eurocontrol_model_table.parquet")
    print(f"  - data/processed/train_eurocontrol.parquet")
    print(f"  - data/processed/valid_eurocontrol.parquet")
    print(f"  - data/processed/test_eurocontrol.parquet")
    print(f"  - outputs/models/hgb_regressor_eurocontrol.pkl")
    print(f"  - outputs/models/hgb_classifier_eurocontrol.pkl")
    print(f"  - outputs/tables/feature_importance_eurocontrol.csv")
    print(f"  - outputs/tables/model_comparison_eurocontrol.csv")
    print(f"  - outputs/tables/test_predictions_eurocontrol.csv")


if __name__ == "__main__":
    main()
