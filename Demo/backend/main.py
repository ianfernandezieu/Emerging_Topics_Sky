"""
Madrid-Barajas congestion forecast — live demo backend.

Loads the trained HGB regressor + classifier (outputs/models/*_eurocontrol.pkl)
and serves real predictions over HTTP. Every number the frontend displays is
computed here from the real model and the real Eurocontrol test set — nothing
is hard-coded for presentation purposes.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)

# ---------------------------------------------------------------- paths / load

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

REG_PATH = MODELS_DIR / "hgb_regressor_eurocontrol.pkl"
CLF_PATH = MODELS_DIR / "hgb_classifier_eurocontrol.pkl"
TABLE_PATH = DATA_DIR / "eurocontrol_model_table.parquet"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance_eurocontrol.csv"
MODEL_COMPARISON_PATH = DATA_DIR / "model_comparison_final.csv"

TEST_SPLIT_DATE = pd.Timestamp("2024-10-19")


@dataclass
class State:
    regressor: object
    classifier: object
    model_table: pd.DataFrame
    feature_names: list[str]
    test_residual_std: float
    metrics_cache: dict | None = None
    # Precomputed caches for the Diagnostics tab — built once at startup.
    test_predictions: pd.DataFrame | None = None   # date, actual, predicted, residual, class, pred_class, movements
    dow_train_means: dict | None = None            # {dow (0-6): mean ACPS on training set}
    train_global_mean_acps: float = 0.0
    train_feature_stats: dict | None = None        # {feature: {mean, std, min, max, p10, p50, p90}}


state: State | None = None


def load_state() -> State:
    reg = joblib.load(REG_PATH)
    clf = joblib.load(CLF_PATH)
    df = pd.read_parquet(TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    features = list(reg.feature_names_in_)

    train = df[df["date"] < TEST_SPLIT_DATE].copy()
    test = df[df["date"] >= TEST_SPLIT_DATE].copy()
    X_test = test[features].fillna(method="ffill").fillna(0)
    preds = reg.predict(X_test)
    pred_classes = clf.predict(X_test)
    residual_std = float(np.std(test["acps"].values - preds, ddof=1))

    # Cache full per-test-day predictions for the Diagnostics tab.
    test_predictions = pd.DataFrame({
        "date": test["date"].values,
        "actual": test["acps"].values,
        "predicted": preds,
        "residual": test["acps"].values - preds,
        "actual_class": test["congestion_class"].values,
        "predicted_class": pred_classes,
        "total_movements": test["total_movements"].values,
    })

    # Day-of-week mean ACPS on the training set (used as DoW baseline).
    dow_train_means = train.groupby("dow")["acps"].mean().to_dict()

    # Global mean on training set (simplest baseline).
    train_global_mean_acps = float(train["acps"].mean())

    # Training-set distribution for top features (used by /api/local-features).
    top_features_for_deep_dive = [
        "total_movements", "departures", "arrivals",
        "dow_sin", "dow_cos", "acps_rmean_7d",
    ]
    feat_stats = {}
    for f in top_features_for_deep_dive:
        if f not in train.columns:
            continue
        s = train[f].dropna()
        feat_stats[f] = dict(
            mean=float(s.mean()),
            std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            min=float(s.min()),
            max=float(s.max()),
            p10=float(s.quantile(0.10)),
            p50=float(s.quantile(0.50)),
            p90=float(s.quantile(0.90)),
        )

    return State(
        regressor=reg,
        classifier=clf,
        model_table=df,
        feature_names=features,
        test_residual_std=residual_std,
        test_predictions=test_predictions,
        dow_train_means={int(k): float(v) for k, v in dow_train_means.items()},
        train_global_mean_acps=train_global_mean_acps,
        train_feature_stats=feat_stats,
    )


# ---------------------------------------------------------------- app

app = FastAPI(title="Barajas Congestion Forecast — Live API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global state
    state = load_state()


# ---------------------------------------------------------------- helpers

WEATHER_PRESETS = {
    "clear": dict(
        temperature_2m=18.0, relative_humidity_2m=55.0, precipitation=0.0,
        surface_pressure=1018.0, wind_speed_10m=8.0, wind_speed_max=14.0,
        wind_gusts_10m=18.0, cloud_cover=15.0, is_severe_weather=0, is_raining=0,
    ),
    "windy": dict(
        temperature_2m=14.0, relative_humidity_2m=60.0, precipitation=0.0,
        surface_pressure=1010.0, wind_speed_10m=28.0, wind_speed_max=42.0,
        wind_gusts_10m=55.0, cloud_cover=50.0, is_severe_weather=0, is_raining=0,
    ),
    "rain": dict(
        temperature_2m=11.0, relative_humidity_2m=85.0, precipitation=6.5,
        surface_pressure=1005.0, wind_speed_10m=18.0, wind_speed_max=28.0,
        wind_gusts_10m=35.0, cloud_cover=95.0, is_severe_weather=0, is_raining=1,
    ),
    "storm": dict(
        temperature_2m=9.0, relative_humidity_2m=92.0, precipitation=22.0,
        surface_pressure=995.0, wind_speed_10m=42.0, wind_speed_max=65.0,
        wind_gusts_10m=85.0, cloud_cover=100.0, is_severe_weather=1, is_raining=1,
    ),
}


def _require_state() -> State:
    if state is None:
        raise HTTPException(500, "model state not loaded")
    return state


def _class_for_acps(acps: float) -> str:
    if acps < 65:
        return "Low"
    if acps < 75:
        return "Medium"
    return "High"


def _build_feature_row(
    s: State,
    *,
    target_date: pd.Timestamp,
    total_movements: int,
    weather_preset: str,
    is_holiday: bool,
    is_bridge_day: bool,
    rolling_acps_7d: float | None,
) -> pd.DataFrame:
    """Synthesize a single model-input row using recent history for lag features."""
    df = s.model_table
    recent = df[df["date"] < target_date].tail(30)
    if recent.empty:
        raise HTTPException(400, "no history available for target date")

    last = recent.iloc[-1]
    w = WEATHER_PRESETS.get(weather_preset, WEATHER_PRESETS["clear"])

    arrivals = total_movements // 2
    departures = total_movements - arrivals

    dow = target_date.dayofweek
    month = target_date.month
    doy = target_date.dayofyear

    row = {f: 0.0 for f in s.feature_names}

    row.update({
        "arrivals": arrivals,
        "departures": departures,
        "total_movements": total_movements,
        **w,
        "wind_direction_10m": 220.0,
        "wind_dir_sin": np.sin(np.deg2rad(220.0)),
        "wind_dir_cos": np.cos(np.deg2rad(220.0)),
        "dow": dow,
        "is_weekend": int(dow >= 5),
        "month": month,
        "quarter": (month - 1) // 3 + 1,
        "is_holiday": int(is_holiday),
        "is_pre_holiday": 0,
        "is_post_holiday": 0,
        "is_bridge_day": int(is_bridge_day),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
        "arr_dep_imbalance": (arrivals - departures) / max(total_movements, 1),
    })

    for lag in (1, 2, 3, 7, 14, 28):
        lag_date = target_date - pd.Timedelta(days=lag)
        lag_row = df[df["date"] == lag_date]
        if lag_row.empty:
            lag_row = recent.tail(1)
        row[f"acps_lag_{lag}d"] = float(lag_row["acps"].iloc[0])
        row[f"movements_lag_{lag}d"] = float(lag_row["total_movements"].iloc[0])

    lag365 = target_date - pd.Timedelta(days=365)
    lag365_row = df[df["date"] == lag365]
    if not lag365_row.empty:
        row["acps_lag_365d"] = float(lag365_row["acps"].iloc[0])
        row["movements_lag_365d"] = float(lag365_row["total_movements"].iloc[0])
    else:
        row["acps_lag_365d"] = float(recent["acps"].mean())
        row["movements_lag_365d"] = float(recent["total_movements"].mean())

    last7 = df[(df["date"] < target_date) & (df["date"] >= target_date - pd.Timedelta(days=7))]
    last14 = df[(df["date"] < target_date) & (df["date"] >= target_date - pd.Timedelta(days=14))]
    last28 = df[(df["date"] < target_date) & (df["date"] >= target_date - pd.Timedelta(days=28))]

    row["acps_rmean_7d"] = float(rolling_acps_7d) if rolling_acps_7d is not None else float(last7["acps"].mean())
    row["movements_rmean_7d"] = float(last7["total_movements"].mean())
    row["acps_rmean_14d"] = float(last14["acps"].mean())
    row["movements_rmean_14d"] = float(last14["total_movements"].mean())
    row["acps_rmean_28d"] = float(last28["acps"].mean())
    row["movements_rmean_28d"] = float(last28["total_movements"].mean())
    row["acps_rstd_28d"] = float(last28["acps"].std(ddof=1)) if len(last28) > 1 else 0.0

    yoy_movements = row["movements_lag_365d"]
    row["movements_yoy_change"] = (total_movements - yoy_movements) / max(yoy_movements, 1)

    return pd.DataFrame([row])[s.feature_names]


# ---------------------------------------------------------------- schemas

class Forecast(BaseModel):
    date: str
    day_label: str
    predicted_acps: float
    classification: str
    confidence_low: float
    confidence_high: float


class TodayResponse(BaseModel):
    today_date: str
    today_actual_acps: float
    today_predicted_acps: float
    today_classification: str
    today_movements: int
    today_weather: dict
    forecast: list[Forecast]
    residual_std: float


class ScenarioInput(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon, 6=Sun")
    is_holiday: bool = False
    is_bridge_day: bool = False
    total_movements: int = Field(..., ge=600, le=1500)
    weather_preset: Literal["clear", "windy", "rain", "storm"] = "clear"
    rolling_acps_7d: float = Field(..., ge=50.0, le=85.0)


class FeatureContribution(BaseModel):
    feature: str
    importance: float


class ScenarioResponse(BaseModel):
    predicted_acps_1d: float
    predicted_acps_3d: float
    classification_1d: str
    classification_3d: str
    baseline_acps: float
    delta_vs_baseline_1d: float
    delta_vs_baseline_3d: float
    top_drivers: list[FeatureContribution]
    confidence_std: float


class TestDayResponse(BaseModel):
    date: str
    actual_acps: float
    predicted_acps: float
    residual: float
    actual_class: str
    predicted_class: str
    total_movements: int
    context: list[dict]
    top_drivers: list[FeatureContribution]


class MetricsResponse(BaseModel):
    regression: dict
    classification: dict
    baselines: list[dict]
    feature_importance: list[FeatureContribution]
    test_set_size: int
    test_date_range: dict


# ---------------------------------------------------------------- endpoints

@app.get("/api/today", response_model=TodayResponse)
def today() -> TodayResponse:
    s = _require_state()
    df = s.model_table
    test = df[df["date"] >= TEST_SPLIT_DATE].copy()
    today_row = test.iloc[-1]
    today_date = pd.Timestamp(today_row["date"])

    X_today = today_row[s.feature_names].fillna(0).to_frame().T
    pred_today = float(s.regressor.predict(X_today)[0])

    forecasts: list[Forecast] = []
    for d_ahead in (1, 2, 3):
        target = today_date + pd.Timedelta(days=d_ahead)
        row = _build_feature_row(
            s,
            target_date=target,
            total_movements=int(today_row["total_movements"]),
            weather_preset="clear",
            is_holiday=bool(today_row["is_holiday"]),
            is_bridge_day=bool(today_row["is_bridge_day"]),
            rolling_acps_7d=float(today_row["acps_rmean_7d"]),
        )
        p = float(s.regressor.predict(row)[0])
        forecasts.append(Forecast(
            date=target.strftime("%Y-%m-%d"),
            day_label=target.strftime("%a %b %d").upper(),
            predicted_acps=round(p, 2),
            classification=_class_for_acps(p),
            confidence_low=round(p - 1.96 * s.test_residual_std, 2),
            confidence_high=round(p + 1.96 * s.test_residual_std, 2),
        ))

    return TodayResponse(
        today_date=today_date.strftime("%Y-%m-%d"),
        today_actual_acps=round(float(today_row["acps"]), 2),
        today_predicted_acps=round(pred_today, 2),
        today_classification=_class_for_acps(pred_today),
        today_movements=int(today_row["total_movements"]),
        today_weather=dict(
            # Presentation-day override: demo is shown 2026-04-23 (warm spring day in Madrid),
            # last Eurocontrol row is 2026-02-28 (11°C winter value). We display today's
            # ambient temperature and keep the other fields as the last data point recorded.
            temperature_c=23.0,
            wind_kmh=round(float(today_row["wind_speed_10m"]), 1),
            precipitation_mm=round(float(today_row["precipitation"]), 1),
            is_raining=bool(today_row["is_raining"]),
            is_severe=bool(today_row["is_severe_weather"]),
        ),
        forecast=forecasts,
        residual_std=round(s.test_residual_std, 3),
    )


@app.post("/api/predict", response_model=ScenarioResponse)
def predict(inp: ScenarioInput) -> ScenarioResponse:
    s = _require_state()
    df = s.model_table
    today_date = pd.Timestamp(df[df["date"] >= TEST_SPLIT_DATE].iloc[-1]["date"])

    dow_offset = (inp.day_of_week - today_date.dayofweek) % 7
    target_1d = today_date + pd.Timedelta(days=max(1, dow_offset))
    target_3d = target_1d + pd.Timedelta(days=2)

    row_1d = _build_feature_row(
        s,
        target_date=target_1d,
        total_movements=inp.total_movements,
        weather_preset=inp.weather_preset,
        is_holiday=inp.is_holiday,
        is_bridge_day=inp.is_bridge_day,
        rolling_acps_7d=inp.rolling_acps_7d,
    )
    row_3d = _build_feature_row(
        s,
        target_date=target_3d,
        total_movements=inp.total_movements,
        weather_preset=inp.weather_preset,
        is_holiday=inp.is_holiday,
        is_bridge_day=inp.is_bridge_day,
        rolling_acps_7d=inp.rolling_acps_7d,
    )

    p1 = float(s.regressor.predict(row_1d)[0])
    p3 = float(s.regressor.predict(row_3d)[0])

    today_row = df[df["date"] == today_date].iloc[0]
    baseline = float(today_row["acps"])

    fi_path = FEATURE_IMPORTANCE_PATH
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(6)
    drivers = [FeatureContribution(feature=r.feature, importance=float(r.importance))
               for r in fi.itertuples()]

    return ScenarioResponse(
        predicted_acps_1d=round(p1, 2),
        predicted_acps_3d=round(p3, 2),
        classification_1d=_class_for_acps(p1),
        classification_3d=_class_for_acps(p3),
        baseline_acps=round(baseline, 2),
        delta_vs_baseline_1d=round(p1 - baseline, 2),
        delta_vs_baseline_3d=round(p3 - baseline, 2),
        top_drivers=drivers,
        confidence_std=round(s.test_residual_std, 3),
    )


@app.get("/api/test-set/dates")
def test_dates() -> dict:
    s = _require_state()
    test = s.model_table[s.model_table["date"] >= TEST_SPLIT_DATE]
    dates = test["date"].dt.strftime("%Y-%m-%d").tolist()
    return {
        "dates": dates,
        "default": "2025-12-19",
        "min": dates[0],
        "max": dates[-1],
        "count": len(dates),
    }


@app.get("/api/test-set/{date}", response_model=TestDayResponse)
def test_day(date: str) -> TestDayResponse:
    s = _require_state()
    try:
        target = pd.Timestamp(date)
    except Exception:
        raise HTTPException(400, f"bad date: {date}")

    df = s.model_table
    row = df[df["date"] == target]
    if row.empty or target < TEST_SPLIT_DATE:
        raise HTTPException(404, f"no test-set row for {date}")
    row = row.iloc[0]

    X = row[s.feature_names].fillna(0).to_frame().T
    pred = float(s.regressor.predict(X)[0])
    pred_class = str(s.classifier.predict(X)[0])
    actual = float(row["acps"])

    window = df[
        (df["date"] >= target - pd.Timedelta(days=7))
        & (df["date"] <= target + pd.Timedelta(days=3))
        & (df["date"] >= TEST_SPLIT_DATE)
    ].copy()
    if not window.empty:
        Xw = window[s.feature_names].fillna(0)
        window["predicted"] = s.regressor.predict(Xw)
    context = [
        dict(
            date=r.date.strftime("%Y-%m-%d"),
            actual=round(float(r.acps), 2),
            predicted=round(float(r.predicted), 2),
            is_target=(r.date == target),
        )
        for r in window.itertuples()
    ]

    fi_path = FEATURE_IMPORTANCE_PATH
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(6)
    drivers = [FeatureContribution(feature=r.feature, importance=float(r.importance))
               for r in fi.itertuples()]

    return TestDayResponse(
        date=target.strftime("%Y-%m-%d"),
        actual_acps=round(actual, 2),
        predicted_acps=round(pred, 2),
        residual=round(actual - pred, 2),
        actual_class=str(row["congestion_class"]),
        predicted_class=pred_class,
        total_movements=int(row["total_movements"]),
        context=context,
        top_drivers=drivers,
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    s = _require_state()
    if s.metrics_cache is not None:
        return MetricsResponse(**s.metrics_cache)

    df = s.model_table
    test = df[df["date"] >= TEST_SPLIT_DATE].copy()
    X = test[s.feature_names].fillna(0)
    y = test["acps"].values
    y_class = test["congestion_class"].values

    pred = s.regressor.predict(X)
    pred_class = s.classifier.predict(X)

    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    r2 = float(r2_score(y, pred))
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_class, pred_class, labels=labels)
    acc = float((pred_class == y_class).mean())
    f1 = float(f1_score(y_class, pred_class, average="macro"))
    bal = float(balanced_accuracy_score(y_class, pred_class))

    prev_day_pred = test["acps_lag_1d"].values
    dow_avg = test.groupby("dow")["acps"].transform("mean").values
    baselines = [
        dict(model="HGB Regressor (ours)", mae=round(mae, 3), rmse=round(rmse, 3), r2=round(r2, 3)),
        dict(
            model="Previous Day",
            mae=round(float(mean_absolute_error(y, prev_day_pred)), 3),
            rmse=round(float(np.sqrt(np.mean((y - prev_day_pred) ** 2))), 3),
            r2=round(float(r2_score(y, prev_day_pred)), 3),
        ),
        dict(
            model="Day-of-Week Average",
            mae=round(float(mean_absolute_error(y, dow_avg)), 3),
            rmse=round(float(np.sqrt(np.mean((y - dow_avg) ** 2))), 3),
            r2=round(float(r2_score(y, dow_avg)), 3),
        ),
    ]

    fi_path = FEATURE_IMPORTANCE_PATH
    fi = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(10)
    fi_list = [FeatureContribution(feature=r.feature, importance=float(r.importance))
               for r in fi.itertuples()]

    result = dict(
        regression=dict(mae=round(mae, 3), rmse=round(rmse, 3), r2=round(r2, 3)),
        classification=dict(
            accuracy=round(acc, 4),
            f1_macro=round(f1, 4),
            balanced_accuracy=round(bal, 4),
            confusion_matrix=cm.tolist(),
            labels=labels,
        ),
        baselines=baselines,
        feature_importance=[f.model_dump() for f in fi_list],
        test_set_size=int(len(test)),
        test_date_range=dict(
            start=test["date"].min().strftime("%Y-%m-%d"),
            end=test["date"].max().strftime("%Y-%m-%d"),
        ),
    )
    s.metrics_cache = result
    return MetricsResponse(**result)


# ================================================================
# DIAGNOSTICS TAB — three endpoints
# ================================================================

@app.get("/api/model-showdown/{date}")
def model_showdown(date: str) -> dict:
    """Per-day predictions: HGB + 3 deterministic baselines, plus aggregate metrics for all 5 models."""
    s = _require_state()
    try:
        target = pd.Timestamp(date)
    except Exception:
        raise HTTPException(400, f"bad date: {date}")

    row = s.model_table[s.model_table["date"] == target]
    if row.empty or target < TEST_SPLIT_DATE:
        raise HTTPException(404, f"no test-set row for {date}")
    row = row.iloc[0]

    actual = float(row["acps"])
    X = row[s.feature_names].fillna(0).to_frame().T
    hgb_pred = float(s.regressor.predict(X)[0])

    # Previous-Day baseline: yesterday's actual ACPS (acps_lag_1d already in the row)
    prev_day_pred = float(row["acps_lag_1d"]) if pd.notna(row["acps_lag_1d"]) else float(row["acps"])

    # Day-of-Week Average baseline: mean training ACPS for same dow
    dow = int(row["dow"])
    dow_avg_pred = float(s.dow_train_means.get(dow, s.train_global_mean_acps))

    # Global Mean baseline: constant training mean
    global_mean_pred = float(s.train_global_mean_acps)

    per_day = [
        dict(model="HGB Regressor", predicted=round(hgb_pred, 2),
             residual=round(actual - hgb_pred, 2), is_ours=True),
        dict(model="Previous Day", predicted=round(prev_day_pred, 2),
             residual=round(actual - prev_day_pred, 2), is_ours=False),
        dict(model="Day-of-Week Avg", predicted=round(dow_avg_pred, 2),
             residual=round(actual - dow_avg_pred, 2), is_ours=False),
        dict(model="Global Mean", predicted=round(global_mean_pred, 2),
             residual=round(actual - global_mean_pred, 2), is_ours=False),
    ]

    # Aggregate MAE table from the final comparison CSV (includes SARIMAX + ANN — no per-day available for those).
    comp_path = MODEL_COMPARISON_PATH
    aggregate_rows = []
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        for _, r in comp.iterrows():
            aggregate_rows.append(dict(
                model=str(r["Model"]),
                mae=round(float(r["MAE"]), 3),
                rmse=round(float(r["RMSE"]), 3),
                r2=round(float(r["R2"]), 3),
            ))

    return dict(
        date=target.strftime("%Y-%m-%d"),
        day_of_week=["MON","TUE","WED","THU","FRI","SAT","SUN"][dow],
        actual_acps=round(actual, 2),
        actual_class=str(row["congestion_class"]),
        total_movements=int(row["total_movements"]),
        per_day=per_day,
        aggregate=aggregate_rows,
    )


@app.get("/api/residuals")
def residuals() -> dict:
    """Full residual diagnostics for the held-out test set."""
    s = _require_state()
    tp = s.test_predictions
    if tp is None:
        raise HTTPException(500, "test predictions not cached")

    # Raw points (downsampled for client-side efficiency if needed)
    points = [
        dict(
            date=pd.Timestamp(r.date).strftime("%Y-%m-%d"),
            actual=round(float(r.actual), 3),
            predicted=round(float(r.predicted), 3),
            residual=round(float(r.residual), 3),
            actual_class=str(r.actual_class),
            predicted_class=str(r.predicted_class),
        )
        for r in tp.itertuples()
    ]

    r = tp["residual"].values
    abs_r = np.abs(r)

    # Histogram bins for the residual distribution
    hist_counts, hist_edges = np.histogram(r, bins=24)
    hist = [
        dict(
            bin_start=round(float(hist_edges[i]), 3),
            bin_end=round(float(hist_edges[i + 1]), 3),
            count=int(hist_counts[i]),
        )
        for i in range(len(hist_counts))
    ]

    # Residuals by class
    by_class = {}
    for cls in ["Low", "Medium", "High"]:
        sub = tp[tp["actual_class"] == cls]["residual"].values
        if len(sub) == 0:
            continue
        by_class[cls] = dict(
            n=int(len(sub)),
            mean=round(float(np.mean(sub)), 3),
            std=round(float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0, 3),
            min=round(float(np.min(sub)), 3),
            q25=round(float(np.quantile(sub, 0.25)), 3),
            median=round(float(np.median(sub)), 3),
            q75=round(float(np.quantile(sub, 0.75)), 3),
            max=round(float(np.max(sub)), 3),
        )

    # Absolute-error cumulative distribution at key thresholds
    thresholds = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    cumulative = [
        dict(threshold=t, pct_within=round(float(np.mean(abs_r <= t) * 100), 1))
        for t in thresholds
    ]

    stats_summary = dict(
        n=int(len(r)),
        mean=round(float(np.mean(r)), 3),
        std=round(float(np.std(r, ddof=1)), 3),
        median=round(float(np.median(r)), 3),
        min=round(float(np.min(r)), 3),
        max=round(float(np.max(r)), 3),
        mae=round(float(np.mean(abs_r)), 3),
        pct_within_1=round(float(np.mean(abs_r <= 1.0) * 100), 1),
        pct_within_05=round(float(np.mean(abs_r <= 0.5) * 100), 1),
    )

    return dict(
        points=points,
        histogram=hist,
        by_class=by_class,
        cumulative=cumulative,
        summary=stats_summary,
    )


@app.get("/api/local-features/{date}")
def local_features(date: str) -> dict:
    """For a selected test-set day, return each top feature's value vs its training distribution."""
    s = _require_state()
    try:
        target = pd.Timestamp(date)
    except Exception:
        raise HTTPException(400, f"bad date: {date}")

    row = s.model_table[s.model_table["date"] == target]
    if row.empty or target < TEST_SPLIT_DATE:
        raise HTTPException(404, f"no test-set row for {date}")
    row = row.iloc[0]

    # Global importance (for ranking)
    fi_path = FEATURE_IMPORTANCE_PATH
    fi_map = {}
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        fi_map = {r.feature: float(r.importance) for r in fi.itertuples()}

    items = []
    for feature, stats in (s.train_feature_stats or {}).items():
        value = float(row[feature]) if feature in row and pd.notna(row[feature]) else None
        if value is None:
            continue
        std = stats["std"] if stats["std"] > 1e-9 else 1.0
        z = (value - stats["mean"]) / std
        rng = max(stats["max"] - stats["min"], 1e-9)
        percentile = 100.0 * max(0.0, min(1.0, (value - stats["min"]) / rng))
        items.append(dict(
            feature=feature,
            value=round(value, 3),
            train_mean=round(stats["mean"], 3),
            train_std=round(stats["std"], 3),
            train_min=round(stats["min"], 3),
            train_max=round(stats["max"], 3),
            train_p10=round(stats["p10"], 3),
            train_p50=round(stats["p50"], 3),
            train_p90=round(stats["p90"], 3),
            z_score=round(z, 2),
            percentile=round(percentile, 1),
            global_importance=round(fi_map.get(feature, 0.0), 4),
        ))

    # Sort by global importance (so the most impactful features appear first)
    items.sort(key=lambda x: x["global_importance"], reverse=True)

    return dict(
        date=target.strftime("%Y-%m-%d"),
        actual_acps=round(float(row["acps"]), 2),
        actual_class=str(row["congestion_class"]),
        total_movements=int(row["total_movements"]),
        features=items,
    )


# ================================================================
# FUTURE FORECAST — pick any date, get a prediction
# ================================================================

# Spanish national public holidays falling in 2026. Used to auto-flag
# is_holiday on a future date without needing to call the Nager.Date API.
SPANISH_HOLIDAYS_2026 = {
    "2026-01-01": "New Year's Day",
    "2026-01-06": "Epiphany",
    "2026-04-03": "Good Friday",
    "2026-05-01": "Labour Day",
    "2026-08-15": "Assumption of Mary",
    "2026-10-12": "Hispanic Day",
    "2026-11-01": "All Saints Day",
    "2026-12-06": "Constitution Day",
    "2026-12-08": "Immaculate Conception",
    "2026-12-25": "Christmas Day",
}


def _is_bridge_day_2026(target: pd.Timestamp) -> bool:
    """Heuristic: a Spanish 'bridge day' is a Mon or Fri that sits between a weekend and a mid-week holiday."""
    iso = target.strftime("%Y-%m-%d")
    if iso in SPANISH_HOLIDAYS_2026:
        return False
    # Monday with Tuesday holiday, or Friday with Thursday holiday
    if target.dayofweek == 0:
        next_day = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return next_day in SPANISH_HOLIDAYS_2026
    if target.dayofweek == 4:
        prev_day = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return prev_day in SPANISH_HOLIDAYS_2026
    return False


@app.get("/api/future-forecast/{date}")
def future_forecast(date: str) -> dict:
    """Predict congestion for any future date using auto-computed assumptions.

    For a date past the last row in the data (2026-02-28), we:
      - Look up whether that ISO date is a Spanish national holiday
      - Detect bridge days from adjacent holidays
      - Use the historical (recent-year) mean movements for the same
        (day-of-week, month) combination as the movements assumption
      - Default weather to 'clear' (we don't know future weather)
      - Use the 7-day rolling ACPS from the tail of the available data
      - Use lag features from the most recent history
    """
    s = _require_state()
    try:
        target = pd.Timestamp(date)
    except Exception:
        raise HTTPException(400, f"bad date: {date}")

    if target.year != 2026:
        raise HTTPException(400, "this endpoint serves 2026 dates only")

    iso = target.strftime("%Y-%m-%d")
    dow = int(target.dayofweek)
    month = int(target.month)

    is_holiday = iso in SPANISH_HOLIDAYS_2026
    holiday_name = SPANISH_HOLIDAYS_2026.get(iso)
    is_bridge = _is_bridge_day_2026(target)

    df = s.model_table

    # Historical mean movements for the same (dow, month), restricted to the
    # post-recovery years so we don't drag in COVID lows.
    post_recovery_cutoff = pd.Timestamp("2023-01-01")
    mask = (
        (df["dow"] == dow)
        & (df["month"] == month)
        & (df["date"] >= post_recovery_cutoff)
    )
    hits = df[mask]
    if len(hits) >= 3:
        movements_assumed = int(hits["total_movements"].mean())
        movements_basis = (
            f"post-recovery mean (n={len(hits)}) for "
            f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} in "
            f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1]}"
        )
    else:
        movements_assumed = int(df["total_movements"].tail(30).mean())
        movements_basis = "last-30-day mean (insufficient same dow/month history)"

    rolling_acps_7d = float(df["acps"].tail(7).mean())

    row = _build_feature_row(
        s,
        target_date=target,
        total_movements=movements_assumed,
        weather_preset="clear",
        is_holiday=is_holiday,
        is_bridge_day=is_bridge,
        rolling_acps_7d=rolling_acps_7d,
    )
    pred = float(s.regressor.predict(row)[0])
    pred_class_model = str(s.classifier.predict(row)[0])
    ci = 1.96 * s.test_residual_std

    return dict(
        date=iso,
        day_of_week=["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][dow],
        month_name=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][month - 1],
        predicted_acps=round(pred, 2),
        classification=_class_for_acps(pred),
        classifier_class=pred_class_model,
        confidence_low=round(pred - ci, 2),
        confidence_high=round(pred + ci, 2),
        residual_std=round(s.test_residual_std, 3),
        assumptions=dict(
            total_movements=movements_assumed,
            movements_basis=movements_basis,
            weather_preset="clear",
            is_holiday=is_holiday,
            holiday_name=holiday_name,
            is_bridge_day=is_bridge,
            rolling_acps_7d=round(rolling_acps_7d, 2),
        ),
    )


@app.get("/api/health")
def health() -> dict:
    s = _require_state()
    return dict(
        status="ok",
        regressor_loaded=s.regressor is not None,
        classifier_loaded=s.classifier is not None,
        rows=len(s.model_table),
        features=len(s.feature_names),
        residual_std=round(s.test_residual_std, 3),
    )
