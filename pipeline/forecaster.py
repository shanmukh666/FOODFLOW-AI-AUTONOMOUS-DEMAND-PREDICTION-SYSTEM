"""
KITCH·AI — Inference / Forecasting Engine
Loads trained XGBoost model and generates N-day ahead forecasts
for any item, using the latest POS history as context window.
"""

import pandas as pd
import numpy as np
import json, pickle, os
from datetime import date, timedelta, datetime

from feature_engineering import (
    FEATURE_COLS, HOLIDAYS, CATEGORY_MAP,
    add_weather_features
)

MODEL_PATH = "models/xgb_demand.pkl"
DATA_DIR   = "data"

# ── Load model ─────────────────────────────────────────────────────────────────
_MODEL_CACHE = None

def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        with open(MODEL_PATH, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
    return _MODEL_CACHE["model"], _MODEL_CACHE["feature_cols"]

# ── Weather forecast stub (replace with real API) ──────────────────────────────
def get_weather_forecast(start: date, days: int) -> list[dict]:
    """
    Returns a list of weather dicts for each forecast day.
    In production: call OpenWeatherMap 7-day forecast API.
    Here: uses seasonal estimate + small noise.
    """
    rng = np.random.default_rng(int(start.strftime("%Y%m%d")))
    month = start.month
    base_temp = 15 + 15 * np.sin(2 * np.pi * (month - 3) / 12)
    result = []
    for i in range(days):
        d = start + timedelta(days=i)
        rain = float(rng.exponential(4)) if rng.random() < 0.2 else 0.0
        temp = base_temp + rng.normal(0, 2)
        result.append({
            "date":             d,
            "temp_c":           round(temp, 1),
            "precipitation_mm": round(rain, 1),
            "feels_like_c":     round(temp - rain * 0.3, 1),
        })
    return result

# ── Event lookup stub (replace with Eventbrite / local DB) ────────────────────
def get_events_for_range(start: date, days: int) -> dict:
    """
    Returns {date: {"lift": float, "type": str}} for known events.
    In production: query event calendar DB or API.
    """
    with open(f"{DATA_DIR}/events.json") as f:
        raw = json.load(f)
    result = {}
    for i in range(days):
        d = start + timedelta(days=i)
        key = d.isoformat()
        if key in raw:
            result[d] = raw[key]
    return result

# ── Build feature row for a single forecast date ───────────────────────────────
def build_forecast_row(
    target_date:  date,
    item_id:      str,
    item_enc:     int,
    category_enc: int,
    history:      pd.Series,     # qty_sold indexed by date, sorted asc
    weather:      dict,
    events:       dict,
    is_promo:     bool = False,
) -> dict:

    dow   = target_date.weekday()
    month = target_date.month
    week  = target_date.isocalendar()[1]

    def lag(n):
        d = target_date - timedelta(days=n)
        return float(history.get(d, history.iloc[-1] if len(history) else 0))

    def rolling(window):
        vals = [history.get(target_date - timedelta(days=i), np.nan)
                for i in range(1, window + 1)]
        valid = [v for v in vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    lag7  = lag(7)
    lag14 = lag(14)

    event  = events.get(target_date, {})
    w      = weather

    row = {
        # Identity
        "item_enc":           item_enc,
        "category_enc":       category_enc,
        # Lags
        "lag_1":              lag(1),
        "lag_2":              lag(2),
        "lag_3":              lag(3),
        "lag_7":              lag7,
        "lag_14":             lag14,
        # Rolling
        "rolling_7":          rolling(7),
        "rolling_14":         rolling(14),
        "rolling_30":         rolling(30),
        "rolling_7_std":      float(pd.Series([history.get(target_date - timedelta(days=i), np.nan)
                                               for i in range(1, 8)]).std()),
        "wow_delta":          lag7 - lag14,
        # Calendar
        "day_of_week":        dow,
        "week_of_year":       week,
        "month":              month,
        "quarter":            (month - 1) // 3 + 1,
        "is_weekend":         int(dow >= 5),
        "is_friday":          int(dow == 4),
        "is_holiday":         int((month, target_date.day) in HOLIDAYS),
        "dow_sin":            np.sin(2 * np.pi * dow / 7),
        "dow_cos":            np.cos(2 * np.pi * dow / 7),
        "month_sin":          np.sin(2 * np.pi * month / 12),
        "month_cos":          np.cos(2 * np.pi * month / 12),
        "days_to_weekend":    max(0, 4 - dow) if dow <= 4 else 0,
        # Weather
        "temp_c":             w.get("temp_c", 15.0),
        "precipitation_mm":   w.get("precipitation_mm", 0.0),
        "feels_like_c":       w.get("feels_like_c", 15.0),
        "rain_dummy":         int(w.get("precipitation_mm", 0) > 5),
        "heavy_rain":         int(w.get("precipitation_mm", 0) > 15),
        "cold_day":           int(w.get("temp_c", 15) < 5),
        "hot_day":            int(w.get("temp_c", 15) > 28),
        "pleasant_day":       int(15 < w.get("temp_c", 15) < 25 and
                                  w.get("precipitation_mm", 0) < 3),
        # External
        "local_event_nearby": int(bool(event)),
        "event_lift":         float(event.get("lift", 0.0)),
        "is_promo_day":       int(is_promo),
        # Cross-item stubs
        "lag7_WB01":          lag7 if item_id == "CB05" else 0.0,
        "lag7_SM07":          lag7 if item_id == "CB05" else 0.0,
    }
    return row

# ── Level classifier ───────────────────────────────────────────────────────────
def classify_level(qty: float, p25: float, p75: float) -> str:
    if qty >= p75: return "high"
    if qty <= p25: return "low"
    return "med"

def level_note(d: date, weather: dict, events: dict) -> str:
    parts = []
    dow = d.weekday()
    if dow == 4: parts.append("Friday boost")
    if dow == 5: parts.append("Saturday peak")
    if dow == 6: parts.append("Sunday slowdown")
    rain = weather.get("precipitation_mm", 0)
    if rain > 15: parts.append("heavy rain penalty")
    elif rain > 5: parts.append("light rain effect")
    if events.get(d): parts.append(f"event nearby")
    if (d.month, d.day) in HOLIDAYS: parts.append("holiday")
    return parts[0] if parts else "baseline trend"

# ── Main forecast function ─────────────────────────────────────────────────────
def forecast_item(
    item_id:   str,
    days:      int = 7,
    start:     date = None,
    data_dir:  str = DATA_DIR,
) -> dict:
    """
    Generate a demand forecast for `item_id` over the next `days` days.
    Returns structured dict ready for the Flask API to return as JSON.
    """
    model, feature_cols = load_model()

    if start is None:
        start = date.today()

    # Load historical POS for this item
    pos = pd.read_csv(f"{data_dir}/pos_sales.csv", parse_dates=["date"])
    item_hist = (
        pos[pos["item_id"] == item_id]
        .sort_values("date")
        .set_index("date")["qty_sold"]
    )

    # Load menu metadata
    with open(f"{data_dir}/menu.json") as f:
        menu = {m["id"]: m for m in json.load(f)}

    item_meta    = menu.get(item_id, {"category": "main", "name": item_id})
    category_enc = CATEGORY_MAP.get(item_meta.get("category", "main"), 0)
    item_enc     = sorted(pos["item_id"].unique()).index(item_id) \
                   if item_id in pos["item_id"].unique() else 0

    # Weather + events for forecast window
    weather_list = get_weather_forecast(start, days)
    weather_map  = {w["date"]: w for w in weather_list}
    events       = get_events_for_range(start, days)

    forecast_days = []
    day_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

    # Rolling history we update as we predict (for next-day lags)
    rolling_hist = item_hist.copy()

    qtys = []
    rows_for_pred = []

    for i in range(days):
        target = start + timedelta(days=i)
        w      = weather_map.get(target, {"temp_c": 15, "precipitation_mm": 0, "feels_like_c": 15})
        row    = build_forecast_row(
            target_date  = target,
            item_id      = item_id,
            item_enc     = item_enc,
            category_enc = category_enc,
            history      = rolling_hist,
            weather      = w,
            events       = events,
            is_promo     = False,
        )
        rows_for_pred.append(row)

    # Batch predict
    X = pd.DataFrame(rows_for_pred)
    # Align to training feature cols
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0)

    raw_preds = np.clip(model.predict(X), 0, None)

    # Percentiles for level classification
    p25, p75 = np.percentile(raw_preds, 25), np.percentile(raw_preds, 75)

    for i, qty in enumerate(raw_preds):
        target = start + timedelta(days=i)
        w      = weather_map.get(target, {})
        qty_int = max(0, int(round(qty)))
        qtys.append(qty_int)

        forecast_days.append({
            "date":    target.isoformat(),
            "name":    day_names[target.weekday()],
            "qty":     qty_int,
            "level":   classify_level(qty, p25, p75),
            "note":    level_note(target, w, events),
            "temp_c":  w.get("temp_c", 15),
            "rain_mm": w.get("precipitation_mm", 0),
            "event":   bool(events.get(target)),
        })

        # Update rolling history with prediction (for subsequent lags)
        rolling_hist[pd.Timestamp(target)] = qty_int

    # Summary stats
    total_week   = sum(qtys)
    peak_day     = forecast_days[int(np.argmax(qtys))]
    low_day      = forecast_days[int(np.argmin(qtys))]
    avg_daily    = round(np.mean(qtys), 1)

    return {
        "item_id":       item_id,
        "item_name":     item_meta.get("name", item_id),
        "category":      item_meta.get("category", "main"),
        "shelf_days":    item_meta.get("shelf_days", 2),
        "forecast_start": start.isoformat(),
        "days":          forecast_days,
        "summary": {
            "total_week":  total_week,
            "avg_daily":   avg_daily,
            "peak_day":    peak_day["name"],
            "peak_qty":    peak_day["qty"],
            "low_day":     low_day["name"],
            "low_qty":     low_day["qty"],
        },
    }


def forecast_all_items(days: int = 7, data_dir: str = DATA_DIR) -> dict:
    """Forecast all menu items and return combined payload."""
    with open(f"{data_dir}/menu.json") as f:
        menu = json.load(f)
    results = {}
    for item in menu:
        try:
            results[item["id"]] = forecast_item(item["id"], days=days, data_dir=data_dir)
        except Exception as e:
            results[item["id"]] = {"error": str(e)}
    return results


if __name__ == "__main__":
    result = forecast_item("WB01", days=7)
    print(json.dumps(result, indent=2, default=str))
