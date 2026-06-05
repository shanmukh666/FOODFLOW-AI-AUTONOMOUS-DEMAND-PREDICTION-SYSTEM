"""
KITCH·AI — Flask REST API
Endpoints:
  GET  /api/health               — system status + model metadata
  GET  /api/menu                 — all menu items
  POST /api/train                — trigger full training pipeline
  GET  /api/forecast/<item_id>   — 7-day forecast for one item
  GET  /api/forecast/all         — forecast for all items
  GET  /api/features             — feature importance from trained model
  GET  /api/metrics              — model accuracy metrics
  GET  /api/history/<item_id>    — historical sales (last N days)
  POST /api/simulate             — simulate a custom scenario
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, pickle, traceback
from datetime import date, timedelta, datetime

app = Flask(__name__)
CORS(app)  # Allow frontend to call from any origin

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}

def err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

# ── Health ─────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    meta = load_meta()
    model_ready = os.path.exists(os.path.join(MODEL_DIR, "xgb_demand.pkl"))
    data_ready  = os.path.exists(os.path.join(DATA_DIR, "pos_sales.csv"))

    return jsonify({
        "status":      "online",
        "timestamp":   datetime.now().isoformat(),
        "model_ready": model_ready,
        "data_ready":  data_ready,
        "trained_at":  meta.get("trained_at"),
        "cv_mape":     meta.get("cv_mean_mape"),
        "holdout_mape": meta.get("holdout_mape"),
        "train_rows":  meta.get("train_rows"),
        "date_range":  meta.get("date_range"),
        "version":     "2.4.0",
    })

# ── Menu ───────────────────────────────────────────────────────────────────────
@app.route("/api/menu")
def menu():
    path = os.path.join(DATA_DIR, "menu.json")
    if not os.path.exists(path):
        return err("Menu data not found. Run data generation first.", 404)
    with open(path) as f:
        items = json.load(f)
    return jsonify({"items": items, "count": len(items)})

# ── Train ──────────────────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train():
    """
    Runs the full pipeline: generate → feature-engineer → train.
    Can take 30-60s for 18 months of data. In production, run async.
    """
    try:
        from generate_pos_data import generate
        from train_model import run_training

        body   = request.get_json(silent=True) or {}
        months = int(body.get("months", 18))

        print(f"[API] /train  months={months}")
        generate(output_dir=DATA_DIR, months=months)
        meta = run_training(data_dir=DATA_DIR)

        return jsonify({
            "status":       "trained",
            "cv_mape":      meta["cv_mean_mape"],
            "holdout_mape": meta["holdout_mape"],
            "train_rows":   meta["train_rows"],
            "waste":        meta["waste_reduction"],
            "trained_at":   meta["trained_at"],
        })
    except Exception as e:
        traceback.print_exc()
        return err(str(e), 500)

# ── Forecast single item ───────────────────────────────────────────────────────
@app.route("/api/forecast/<item_id>")
def forecast_one(item_id):
    try:
        from forecaster import forecast_item
        days  = int(request.args.get("days",  7))
        start_str = request.args.get("start")
        start = date.fromisoformat(start_str) if start_str else date.today()

        print(f"[API] /forecast/{item_id}  days={days}  start={start}")
        result = forecast_item(item_id, days=days, start=start, data_dir=DATA_DIR)
        return jsonify(result)
    except FileNotFoundError as e:
        return err(str(e), 404)
    except Exception as e:
        traceback.print_exc()
        return err(str(e), 500)

# ── Forecast all items ─────────────────────────────────────────────────────────
@app.route("/api/forecast/all")
def forecast_all():
    try:
        from forecaster import forecast_all_items
        days = int(request.args.get("days", 7))
        print(f"[API] /forecast/all  days={days}")
        results = forecast_all_items(days=days, data_dir=DATA_DIR)
        return jsonify({"forecasts": results, "item_count": len(results)})
    except Exception as e:
        traceback.print_exc()
        return err(str(e), 500)

# ── Feature importance ─────────────────────────────────────────────────────────
@app.route("/api/features")
def features():
    meta = load_meta()
    if not meta:
        return err("Model not trained yet.", 404)
    return jsonify({
        "features":   meta.get("feature_importance", []),
        "cv_mape":    meta.get("cv_mean_mape"),
        "holdout_mape": meta.get("holdout_mape"),
    })

# ── Accuracy metrics ───────────────────────────────────────────────────────────
@app.route("/api/metrics")
def metrics():
    meta = load_meta()
    if not meta:
        return err("Model not trained yet.", 404)
    return jsonify({
        "cv_mean_mape":     meta.get("cv_mean_mape"),
        "cv_mean_rmse":     meta.get("cv_mean_rmse"),
        "holdout_mape":     meta.get("holdout_mape"),
        "holdout_rmse":     meta.get("holdout_rmse"),
        "waste_reduction":  meta.get("waste_reduction"),
        "cv_folds":         meta.get("cv_folds", []),
        "trained_at":       meta.get("trained_at"),
        "train_rows":       meta.get("train_rows"),
        "date_range":       meta.get("date_range"),
    })

# ── Historical sales ───────────────────────────────────────────────────────────
@app.route("/api/history/<item_id>")
def history(item_id):
    try:
        pos_path = os.path.join(DATA_DIR, "pos_sales.csv")
        if not os.path.exists(pos_path):
            return err("POS data not found. Run /api/train first.", 404)

        days = int(request.args.get("days", 90))
        pos  = pd.read_csv(pos_path, parse_dates=["date"])
        item = pos[pos["item_id"] == item_id].sort_values("date")

        if item.empty:
            return err(f"Item '{item_id}' not found.", 404)

        cutoff = item["date"].max() - pd.Timedelta(days=days)
        item   = item[item["date"] >= cutoff]

        # Day-of-week averages
        dow_avg = (
            item.groupby("day_of_week")["qty_sold"]
            .mean().round(1).to_dict()
        )
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        records = item[["date", "qty_sold", "revenue", "day_of_week", "event_type"]].copy()
        records["date"] = records["date"].dt.date.astype(str)
        records["qty_sold"]    = records["qty_sold"].astype(int)
        records["day_of_week"] = records["day_of_week"].astype(int)

        return jsonify({
            "item_id":       item_id,
            "item_name":     item["item_name"].iloc[0] if len(item) else item_id,
            "days_returned": len(item),
            "records":       records.to_dict(orient="records"),
            "summary": {
                "mean_daily":   round(float(item["qty_sold"].mean()), 1),
                "max_daily":    int(item["qty_sold"].max()),
                "min_daily":    int(item["qty_sold"].min()),
                "total_qty":    int(item["qty_sold"].sum()),
                "total_rev":    round(float(item["revenue"].sum()), 2),
                "dow_averages": {dow_names[int(k)]: round(float(v), 1)
                                 for k, v in dow_avg.items()},
            },
        })
    except Exception as e:
        traceback.print_exc()
        return err(str(e), 500)

# ── Custom scenario simulation ─────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Simulate a custom scenario by overriding signals.
    Body: { item_id, days, overrides: { rain: bool, event: bool, promo: bool, temp_c: float } }
    """
    try:
        from forecaster import forecast_item, load_model, build_forecast_row
        from feature_engineering import CATEGORY_MAP

        body      = request.get_json(silent=True) or {}
        item_id   = body.get("item_id", "WB01")
        days      = int(body.get("days", 7))
        overrides = body.get("overrides", {})

        # Run baseline forecast
        baseline = forecast_item(item_id, days=days, data_dir=DATA_DIR)

        # Apply overrides — adjust qty manually based on signal factors
        rain_factor  = -0.15 if overrides.get("rain")  else 0
        event_factor = +0.45 if overrides.get("event") else 0
        promo_factor = +0.20 if overrides.get("promo") else 0
        temp_delta   = float(overrides.get("temp_c", 0))
        temp_factor  = 0.005 * temp_delta  # 0.5% per degree C

        total_factor = 1 + rain_factor + event_factor + promo_factor + temp_factor

        simulated_days = []
        for d in baseline["days"]:
            new_qty = max(0, int(round(d["qty"] * total_factor)))
            simulated_days.append({**d, "qty": new_qty, "simulated": True})

        delta_avg = round(
            np.mean([s["qty"] for s in simulated_days]) -
            np.mean([d["qty"] for d in baseline["days"]]), 1
        )

        return jsonify({
            "item_id":      item_id,
            "item_name":    baseline["item_name"],
            "overrides":    overrides,
            "factors": {
                "rain":   rain_factor,
                "event":  event_factor,
                "promo":  promo_factor,
                "temp":   temp_factor,
                "total":  round(total_factor - 1, 3),
            },
            "baseline_days":  baseline["days"],
            "simulated_days": simulated_days,
            "delta_avg_daily": delta_avg,
        })
    except Exception as e:
        traceback.print_exc()
        return err(str(e), 500)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  KITCH·AI  —  Flask Backend  —  Port 5000")
    print("=" * 55)
    print("  POST /api/train          — run full pipeline")
    print("  GET  /api/forecast/WB01  — 7-day item forecast")
    print("  GET  /api/forecast/all   — all items")
    print("  GET  /api/metrics        — model accuracy")
    print("  GET  /api/history/WB01   — historical sales")
    print("  POST /api/simulate       — scenario simulation")
    print("=" * 55)
    app.run(debug=True, port=5000, host="0.0.0.0")
