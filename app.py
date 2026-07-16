"""
KITCH·AI — Live Demo Backend
Lightweight Flask app for deployment on Render/Railway.
Uses pre-baked realistic forecast data so no model file is needed.
Wakes up instantly — no heavy ML dependencies required.
"""
import json, os, random, math
from datetime import date, timedelta
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ── Pre-baked menu ─────────────────────────────────────────────────────────────
MENU = [
    {"id":"WB01","name":"Wagyu Burger",     "category":"main",   "base":48,"shelf_days":2,"price":28},
    {"id":"PS02","name":"Porcini Risotto",  "category":"main",   "base":32,"shelf_days":1,"price":24},
    {"id":"TC03","name":"Tuna Crudo",       "category":"starter","base":26,"shelf_days":1,"price":18},
    {"id":"TF04","name":"Truffle Fries",    "category":"side",   "base":55,"shelf_days":1,"price":12},
    {"id":"CB05","name":"Craft Brioche Bun","category":"comp",   "base":48,"shelf_days":2,"price":0},
    {"id":"LM06","name":"Lemon Tart",       "category":"dessert","base":22,"shelf_days":2,"price":14},
    {"id":"SM07","name":"Smash Burger",     "category":"main",   "base":38,"shelf_days":2,"price":22},
    {"id":"CS08","name":"Caesar Salad",     "category":"starter","base":30,"shelf_days":1,"price":16},
]
MENU_MAP = {m["id"]: m for m in MENU}

DAYS     = ["MON","TUE","WED","THU","FRI","SAT","SUN"]
DOW_MULT = [0.72, 0.75, 0.80, 0.88, 1.15, 1.30, 0.95]
NOTES    = ["baseline trend","pre-weekend build","mid-week uptick",
            "Friday boost","Saturday peak","Sunday slowdown","baseline trend"]

FEATURE_IMPORTANCE = [
    {"name":"lag_7 (last week)",   "importance":88},
    {"name":"day_of_week",         "importance":74},
    {"name":"rolling_14 avg",      "importance":68},
    {"name":"is_holiday",          "importance":55},
    {"name":"precipitation_mm",    "importance":47},
    {"name":"local_event_nearby",  "importance":42},
    {"name":"temperature_c",       "importance":31},
    {"name":"is_promo_day",        "importance":28},
]

def make_forecast(item_id, days=7):
    item  = MENU_MAP.get(item_id, MENU[0])
    base  = item["base"]
    today = date.today()
    result= []
    qtys  = []
    for i in range(days):
        d   = today + timedelta(days=i)
        dow = d.weekday()
        qty = max(1, int(base * DOW_MULT[dow] * (1 + random.uniform(-0.06, 0.06))))
        qtys.append(qty)
        result.append({
            "date":  d.isoformat(),
            "name":  DAYS[dow],
            "qty":   qty,
            "level": "high" if qty >= base*1.1 else ("low" if qty <= base*0.82 else "med"),
            "note":  NOTES[dow],
            "temp_c": round(18 + 6*math.sin(2*math.pi*(d.month-3)/12) + random.uniform(-2,2), 1),
            "rain_mm": round(random.uniform(0,3), 1),
            "event": dow in (4,5) and random.random() < 0.3,
        })
    peak = max(result, key=lambda x: x["qty"])
    low  = min(result, key=lambda x: x["qty"])
    return {
        "item_id":    item_id,
        "item_name":  item["name"],
        "category":   item["category"],
        "shelf_days": item["shelf_days"],
        "forecast_start": date.today().isoformat(),
        "days": result,
        "summary": {
            "total_week": sum(qtys),
            "avg_daily":  round(sum(qtys)/len(qtys), 1),
            "peak_day":   peak["name"],
            "peak_qty":   peak["qty"],
            "low_day":    low["name"],
            "low_qty":    low["qty"],
        }
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status":       "online",
        "model_ready":  True,
        "data_ready":   True,
        "cv_mape":      9.5,
        "holdout_mape": 3.8,
        "train_rows":   6088,
        "date_range":   {"start":"2024-05-01","end":"2026-05-31"},
        "version":      "2.4.0-demo",
        "demo":         True,
    })

@app.route("/api/menu")
def menu():
    return jsonify({"items": MENU, "count": len(MENU)})

@app.route("/api/forecast/<item_id>")
def forecast(item_id):
    from flask import request
    days = int(request.args.get("days", 7))
    return jsonify(make_forecast(item_id, days))

@app.route("/api/forecast/all")
def forecast_all():
    from flask import request
    days = int(request.args.get("days", 7))
    return jsonify({"forecasts": {m["id"]: make_forecast(m["id"], days) for m in MENU}, "item_count": len(MENU)})

@app.route("/api/features")
def features():
    return jsonify({"features": FEATURE_IMPORTANCE, "cv_mape": 9.5, "holdout_mape": 3.8})

@app.route("/api/metrics")
def metrics():
    return jsonify({
        "cv_mean_mape":    9.5,
        "cv_mean_rmse":    4.6,
        "holdout_mape":    3.8,
        "holdout_rmse":    1.6,
        "waste_reduction": {"waste_reduction_pct": 72.6, "kg_food_saved": 323},
        "train_rows":      6088,
        "date_range":      {"start":"2024-05-01","end":"2026-05-31"},
    })

@app.route("/api/history/<item_id>")
def history(item_id):
    item = MENU_MAP.get(item_id, MENU[0])
    base = item["base"]
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    return jsonify({
        "item_id":      item_id,
        "item_name":    item["name"],
        "days_returned":90,
        "summary": {
            "mean_daily":   round(base * 0.98, 1),
            "max_daily":    int(base * 1.35),
            "min_daily":    int(base * 0.65),
            "total_qty":    int(base * 0.98 * 90),
            "total_rev":    round(base * 0.98 * 90 * item["price"], 2),
            "dow_averages": {d: round(base * DOW_MULT[i], 1) for i, d in enumerate(dow_names)},
        }
    })

@app.route("/api/simulate", methods=["POST"])
def simulate():
    from flask import request
    body      = request.get_json(silent=True) or {}
    item_id   = body.get("item_id", "WB01")
    days      = int(body.get("days", 7))
    overrides = body.get("overrides", {})
    baseline  = make_forecast(item_id, days)
    rain_f    = -0.15 if overrides.get("rain")  else 0
    event_f   = +0.45 if overrides.get("event") else 0
    promo_f   = +0.20 if overrides.get("promo") else 0
    temp_f    = 0.005 * float(overrides.get("temp_c", 0))
    total_f   = 1 + rain_f + event_f + promo_f + temp_f
    sim_days  = [{**d, "qty": max(0, int(round(d["qty"]*total_f))), "simulated": True}
                 for d in baseline["days"]]
    return jsonify({
        "item_id":        item_id,
        "item_name":      baseline["item_name"],
        "overrides":      overrides,
        "baseline_days":  baseline["days"],
        "simulated_days": sim_days,
        "factors":        {"rain":rain_f,"event":event_f,"promo":promo_f,"temp":temp_f,"total":round(total_f-1,3)},
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
