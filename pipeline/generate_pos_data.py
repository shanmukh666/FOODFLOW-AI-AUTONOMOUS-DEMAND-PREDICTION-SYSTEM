"""
KITCH·AI — POS Data Generator
Simulates 18 months of realistic restaurant sales data including:
  - Daily item-level order volumes
  - Weather effects (rain suppresses, warm boosts)
  - Event spikes (concerts, sports, holidays)
  - Day-of-week and seasonal patterns
  - Promotional lift periods
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import json, os, random

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Restaurant menu ────────────────────────────────────────────────────────────
MENU_ITEMS = [
    {"id": "WB01", "name": "Wagyu Burger",      "category": "main",    "base_daily": 48, "shelf_days": 2, "price": 28},
    {"id": "PS02", "name": "Porcini Risotto",   "category": "main",    "base_daily": 32, "shelf_days": 1, "price": 24},
    {"id": "TC03", "name": "Tuna Crudo",        "category": "starter", "base_daily": 26, "shelf_days": 1, "price": 18},
    {"id": "TF04", "name": "Truffle Fries",     "category": "side",    "base_daily": 55, "shelf_days": 1, "price": 12},
    {"id": "CB05", "name": "Craft Brioche Bun", "category": "comp",    "base_daily": 48, "shelf_days": 2, "price": 0},
    {"id": "LM06", "name": "Lemon Tart",        "category": "dessert", "base_daily": 22, "shelf_days": 2, "price": 14},
    {"id": "SM07", "name": "Smash Burger",      "category": "main",    "base_daily": 38, "shelf_days": 2, "price": 22},
    {"id": "CS08", "name": "Caesar Salad",      "category": "starter", "base_daily": 30, "shelf_days": 1, "price": 16},
]

# ── Event calendar ─────────────────────────────────────────────────────────────
def build_events(start: date, end: date) -> dict:
    events = {}
    d = start
    while d <= end:
        # Public holidays
        if d.month == 1  and d.day == 1:   events[d] = {"type": "holiday", "name": "New Year",     "lift": 0.6}
        if d.month == 2  and d.day == 14:  events[d] = {"type": "holiday", "name": "Valentine's",  "lift": 0.8}
        if d.month == 7  and d.day == 4:   events[d] = {"type": "holiday", "name": "July 4th",     "lift": 0.5}
        if d.month == 11 and d.day == 28:  events[d] = {"type": "holiday", "name": "Thanksgiving", "lift": -0.3}
        if d.month == 12 and d.day == 25:  events[d] = {"type": "holiday", "name": "Christmas",    "lift": -0.5}
        if d.month == 12 and d.day == 31:  events[d] = {"type": "holiday", "name": "NYE",          "lift": 1.1}
        d += timedelta(days=1)

    # Concerts / sports events (random Fridays and Saturdays)
    d = start
    while d <= end:
        if d.weekday() in (4, 5) and rng.random() < 0.25:
            lift = rng.uniform(0.3, 0.7)
            events[d] = {"type": "concert", "name": f"Local event", "lift": lift}
        d += timedelta(days=1)

    return events

# ── Weather simulation ─────────────────────────────────────────────────────────
def simulate_weather(start: date, end: date) -> pd.DataFrame:
    days = (end - start).days + 1
    dates = [start + timedelta(i) for i in range(days)]

    # Temperature follows seasonal sinusoidal curve (US temperate city)
    t = np.arange(days)
    temp_c = 15 + 15 * np.sin(2 * np.pi * (t - 80) / 365) + rng.normal(0, 3, days)

    # Rain: more in winter/spring
    rain_prob = 0.18 + 0.12 * np.sin(2 * np.pi * (t + 60) / 365)
    rain_mm = np.where(rng.random(days) < rain_prob, rng.exponential(8, days), 0)

    return pd.DataFrame({
        "date": dates,
        "temp_c": np.round(temp_c, 1),
        "precipitation_mm": np.round(rain_mm, 1),
        "feels_like_c": np.round(temp_c - rain_mm * 0.3 + rng.normal(0, 1, days), 1),
    })

# ── Promo calendar ─────────────────────────────────────────────────────────────
def build_promos(start: date, end: date) -> dict:
    promos = {}
    # Monthly happy hour promos — first two weeks of each month
    d = start
    while d <= end:
        if d.day <= 14 and rng.random() < 0.4:
            promos[d] = {"name": "Happy Hour 50%", "item_boost": {"TF04": 0.5, "CB05": 0.2}}
        d += timedelta(days=1)
    return promos

# ── Main generation function ───────────────────────────────────────────────────
def generate(output_dir: str = "data", months: int = 18):
    end   = date.today() - timedelta(days=1)
    start = date(end.year - (1 if months <= 12 else 2),
                 end.month, 1) if months > 12 else \
            date(end.year - 1, end.month, 1)

    print(f"[GENERATE] Simulating {(end - start).days} days from {start} → {end}")

    weather = simulate_weather(start, end)
    events  = build_events(start, end)
    promos  = build_promos(start, end)

    records = []
    d = start
    while d <= end:
        dow     = d.weekday()           # 0=Mon … 6=Sun
        week    = d.isocalendar()[1]
        month   = d.month

        # Day-of-week multiplier (Fri/Sat busiest)
        dow_mult = [0.72, 0.75, 0.80, 0.88, 1.15, 1.30, 0.95][dow]

        # Seasonal multiplier (summer + Dec peak)
        season_mult = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)

        # Event lift
        event = events.get(d, {})
        event_lift = event.get("lift", 0.0)

        # Weather penalty (rain & cold suppress)
        w_row = weather[weather["date"] == d]
        rain  = float(w_row["precipitation_mm"].iloc[0]) if len(w_row) else 0
        temp  = float(w_row["temp_c"].iloc[0]) if len(w_row) else 15
        weather_mult = 1.0
        if rain > 5:  weather_mult -= 0.12
        if rain > 15: weather_mult -= 0.10
        if temp < 5:  weather_mult -= 0.08

        promo = promos.get(d, {})
        promo_boosts = promo.get("item_boost", {})

        for item in MENU_ITEMS:
            base = item["base_daily"]
            noise = rng.normal(1.0, 0.08)
            promo_mult = 1.0 + promo_boosts.get(item["id"], 0)

            qty = max(0, int(
                base
                * dow_mult
                * season_mult
                * weather_mult
                * (1 + event_lift)
                * promo_mult
                * noise
            ))

            records.append({
                "date":         d.isoformat(),
                "item_id":      item["id"],
                "item_name":    item["name"],
                "category":     item["category"],
                "qty_sold":     qty,
                "revenue":      round(qty * item["price"], 2),
                "day_of_week":  dow,
                "week_of_year": week,
                "month":        month,
                "is_weekend":   int(dow >= 5),
                "event_type":   event.get("type", "none"),
                "event_name":   event.get("name", ""),
                "has_promo":    int(bool(promo)),
            })
        d += timedelta(days=1)

    pos_df     = pd.DataFrame(records)
    weather_df = weather.copy()
    weather_df["date"] = weather_df["date"].astype(str)

    os.makedirs(output_dir, exist_ok=True)
    pos_df.to_csv(f"{output_dir}/pos_sales.csv",   index=False)
    weather_df.to_csv(f"{output_dir}/weather.csv", index=False)

    # Save event & promo metadata as JSON
    events_out = {str(k): v for k, v in events.items()}
    promos_out = {str(k): {"name": v["name"]} for k, v in promos.items()}
    with open(f"{output_dir}/events.json", "w") as f: json.dump(events_out, f, indent=2)
    with open(f"{output_dir}/promos.json", "w") as f: json.dump(promos_out, f, indent=2)
    with open(f"{output_dir}/menu.json",   "w") as f: json.dump(MENU_ITEMS, f, indent=2)

    print(f"[GENERATE] ✓ {len(pos_df):,} POS records  |  {len(weather_df)} weather rows")
    print(f"[GENERATE] ✓ {len(events)} events  |  {len(promos)} promo days")
    print(f"[GENERATE] Files saved to: {output_dir}/")
    return pos_df, weather_df

if __name__ == "__main__":
    generate(output_dir="data")
