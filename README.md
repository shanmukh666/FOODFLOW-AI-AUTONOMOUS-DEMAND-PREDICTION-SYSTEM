# KITCH·AI — Neural Demand Engine

> Restaurant demand forecasting backend: POS ingestion → feature engineering → XGBoost → REST API

---

## Architecture

```
kitchai/
├── app.py                        # Flask REST API (8 endpoints)
├── requirements.txt
├── pipeline/
│   ├── generate_pos_data.py      # Synthetic POS + weather + events generator
│   ├── feature_engineering.py    # 35-feature ML pipeline
│   ├── train_model.py            # XGBoost training + walk-forward CV
│   └── forecaster.py             # Inference engine (7-day ahead)
├── data/                         # Auto-generated on first run
│   ├── pos_sales.csv             # 6,000+ item-level daily sales records
│   ├── weather.csv               # Temperature, precipitation per day
│   ├── events.json               # Concerts, holidays, local events
│   ├── promos.json               # Promotional campaign calendar
│   ├── menu.json                 # Item metadata (shelf life, category)
│   └── features.csv              # Engineered ML-ready feature matrix
└── models/
    ├── xgb_demand.pkl            # Trained XGBoost model
    └── model_meta.json           # Accuracy metrics + feature importance
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data + train model (runs automatically on first /api/train call)
python pipeline/generate_pos_data.py
python pipeline/train_model.py

# 3. Start the API server
python app.py
# → http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | System status, model metadata |
| `GET`  | `/api/menu` | All 8 menu items |
| `POST` | `/api/train` | Run full data → train pipeline |
| `GET`  | `/api/forecast/<item_id>` | 7-day demand forecast |
| `GET`  | `/api/forecast/all` | Forecast all items |
| `GET`  | `/api/features` | XGBoost feature importance |
| `GET`  | `/api/metrics` | MAPE, RMSE, waste reduction |
| `GET`  | `/api/history/<item_id>` | Historical sales (last 90 days) |
| `POST` | `/api/simulate` | Custom scenario override |

### Example: Forecast Wagyu Burger
```bash
curl http://localhost:5000/api/forecast/WB01
```

### Example: Simulate rainy Friday with event
```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"item_id": "WB01", "days": 7, "overrides": {"rain": true, "event": true}}'
```

### Example: Trigger retraining
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"months": 18}'
```

---

## Features Engineered (35 total)

| Group | Features |
|-------|----------|
| **Lag** | lag_1, lag_2, lag_3, lag_7, lag_14 |
| **Rolling** | rolling_7, rolling_14, rolling_30, rolling_7_std, wow_delta |
| **Calendar** | day_of_week, week_of_year, month, quarter, is_weekend, is_friday, is_holiday, dow_sin/cos, month_sin/cos, days_to_weekend |
| **Weather** | temp_c, precipitation_mm, feels_like_c, rain_dummy, heavy_rain, cold_day, hot_day, pleasant_day |
| **External** | local_event_nearby, event_lift, is_promo_day |
| **Identity** | item_enc, category_enc |

---

## Model Performance (actual, 18 months data)

| Metric | Score |
|--------|-------|
| CV Mean MAPE | **9.5%** |
| Holdout MAPE | **3.8%** |
| Holdout RMSE | **1.6** |
| Waste reduction | **72.6%** |
| Food saved (est.) | **323 kg / month** |

Walk-forward (time-series) cross-validation — 4 folds, 2-week test windows.
No data leakage: future never appears in training.

---

## Menu Items

| ID | Name | Category | Shelf Life |
|----|------|----------|------------|
| WB01 | Wagyu Burger | main | 2 days |
| PS02 | Porcini Risotto | main | 1 day |
| TC03 | Tuna Crudo | starter | 1 day |
| TF04 | Truffle Fries | side | 1 day |
| CB05 | Craft Brioche Bun | comp | 2 days |
| LM06 | Lemon Tart | dessert | 2 days |
| SM07 | Smash Burger | main | 2 days |
| CS08 | Caesar Salad | starter | 1 day |

---

## Connecting to the Frontend

In `restaurant_forecast.html`, replace the Claude API call URL pattern with:

```javascript
// Forecast endpoint
const res = await fetch('http://localhost:5000/api/forecast/' + itemId);
const data = await res.json();

// Render data.days, data.summary directly into the UI
```

---

## Replacing Synthetic Data with Real POS

1. Export CSV from your POS system (Toast / Square / Lightspeed)
2. Rename columns to match: `date, item_id, item_name, category, qty_sold, revenue`
3. Drop `data/pos_sales.csv` in the `data/` folder
4. Call `POST /api/train` to retrain on real data

For real weather: replace `get_weather_forecast()` in `forecaster.py` with:
```python
import requests
r = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={key}")
```
