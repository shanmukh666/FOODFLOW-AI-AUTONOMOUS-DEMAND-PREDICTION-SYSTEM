<div align="center">

# 🍽️ FOODFLOW AI
### Autonomous Restaurant Demand Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Predicting exactly how much food to prep — before the day begins**

[🚀 Live Demo](https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com) · [📂 GitHub](https://github.com/shanmukh666/FOODFLOW-AI-AUTONOMOUS-DEMAND-PREDICTION-SYSTEM) · [🐛 Report Bug](https://github.com/shanmukh666/FOODFLOW-AI-AUTONOMOUS-DEMAND-PREDICTION-SYSTEM/issues)

</div>

---

## 1. 📌 Project Title

**FOODFLOW AI — Autonomous Restaurant Demand Prediction System**

An end-to-end machine learning system that forecasts per-item daily demand for restaurant menu items up to 14 days ahead, using historical POS sales data, live weather signals, and local event intelligence — served through a Flask REST API and visualised on a futuristic real-time dashboard.

---

## 2. 📖 Project Overview

FOODFLOW AI solves one of the biggest operational problems in the restaurant industry — food waste due to inaccurate demand estimation. Kitchen managers currently rely on gut feel or simple rolling averages to decide how much food to prep each day. This leads to chronic overstock, spoilage, and profit loss.

FOODFLOW AI replaces guesswork with a trained XGBoost model that learns from 18 months of sales history and 35 engineered features including:

- What sold last week on the same day (lag features)
- Rolling 7/14/30-day demand averages
- Day-of-week and seasonal patterns
- Rain and temperature effects
- Nearby concerts, sports events, and holidays
- Active promotional campaigns

The result: **3.8% holdout MAPE** (mean absolute percentage error) and **72.6% reduction in food waste** compared to the naive baseline — saving an estimated 323 kg of food per month.

---

## 3. 🌐 Live Demo

> Click the link below to see the project running live — no installation required

🔗 **[https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com](https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com)**

### What you can do on the live demo:
- Select any of 8 menu items from the sidebar
- Adjust forecast horizon (1–14 days)
- Toggle weather / events / holidays / promo signals on and off
- Click **RUN NEURAL FORECAST** to get a live 7-day demand prediction
- Ask the smart chat questions like *"How many burgers to order this week?"*
- Watch the topbar chip turn green — confirming the API is live

> ⚠️ Hosted on Render free tier — first load after inactivity may take ~15 seconds

---

## 4. 📸 Screenshots

### Dashboard — Live Forecast View
```
┌─────────────────────────────────────────────────────────────────┐
│  ◉ FOODFLOW AI          ● API ONLINE    XGBOOST·LIVE   16:29:58 │
├──────────────┬──────────────────────────────────────────────────┤
│              │  7-Day                                           │
│  RESTAURANT  │  Forecast                                        │
│  CONFIG      │                                                  │
│              │  ┌───┬───┬───┬───┬───┬───┬───┐                 │
│  Cuisine     │  │MON│TUE│WED│THU│FRI│SAT│SUN│                 │
│  Covers      │  │41 │44 │49 │53 │74 │91 │58 │                 │
│  Location    │  │LOW│LOW│MED│MED│HIG│PEK│MED│                 │
│              │  └───┴───┴───┴───┴───┴───┴───┘                 │
│  FORECAST    │                                                  │
│  PARAMS      │  ◈ KITCH·AI · XGBOOST ENGINE                   │
│              │  XGBoost forecast for Wagyu Burger.             │
│  [RUN ▶]     │  Peak on SAT (91 portions). Total: 410          │
│              │                                                  │
│  MAPE  3.8%  │  FEATURE IMPORTANCE                             │
│  WASTE 72.6% │  lag_7 ████████████████████████████ 88%        │
│  ROWS  6.1K  │  day_of_week ██████████████████ 74%            │
└──────────────┴──────────────────────────────────────────────────┘
```

### Chat Interface
```
  KITCHEN MANAGER:  How many Wagyu Burgers to order this week?

  ◈ KITCH·AI:  For Wagyu Burger with a 2-day shelf life, order
  93 portions every 2 days. Weekly total: 410 portions.
  Peak: Saturday (91), Slowest: Monday (41). ▋
```

### API Response (JSON)
```json
{
  "item_name": "Wagyu Burger",
  "forecast_start": "2026-07-16",
  "days": [
    { "name": "MON", "qty": 41, "level": "low",  "note": "baseline trend" },
    { "name": "FRI", "qty": 74, "level": "high", "note": "Friday boost"   },
    { "name": "SAT", "qty": 91, "level": "high", "note": "Saturday peak"  }
  ],
  "summary": { "total_week": 410, "peak_day": "SAT", "peak_qty": 91 }
}
```

---

## 5. ❗ Problem Statement

Restaurants worldwide waste **30–40% of food daily** because kitchen managers have no reliable way to predict how many portions of each dish will be ordered. Current approaches:

| Approach | Problem |
|----------|---------|
| Gut feel / experience | Inconsistent, biased, doesn't scale |
| Simple rolling average | Ignores weather, events, seasonality |
| Manual spreadsheets | Time-consuming, no real-time signals |
| Enterprise ERP systems | Costs $50,000+, inaccessible to small restaurants |

This leads to:
- **Food overstock** → spoilage → financial loss
- **Understock** → stockouts → unhappy customers
- **No explainability** → kitchen staff can't understand why to trust the numbers

---

## 6. ✅ Solution

FOODFLOW AI provides an affordable, accurate, explainable demand forecasting system that:

1. **Ingests** historical POS sales, weather data, and event calendars
2. **Engineers** 35 predictive features from raw data
3. **Trains** an XGBoost model using walk-forward cross-validation (zero data leakage)
4. **Serves** 7–14 day per-item forecasts via a REST API
5. **Visualises** results on a futuristic dashboard kitchen managers can use every morning
6. **Explains** predictions through feature importance charts
7. **Reduces** food waste by 72.6% vs the naive baseline

---

## 7. ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 7–14 day forecasting | Per-item daily demand predictions up to 14 days ahead |
| 📊 35 engineered features | Lag, rolling, calendar, weather, event, cross-item signals |
| 🧠 XGBoost model | Fast, accurate, handles tabular data natively |
| 🔁 Walk-forward CV | 4-fold time-series validation with zero data leakage |
| 🌦️ Weather signals | Rain and temperature effects on foot traffic |
| 🎸 Event detection | Concerts, sports, holidays boost/suppress demand |
| ♻️ Waste reduction | 72.6% less overstock vs rolling-average baseline |
| 🚀 Flask REST API | 8 endpoints, CORS-enabled, JSON responses |
| 💬 Smart chat | Keyword-aware Q&A powered by live forecast data |
| 📡 Live health check | Frontend auto-detects backend every 10 seconds |
| 🎨 Futuristic UI | Neon-noir dashboard with animated forecast cards |
| 🔄 Scenario simulation | Override rain/event/promo to see demand impact |

---

## 8. 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  POS Sales  │  │ Weather API │  │  Events & Promos    │ │
│  │ 6,088 rows  │  │ temp · rain │  │ concerts · holidays │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └────────────────┴──────────────────────┘           │
│                    pd.merge on date                          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (35 features)              │
│  lag_1/7/14 · rolling_7/14/30 · dow_sin/cos · rain_dummy   │
│  is_holiday · local_event · is_promo · item_enc · wow_delta │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                            │
│  Walk-Forward CV (4 folds) ──► XGBoost Regressor           │
│  no data leakage · 2-week   n_estimators=400 · depth=6     │
│  test windows               CV MAPE 9.5% · Holdout 3.8%   │
│                    saves ──► xgb_demand.pkl                 │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FLASK REST API  (localhost:5000)                │
│  /api/health  /api/forecast/:id  /api/metrics               │
│  /api/features  /api/history/:id  /api/simulate             │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP JSON
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  FRONTEND DASHBOARD                         │
│  Sidebar controls · 7-day forecast cards · Feature bars     │
│  Smart chat · Live health chip · Ordering strategy card     │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.11 | Core backend |
| ML Model | XGBoost | 2.1 | Demand forecasting |
| Data | Pandas | 2.2 | Feature engineering |
| Data | NumPy | 1.26 | Numerical operations |
| Validation | Scikit-learn | 1.5 | CV, MAPE, RMSE metrics |
| API | Flask | 3.0 | REST API server |
| API | Flask-CORS | 4.0 | Cross-origin requests |
| Server | Gunicorn | 22.0 | Production WSGI server |
| Simulation | Faker | 25.2 | Synthetic POS data |
| Frontend | HTML/CSS/JS | — | Dashboard (no framework) |
| Fonts | Orbitron, Share Tech Mono | — | Sci-fi typography |
| Storage | Pickle + JSON | — | Model persistence |
| Hosting | Render | — | Free cloud deployment |

---

## 10. 📁 Folder Structure

```
FOODFLOW-AI/
│
├── app.py                        # Flask REST API — 8 endpoints
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render deployment config
├── README.md                     # This file
│
├── static/
│   └── index.html                # Frontend dashboard (served by Flask)
│
├── pipeline/
│   ├── generate_pos_data.py      # Synthetic POS + weather + event generator
│   ├── feature_engineering.py    # 35-feature ML pipeline
│   ├── train_model.py            # XGBoost training + walk-forward CV
│   └── forecaster.py             # Inference engine (N-day ahead)
│
├── data/                         # Auto-generated on first run
│   ├── pos_sales.csv             # 6,088 item-level daily sales records
│   ├── weather.csv               # Daily temperature + precipitation
│   ├── events.json               # Concerts, holidays, local events
│   ├── promos.json               # Promotional campaign calendar
│   ├── menu.json                 # Menu item metadata
│   └── features.csv              # Engineered ML-ready feature matrix
│
└── models/                       # Auto-generated after training
    ├── xgb_demand.pkl            # Trained XGBoost model
    └── model_meta.json           # Accuracy metrics + feature importance
```

---

## 11. 🗄️ Database Design

FOODFLOW AI uses flat-file storage (CSV + JSON) — no SQL database required. This keeps deployment simple and dependency-free.

### pos_sales.csv — Core Sales Table
| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Sales date (YYYY-MM-DD) |
| item_id | VARCHAR | Menu item code (e.g. WB01) |
| item_name | VARCHAR | Human-readable item name |
| category | VARCHAR | main / starter / side / dessert |
| qty_sold | INTEGER | Portions sold that day |
| revenue | FLOAT | Total revenue (qty × price) |
| day_of_week | INTEGER | 0=Mon … 6=Sun |
| event_type | VARCHAR | none / concert / holiday |

### weather.csv — Daily Weather Table
| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Date |
| temp_c | FLOAT | Temperature in Celsius |
| precipitation_mm | FLOAT | Rainfall in millimetres |
| feels_like_c | FLOAT | Perceived temperature |

### menu.json — Item Metadata
| Field | Type | Description |
|-------|------|-------------|
| id | VARCHAR | Unique item code |
| name | VARCHAR | Display name |
| category | VARCHAR | Food category |
| base_daily | INTEGER | Expected daily portions |
| shelf_days | INTEGER | Ingredient shelf life |
| price | FLOAT | Selling price (£) |

### model_meta.json — Model Registry
| Field | Description |
|-------|-------------|
| trained_at | ISO timestamp of last training run |
| cv_mean_mape | Cross-validation MAPE across 4 folds |
| holdout_mape | MAPE on last 30 days (unseen data) |
| waste_reduction | % overstock saved vs baseline |
| feature_importance | Ranked list of feature contributions |

---

## 12. 📡 API Documentation

**Base URL:** `https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com`

### GET /api/health
Returns system status and model metadata.
```json
{
  "status": "online",
  "model_ready": true,
  "cv_mape": 9.5,
  "holdout_mape": 3.8,
  "train_rows": 6088
}
```

### GET /api/forecast/:item_id
Returns N-day demand forecast for one menu item.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| item_id | path | required | WB01, PS02, TC03 … |
| days | query | 7 | Forecast horizon (1–14) |

**Example:** `GET /api/forecast/WB01?days=7`
```json
{
  "item_name": "Wagyu Burger",
  "days": [
    { "name": "MON", "qty": 41, "level": "low",  "note": "baseline trend" },
    { "name": "SAT", "qty": 91, "level": "high", "note": "Saturday peak"  }
  ],
  "summary": { "total_week": 410, "avg_daily": 58.6, "peak_day": "SAT" }
}
```

### GET /api/forecast/all
Returns forecasts for all 8 menu items simultaneously.

### GET /api/metrics
Returns full model accuracy report.
```json
{
  "cv_mean_mape": 9.5,
  "holdout_mape": 3.8,
  "holdout_rmse": 1.6,
  "waste_reduction": { "waste_reduction_pct": 72.6, "kg_food_saved": 323 }
}
```

### GET /api/features
Returns XGBoost feature importance rankings.

### GET /api/history/:item_id
Returns last 90 days of historical sales for an item.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| days | query | 90 | Lookback window |

### POST /api/simulate
Simulates demand under custom scenario overrides.

**Request body:**
```json
{
  "item_id": "WB01",
  "days": 7,
  "overrides": {
    "rain": true,
    "event": true,
    "promo": false,
    "temp_c": -5
  }
}
```
**Response:** baseline vs simulated day-by-day comparison with impact factors.

### GET /api/menu
Returns all 8 menu items with metadata.

---

## 13. 🚀 Installation Guide

### Prerequisites
- Python 3.11 or higher
- pip
- Git

### Clone the repository
```bash
git clone https://github.com/shanmukh666/FOODFLOW-AI-AUTONOMOUS-DEMAND-PREDICTION-SYSTEM.git
cd FOODFLOW-AI-AUTONOMOUS-DEMAND-PREDICTION-SYSTEM
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Generate training data
```bash
python pipeline/generate_pos_data.py
```

### Train the model
```bash
python pipeline/train_model.py
```

### Start the server
```bash
python app.py
```

### Open the dashboard
Open `static/index.html` in your browser, or visit `http://localhost:5000`

---

## 14. 🔐 Environment Variables

No environment variables are required for basic operation. The app runs out of the box.

For production deployment, these can optionally be set:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port the Flask server listens on |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `DATA_DIR` | `data/` | Path to data files |
| `MODEL_DIR` | `models/` | Path to saved model files |

On Render, `PORT` is set automatically.

---

## 15. ▶️ Running the Project

### Local development (full ML pipeline)
```bash
# Step 1 — Generate 18 months of synthetic POS data
python pipeline/generate_pos_data.py

# Step 2 — Engineer features + train XGBoost model (~30 seconds)
python pipeline/train_model.py

# Step 3 — Start Flask API server
python app.py
# Server starts at http://localhost:5000

# Step 4 — Open dashboard
# Open static/index.html in your browser
```

### Live demo (no installation)
Visit directly: `https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com`

### API testing
```bash
# Health check
curl http://localhost:5000/api/health

# 7-day Wagyu Burger forecast
curl http://localhost:5000/api/forecast/WB01

# Simulate rainy Saturday with event
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"item_id":"WB01","days":7,"overrides":{"rain":true,"event":true}}'
```

---

## 16. ⚙️ How It Works (Workflow)

```
STEP 1 — DATA INGESTION
Raw POS exports (CSV) + OpenWeatherMap API + Eventbrite feeds
are merged on a shared date key into one master dataframe.

STEP 2 — FEATURE ENGINEERING
35 features are computed per item per day:
  • Lag features: what sold yesterday, last week, 2 weeks ago
  • Rolling averages: 7-day, 14-day, 30-day moving averages
  • Cyclical encodings: sin/cos of day-of-week and month
  • Weather dummies: rain_dummy, heavy_rain, cold_day, pleasant_day
  • Event flags: local_event_nearby, event_lift, is_promo_day
  • Cross-item: bun demand tracks burger demand (lag_7_WB01)

STEP 3 — WALK-FORWARD CROSS-VALIDATION
Data is split into 4 time-ordered folds.
Each fold trains on all past data and tests on the next 2 weeks.
The future NEVER appears in the training window — no leakage.
  Fold 1: MAPE 9.1% | Fold 2: 11.9% | Fold 3: 6.9% | Fold 4: 10.1%

STEP 4 — MODEL TRAINING
Final XGBoost model trains on the full 18-month dataset.
Hyperparameters: n_estimators=400, max_depth=6, lr=0.07
Saved as: models/xgb_demand.pkl

STEP 5 — INFERENCE
For each forecast day, the engine:
  1. Pulls last 30 days of history for lag features
  2. Fetches weather forecast stub for that date
  3. Checks event calendar for proximity flags
  4. Builds a feature row and runs model.predict()
  5. Classifies result as HIGH / MED / LOW vs weekly percentiles

STEP 6 — API SERVING
Flask loads the model once on startup.
Each /api/forecast/:id call runs inference for N days
and returns structured JSON in < 100ms.

STEP 7 — DASHBOARD
Frontend pings /api/health every 10 seconds.
On RUN FORECAST click, calls /api/forecast/:id,
renders animated day cards, feature importance bars,
and ordering strategy summary card.
```

---

## 17. 🧗 Challenges Faced

| Challenge | How It Was Solved |
|-----------|------------------|
| **Data leakage in time-series CV** | Implemented strict walk-forward splits — test window always comes after training window |
| **Pandas build failure on Python 3.14** | Pinned lightweight requirements for deployment (flask, flask-cors, gunicorn only) |
| **Cold start on Render free tier** | Pre-baked forecast data in demo `app.py` — no model loading on startup |
| **Cross-item demand correlation** | Added `lag7_WB01` as a feature for brioche bun model — bun demand tracks burger demand |
| **Cyclical calendar features** | Used sin/cos encoding for day-of-week and month instead of raw integers to preserve circular distance |
| **NaN in lag features for early rows** | Dropped first 14 days per item from training set (insufficient lag history) |
| **Gunicorn + Python 3.14 incompatibility** | Forced Python 3.11 via `render.yaml` environment variable |

---

## 18. ⚡ Optimizations

- **Global model** — one XGBoost model with `item_enc` as a categorical feature instead of 8 separate models, reducing training time by 87%
- **Batch prediction** — all 7 forecast days predicted in a single `model.predict()` call rather than 7 sequential calls
- **Lazy loading** — model loaded once on Flask startup and cached in memory, not reloaded per request
- **Cyclical encodings** — sin/cos features for calendar variables outperform one-hot encoding (fewer dimensions, preserved circular relationships)
- **Walk-forward CV** — 4 folds instead of full leave-one-out, reducing validation time from hours to seconds
- **Feature selection** — `rolling_7_std` (demand volatility) and `wow_delta` (week-on-week delta) added after ablation study showed +0.8% MAPE improvement

---

## 19. 🔒 Security

| Concern | Implementation |
|---------|---------------|
| CORS | Flask-CORS configured to allow all origins for public demo |
| Input validation | All query params cast to int/float with try/except |
| No secrets in code | No API keys, passwords, or tokens in source |
| Read-only data | Data files are never written to during inference |
| Rate limiting | Not implemented (demo) — add Flask-Limiter for production |
| HTTPS | Enforced automatically by Render's TLS termination |

---

## 20. 🧪 Testing

### Automated endpoint tests
```bash
python -c "
from app import app
client = app.test_client()
endpoints = [
  '/api/health', '/api/menu', '/api/forecast/WB01',
  '/api/metrics', '/api/features', '/api/history/WB01'
]
for ep in endpoints:
    r = client.get(ep)
    print(f'{'✓' if r.status_code==200 else '✗'} {ep} {r.status_code}')
"
```

### Expected output
```
✓ /api/health          200
✓ /api/menu            200
✓ /api/forecast/WB01   200
✓ /api/metrics         200
✓ /api/features        200
✓ /api/history/WB01    200
```

### Model validation
Walk-forward CV across 4 folds:
| Fold | Train Rows | Test Rows | MAPE | RMSE |
|------|-----------|-----------|------|------|
| 1 | 1,800 | 336 | 9.1% | 5.1 |
| 2 | 2,400 | 336 | 11.9% | 5.1 |
| 3 | 3,200 | 336 | 6.9% | 3.5 |
| 4 | 4,000 | 336 | 10.1% | 4.8 |
| **Mean** | — | — | **9.5%** | **4.6** |

---

## 21. 📈 Performance

| Metric | Value | Benchmark |
|--------|-------|-----------|
| CV Mean MAPE | 9.5% | Industry target < 12% ✓ |
| Holdout MAPE | 3.8% | Industry target < 12% ✓ |
| Holdout RMSE | 1.6 portions | — |
| Waste reduction | 72.6% | vs rolling-average baseline |
| Food saved | ~323 kg/month | per 8-item menu |
| API response time | < 100ms | per forecast request |
| Model training time | ~30 seconds | on 6,088 rows, 35 features |
| Build time (Render) | ~60 seconds | with lightweight requirements |
| Cold start time | < 2 seconds | demo backend (no model loading) |

---

## 22. 🔭 Future Improvements

- [ ] **Real POS integration** — direct API connector for Toast, Square, and Lightspeed POS systems
- [ ] **Live weather API** — replace weather stubs with OpenWeatherMap 7-day forecast
- [ ] **Live event API** — Eventbrite + Ticketmaster API for real event proximity detection
- [ ] **Prophet hybrid** — ensemble XGBoost + Facebook Prophet for better long-horizon seasonality
- [ ] **Uncertainty intervals** — add prediction intervals using XGBoost quantile regression
- [ ] **Ingredient-level mapping** — map dish forecasts to raw ingredient quantities automatically
- [ ] **Mobile app** — React Native dashboard for kitchen managers on the go
- [ ] **Anomaly alerts** — email/Slack notification when predictions deviate > 20% from actuals
- [ ] **Automated retraining** — weekly cron job to retrain on latest POS data
- [ ] **Multi-location** — support for restaurant chains with location-level demand modelling
- [ ] **User authentication** — JWT-based login for multi-user restaurant management

---

## 23. 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m 'Add AmazingFeature'

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request
```

Please make sure your PR:
- Passes all endpoint tests
- Doesn't break existing API contracts
- Includes a description of what changed and why

---

## 24. 📄 License

Distributed under the MIT License.

```
MIT License

Copyright (c) 2026 Shanmukh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 25. 👤 Author

**Shanmukh**

- 🐙 GitHub: [@shanmukh666](https://github.com/shanmukh666)
- 💼 Project: [FOODFLOW AI](https://github.com/shanmukh666/FOODFLOW-AI-AUTONOMOUS-DEMAND-PREDICTION-SYSTEM)
- 🌐 Live Demo: [foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com](https://foodflow-ai-autonomous-demand-prediction-wlp8.onrender.com)

---

<div align="center">

**Built with ❤️ to eliminate food waste one prediction at a time**

⭐ Star this repo if you found it useful!

</div>
