"""
KITCH·AI — Feature Engineering Pipeline
Transforms merged POS + weather + event data into ML-ready features:
  - Lag features (1d, 7d, 14d)
  - Rolling averages (7d, 14d, 30d)
  - Calendar encodings
  - Weather dummies
  - Event proximity flags
  - Cross-item demand correlations (bun ↔ burger)
"""

import pandas as pd
import numpy as np
import json, os

# ── Load & merge raw data ──────────────────────────────────────────────────────
def load_and_merge(data_dir: str = "data") -> pd.DataFrame:
    pos     = pd.read_csv(f"{data_dir}/pos_sales.csv",  parse_dates=["date"])
    weather = pd.read_csv(f"{data_dir}/weather.csv",    parse_dates=["date"])

    with open(f"{data_dir}/events.json") as f:
        events_raw = json.load(f)
    with open(f"{data_dir}/promos.json") as f:
        promos_raw = json.load(f)

    # Event + promo flags per date
    event_df = pd.DataFrame([
        {"date": pd.to_datetime(k), "local_event_nearby": 1,
         "event_lift": v.get("lift", 0.0)}
        for k, v in events_raw.items()
    ])
    promo_df = pd.DataFrame([
        {"date": pd.to_datetime(k), "is_promo_day": 1}
        for k in promos_raw
    ])

    df = pos.merge(weather,  on="date", how="left")
    if len(event_df):
        df = df.merge(event_df, on="date", how="left")
    if len(promo_df):
        df = df.merge(promo_df, on="date", how="left")

    df["local_event_nearby"] = df.get("local_event_nearby", pd.Series(0)).fillna(0).astype(int)
    df["event_lift"]         = df.get("event_lift",         pd.Series(0)).fillna(0)
    df["is_promo_day"]       = df.get("is_promo_day",       pd.Series(0)).fillna(0).astype(int)

    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    print(f"[MERGE] {len(df):,} rows  |  {df['item_id'].nunique()} items  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df

# ── Calendar features ──────────────────────────────────────────────────────────
HOLIDAYS = {
    (1,  1), (2, 14), (7, 4), (11, 28), (12, 25), (12, 31)
}

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"]     = df["date"].dt.dayofweek          # 0=Mon
    df["week_of_year"]    = df["date"].dt.isocalendar().week.astype(int)
    df["month"]           = df["date"].dt.month
    df["quarter"]         = df["date"].dt.quarter
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
    df["is_friday"]       = (df["day_of_week"] == 4).astype(int)
    df["is_holiday"]      = df["date"].apply(
        lambda d: int((d.month, d.day) in HOLIDAYS))

    # Cyclical encodings (avoids ordinal distance artifacts)
    df["dow_sin"]         = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]         = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)

    # Days until next weekend (Friday = 0, Saturday = 0)
    df["days_to_weekend"] = df["day_of_week"].apply(
        lambda d: max(0, 4 - d) if d <= 4 else 0)

    return df

# ── Lag & rolling features (per item) ─────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["item_id", "date"])

    grp = df.groupby("item_id")["qty_sold"]

    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling averages (shift 1 to avoid leakage)
    for window in [7, 14, 30]:
        df[f"rolling_{window}"] = (
            grp.shift(1)
               .rolling(window, min_periods=max(1, window // 2))
               .mean()
               .values
        )

    # Rolling std (demand volatility)
    df["rolling_7_std"] = (
        grp.shift(1).rolling(7, min_periods=3).std().values
    )

    # Same day last week delta
    df["wow_delta"] = df["lag_7"] - df.groupby("item_id")["qty_sold"].shift(14)

    return df

# ── Weather features ───────────────────────────────────────────────────────────
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rain_dummy"]    = (df["precipitation_mm"] > 5).astype(int)
    df["heavy_rain"]    = (df["precipitation_mm"] > 15).astype(int)
    df["cold_day"]      = (df["temp_c"] < 5).astype(int)
    df["hot_day"]       = (df["temp_c"] > 28).astype(int)
    df["pleasant_day"]  = ((df["temp_c"].between(16, 24)) &
                           (df["precipitation_mm"] < 3)).astype(int)
    return df

# ── Cross-item correlations ────────────────────────────────────────────────────
ITEM_LINKS = {
    "CB05": "WB01",   # Brioche bun demand tracks wagyu burger
    "CB05": "SM07",   # also smash burger
}

def add_cross_item_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["item_id", "date"])

    # For each linked pair, add lag_7 of the driver item as a feature
    for dependent, driver in ITEM_LINKS.items():
        driver_lag = (
            df[df["item_id"] == driver][["date", "qty_sold"]]
            .rename(columns={"qty_sold": f"driver_lag7_{driver}"})
            .assign(**{f"driver_lag7_{driver}":
                lambda x: x["qty_sold"].shift(7) if "qty_sold" in x.columns
                else x[f"driver_lag7_{driver}"]})
        )
        # Simpler: just pull lag_7 values for driver and merge
        driver_df = df[df["item_id"] == driver][["date", "lag_7"]].copy()
        driver_df = driver_df.rename(columns={"lag_7": f"lag7_{driver}"})
        df = df.merge(driver_df, on="date", how="left")

    return df

# ── Item encoding ──────────────────────────────────────────────────────────────
CATEGORY_MAP = {"main": 0, "starter": 1, "side": 2, "comp": 3, "dessert": 4}

def add_item_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    items = sorted(df["item_id"].unique())
    item_map = {item: i for i, item in enumerate(items)}
    df["item_enc"]     = df["item_id"].map(item_map)
    df["category_enc"] = df["category"].map(CATEGORY_MAP).fillna(0).astype(int)
    return df

# ── Full pipeline ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Item identity
    "item_enc", "category_enc",
    # Lags
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    # Rolling stats
    "rolling_7", "rolling_14", "rolling_30", "rolling_7_std", "wow_delta",
    # Calendar
    "day_of_week", "is_weekend", "is_friday", "is_holiday",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "week_of_year", "month", "quarter", "days_to_weekend",
    # Weather
    "temp_c", "precipitation_mm", "feels_like_c",
    "rain_dummy", "heavy_rain", "cold_day", "hot_day", "pleasant_day",
    # External signals
    "local_event_nearby", "event_lift", "is_promo_day",
]

TARGET_COL = "qty_sold"

def build_features(data_dir: str = "data",
                   output_dir: str = "data") -> pd.DataFrame:
    df = load_and_merge(data_dir)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_weather_features(df)
    df = add_cross_item_features(df)
    df = add_item_encoding(df)

    # Drop rows with NaN in critical lag columns (first ~14 days per item)
    df = df.dropna(subset=["lag_7", "lag_14", "rolling_14"])

    # Clip negatives from lag arithmetic
    for col in [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]:
        df[col] = df[col].clip(lower=0)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/features.csv", index=False)

    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"[FEATURES] ✓ {len(df):,} training rows")
    print(f"[FEATURES] ✓ {len(available)} features: {available}")
    print(f"[FEATURES] ✓ Target range: {df[TARGET_COL].min()} – {df[TARGET_COL].max()}")
    return df

if __name__ == "__main__":
    build_features(data_dir="data", output_dir="data")
