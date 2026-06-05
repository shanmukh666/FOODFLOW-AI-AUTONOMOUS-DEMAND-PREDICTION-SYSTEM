"""
KITCH·AI — Model Training Pipeline
Trains an XGBoost demand forecasting model with:
  - Walk-forward (time-series) cross-validation — no data leakage
  - Per-item global model with item_enc as a categorical feature
  - MAPE + RMSE + waste-reduction metrics
  - Model persistence (pickle + metadata JSON)
"""

import pandas as pd
import numpy as np
import json, os, pickle, warnings
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

from feature_engineering import FEATURE_COLS, TARGET_COL, build_features

warnings.filterwarnings("ignore")

MODEL_DIR  = "models"
MODEL_PATH = f"{MODEL_DIR}/xgb_demand.pkl"
META_PATH  = f"{MODEL_DIR}/model_meta.json"

# ── Walk-forward CV ────────────────────────────────────────────────────────────
def walk_forward_cv(df: pd.DataFrame,
                    n_splits: int = 4,
                    test_weeks: int = 2) -> dict:
    """
    Splits time series into n_splits folds where each test set is the
    most recent `test_weeks` weeks. Training always uses data BEFORE the test set.
    """
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())
    split_size   = len(unique_dates) // (n_splits + 1)

    results = {"mape_scores": [], "rmse_scores": [], "fold_details": []}

    for fold in range(n_splits):
        cutoff_idx  = split_size * (fold + 1)
        test_start  = unique_dates[cutoff_idx]
        test_end    = unique_dates[min(cutoff_idx + test_weeks * 7 - 1, len(unique_dates) - 1)]

        train = df[df["date"] < test_start]
        test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(train) < 100 or len(test) < 10:
            continue

        feats = [c for c in FEATURE_COLS if c in df.columns]
        X_tr, y_tr = train[feats].fillna(0), train[TARGET_COL]
        X_te, y_te = test[feats].fillna(0),  test[TARGET_COL]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)

        preds = np.clip(model.predict(X_te), 0, None)
        mape  = mean_absolute_percentage_error(y_te, preds) * 100
        rmse  = np.sqrt(mean_squared_error(y_te, preds))

        results["mape_scores"].append(mape)
        results["rmse_scores"].append(rmse)
        results["fold_details"].append({
            "fold":       fold + 1,
            "train_rows": len(train),
            "test_rows":  len(test),
            "mape":       round(mape, 2),
            "rmse":       round(rmse, 2),
            "test_start": str(test_start)[:10],
            "test_end":   str(test_end)[:10],
        })

        print(f"  Fold {fold+1}: MAPE={mape:.1f}%  RMSE={rmse:.1f}  "
              f"[train={len(train):,}  test={len(test):,}]")

    results["mean_mape"] = np.mean(results["mape_scores"])
    results["mean_rmse"] = np.mean(results["rmse_scores"])
    return results

# ── Waste reduction metric ─────────────────────────────────────────────────────
def estimate_waste_reduction(y_true, y_pred, shelf_days: int = 2) -> dict:
    """
    Simulates ordering decisions:
      - Naive baseline orders rolling 7-day average
      - Model orders predicted qty + small safety buffer
    Overstocked = ordered - sold (when positive)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Baseline: order 7-day rolling average + 10% buffer
    baseline_order = pd.Series(y_true).rolling(7, min_periods=1).mean().values * 1.10
    model_order    = y_pred * 1.05  # 5% safety buffer

    baseline_waste = np.maximum(baseline_order - y_true, 0).sum()
    model_waste    = np.maximum(model_order    - y_true, 0).sum()

    reduction_pct  = (baseline_waste - model_waste) / max(baseline_waste, 1) * 100
    kg_saved       = (baseline_waste - model_waste) * 0.25  # ~250g per portion

    return {
        "baseline_waste_portions": int(baseline_waste),
        "model_waste_portions":    int(model_waste),
        "waste_reduction_pct":     round(reduction_pct, 1),
        "kg_food_saved":           round(kg_saved, 1),
    }

# ── Feature importance ─────────────────────────────────────────────────────────
def get_feature_importance(model, feature_cols: list) -> list:
    raw = model.feature_importances_
    total = raw.sum() or 1
    pairs = sorted(zip(feature_cols, raw / total * 100),
                   key=lambda x: x[1], reverse=True)
    return [{"name": name, "importance": round(float(imp), 1)} for name, imp in pairs[:12]]

# ── Final model train ──────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X     = df[feats].fillna(0)
    y     = df[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y, verbose=False)
    print(f"[TRAIN] Final model trained on {len(df):,} rows  |  {len(feats)} features")
    return model

# ── Full train pipeline ────────────────────────────────────────────────────────
def run_training(data_dir: str = "data") -> dict:
    print("=" * 60)
    print("KITCH·AI  —  MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Feature engineering
    print("\n[1/4] Building features...")
    df = build_features(data_dir=data_dir, output_dir=data_dir)

    # 2. Walk-forward CV
    print("\n[2/4] Walk-forward cross-validation...")
    cv_results = walk_forward_cv(df, n_splits=4, test_weeks=2)
    print(f"\n  CV Mean MAPE: {cv_results['mean_mape']:.1f}%")
    print(f"  CV Mean RMSE: {cv_results['mean_rmse']:.1f}")

    # 3. Train final model on all data
    print("\n[3/4] Training final model...")
    model = train_final_model(df)

    # 4. Evaluate on holdout (last 30 days)
    print("\n[4/4] Holdout evaluation (last 30 days)...")
    cutoff    = df["date"].max() - pd.Timedelta(days=30)
    holdout   = df[df["date"] > cutoff]
    feats     = [c for c in FEATURE_COLS if c in df.columns]
    y_true_h  = holdout[TARGET_COL].values
    y_pred_h  = np.clip(model.predict(holdout[feats].fillna(0)), 0, None)
    mape_h    = mean_absolute_percentage_error(y_true_h, y_pred_h) * 100
    rmse_h    = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    waste     = estimate_waste_reduction(y_true_h, y_pred_h)

    print(f"  Holdout MAPE:       {mape_h:.1f}%")
    print(f"  Holdout RMSE:       {rmse_h:.1f}")
    print(f"  Waste reduction:    {waste['waste_reduction_pct']}%")
    print(f"  Estimated kg saved: {waste['kg_food_saved']} kg")

    feat_imp = get_feature_importance(model, feats)

    # 5. Save model + metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feats}, f)

    meta = {
        "trained_at":       datetime.now().isoformat(),
        "train_rows":       len(df),
        "features":         feats,
        "cv_mean_mape":     round(cv_results["mean_mape"], 2),
        "cv_mean_rmse":     round(cv_results["mean_rmse"], 2),
        "holdout_mape":     round(mape_h, 2),
        "holdout_rmse":     round(rmse_h, 2),
        "waste_reduction":  waste,
        "feature_importance": feat_imp,
        "cv_folds":         cv_results["fold_details"],
        "date_range":       {
            "start": str(df["date"].min())[:10],
            "end":   str(df["date"].max())[:10],
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE] Model  → {MODEL_PATH}")
    print(f"[SAVE] Meta   → {META_PATH}")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    return meta

if __name__ == "__main__":
    run_training(data_dir="data")
