#!/usr/bin/env python3
"""
Numerai Tournament Starter Script
==================================
A complete pipeline: download data -> train model -> predict -> submit.

Numerai is a data science competition where you build ML models to predict
the stock market using obfuscated financial data. Successful predictions
earn NMR (Numeraire) cryptocurrency rewards.

Installation:
    pip install numerapi pandas pyarrow lightgbm scikit-learn cloudpickle

API Key Setup:
    1. Sign up at https://numer.ai
    2. Go to Account -> Custom API Keys
    3. Create a key with "Upload submissions" permission
    4. Set environment variables or pass directly:
       export NUMERAI_PUBLIC_ID="your_public_id"
       export NUMERAI_SECRET_KEY="your_secret_key"

Usage:
    # Full pipeline: download, train, predict, submit
    python numerai_starter.py --mode full

    # Download data only
    python numerai_starter.py --mode download

    # Train model only (requires downloaded data)
    python numerai_starter.py --mode train

    # Predict and submit only (requires trained model)
    python numerai_starter.py --mode submit

    # Diagnostics: validate on historical data
    python numerai_starter.py --mode diagnostics

Author: Asset Shield Project
Date: 2026-02-10
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dataset version - Numerai updates this periodically; check docs.numer.ai
# v5.2 "Faith II" (Jan 2026): added target_ender_20 (new payout target), target_jasper_20/60
DATA_VERSION = "v5.2"

# Directory to store downloaded data and trained models
DATA_DIR = Path(__file__).parent.parent / "data" / "numerai"
MODEL_DIR = DATA_DIR / "models"

# Feature set: "rain" (666) outperforms "medium" (705) on all metrics per Numerai research
# Options: small(42), medium(705), rain(666), sunshine(325), fncv3(400), all(2376)
FEATURE_SET = "rain"

# Target to predict
# 2026 payout: 0.75*CORR + 2.25*MMC, scored on target_ender_20
TARGET_COL = "target"

# LightGBM hyperparameters
# Standard configuration from Numerai docs
LGBM_PARAMS_STANDARD = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 2**5,
    "colsample_bytree": 0.1,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# Deep configuration - better performance but slower training (~2hrs)
LGBM_PARAMS_DEEP = {
    "n_estimators": 20000,
    "learning_rate": 0.001,
    "max_depth": 6,
    "num_leaves": 2**6,
    "colsample_bytree": 0.1,
    "subsample": 0.8,
    "min_child_samples": 10000,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# Set to True for deep model (slower but better)
USE_DEEP_MODEL = False

# Ensemble: train one model per target, average predictions
# target_ender_20 = payout target since Jan 2026 (most important)
ENSEMBLE_TARGETS = [
    "target",
    "target_ender_20",
    "target_jasper_20",
    "target_cyrusd_20",
    "target_ralph_20",
    "target_victor_20",
    "target_waldo_20",
]

# Neutralization proportion (0 = no neutralization, 1 = full)
# 0.5-0.7 recommended; 0.6 balances consistency & performance
NEUTRALIZE_PROPORTION = 0.6

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("numerai_starter")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def ensure_dirs():
    """Create data and model directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Data directory: {DATA_DIR}")
    log.info(f"Model directory: {MODEL_DIR}")


def get_napi(public_id: Optional[str] = None, secret_key: Optional[str] = None):
    """
    Initialize NumerAPI client.

    API keys can be provided directly, via environment variables
    (NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY), or omitted for read-only
    operations like downloading data.
    """
    from numerapi import NumerAPI

    pub = public_id or os.environ.get("NUMERAI_PUBLIC_ID", "")
    sec = secret_key or os.environ.get("NUMERAI_SECRET_KEY", "")

    if pub and sec:
        napi = NumerAPI(pub, sec)
        log.info("Authenticated NumerAPI client created.")
    else:
        napi = NumerAPI()
        log.info("Anonymous NumerAPI client created (download only).")

    return napi


def get_feature_columns(df: pd.DataFrame, feature_metadata: Optional[dict] = None) -> list:
    """
    Get feature columns based on the selected FEATURE_SET.

    If feature_metadata is provided (from features.json), uses the
    predefined feature groups. Otherwise, filters columns by name.
    """
    if feature_metadata and FEATURE_SET in feature_metadata.get("feature_sets", {}):
        features = feature_metadata["feature_sets"][FEATURE_SET]
        # Only keep features that exist in the dataframe
        features = [f for f in features if f in df.columns]
        log.info(f"Using '{FEATURE_SET}' feature set: {len(features)} features")
        return features

    # Fallback: all columns containing "feature"
    features = [c for c in df.columns if "feature" in c]
    log.info(f"Using all feature columns: {len(features)} features")
    return features


def neutralize(
    df: pd.DataFrame,
    columns: list,
    neutralizers: list,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """
    Neutralize predictions to reduce exposure to certain features.

    This is a common technique in Numerai to improve MMC score.
    It removes the component of your predictions that is linearly
    explained by the neutralizer features.
    """
    scores = df[columns].values.astype(np.float64)
    exposures = df[neutralizers].values.astype(np.float64)

    # Drop rows with NaN/Inf
    valid_rows = np.isfinite(exposures).all(axis=1) & np.isfinite(scores).all(axis=1)
    if valid_rows.sum() < 10:
        log.warning("neutralize: too few valid rows, returning unneutralized scores")
        return df[columns].copy()

    scores_clean = scores[valid_rows]
    exposures_clean = exposures[valid_rows]

    # Drop zero-variance columns (cause divide-by-zero in lstsq/matmul)
    col_std = exposures_clean.std(axis=0)
    good_cols = col_std > 1e-8
    exposures_clean = exposures_clean[:, good_cols]

    if exposures_clean.shape[1] == 0:
        log.warning("neutralize: all exposures zero-variance, returning unneutralized scores")
        return df[columns].copy()

    # Add constant for intercept
    exposures_clean = np.hstack([exposures_clean, np.ones((exposures_clean.shape[0], 1))])

    # Ridge regression with Apple Silicon BLAS workaround
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        XtX = exposures_clean.T @ exposures_clean
        XtX[np.diag_indices_from(XtX)] += 1e-6
        coefs = np.linalg.solve(XtX, exposures_clean.T @ scores_clean)
        correction = exposures_clean @ coefs
    # Sanitize any residual NaN/Inf from numerical edge cases
    np.nan_to_num(correction, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove the explained component
    neutralized = np.copy(scores)
    neutralized[valid_rows] = scores_clean - proportion * correction
    return pd.DataFrame(neutralized, index=df.index, columns=columns)


def rank_gauss_normalize(series: pd.Series) -> pd.Series:
    """
    Apply rank-based Gaussian normalization per era.

    Converts predictions to ranks, then maps to a Gaussian distribution.
    This is a standard Numerai preprocessing technique.
    """
    from scipy.stats import norm

    ranked = series.rank(method="average", pct=True)
    # Clip to avoid infinities at 0 and 1
    ranked = ranked.clip(0.0001, 0.9999)
    return pd.Series(norm.ppf(ranked), index=series.index)


# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------


def download_data(napi) -> Dict[str, Path]:
    """
    Download the latest Numerai tournament data.

    Downloads:
    - Training data (historical features + targets)
    - Validation data (for local evaluation)
    - Live data (current round features for submission)
    - Feature metadata (feature groups and descriptions)

    Returns dict of {name: filepath} for downloaded files.
    """
    ensure_dirs()

    files = {
        "features_json": f"{DATA_VERSION}/features.json",
        "train": f"{DATA_VERSION}/train.parquet",
        "validation": f"{DATA_VERSION}/validation.parquet",
    }

    current_round = napi.get_current_round()
    log.info(f"Current tournament round: {current_round}")
    files["live"] = f"{DATA_VERSION}/live.parquet"

    downloaded = {}
    for name, remote_path in files.items():
        local_path = DATA_DIR / remote_path.split("/")[-1]

        # Skip re-download if file exists and is recent (< 12 hours)
        if local_path.exists():
            age_hours = (time.time() - local_path.stat().st_mtime) / 3600
            if age_hours < 12 and name != "live":
                log.info(f"  {name}: Using cached file ({age_hours:.1f}h old)")
                downloaded[name] = local_path
                continue

        log.info(f"  Downloading {name}: {remote_path} ...")
        napi.download_dataset(remote_path, str(local_path))
        downloaded[name] = local_path
        log.info(f"  {name}: Saved to {local_path}")

    return downloaded


def load_data(file_paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Load downloaded data files into memory.

    Returns a dict with DataFrames and feature metadata.
    """
    log.info("Loading data into memory...")

    # Load feature metadata
    feature_metadata = None
    if "features_json" in file_paths:
        with open(file_paths["features_json"]) as f:
            feature_metadata = json.load(f)
        log.info(f"  Feature metadata loaded. Feature sets: {list(feature_metadata.get('feature_sets', {}).keys())}")

    # Load training data
    log.info("  Loading training data (this may take 1-5 minutes)...")
    train_df = pd.read_parquet(file_paths["train"])
    log.info(f"  Training data: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")

    # Load validation data
    log.info("  Loading validation data...")
    val_df = pd.read_parquet(file_paths["validation"])
    log.info(f"  Validation data: {val_df.shape[0]:,} rows x {val_df.shape[1]} cols")

    # Load live data
    live_df = None
    if "live" in file_paths and file_paths["live"].exists():
        log.info("  Loading live data...")
        live_df = pd.read_parquet(file_paths["live"])
        log.info(f"  Live data: {live_df.shape[0]:,} rows x {live_df.shape[1]} cols")

    # Determine feature columns
    features = get_feature_columns(train_df, feature_metadata)

    return {
        "train": train_df,
        "validation": val_df,
        "live": live_df,
        "features": features,
        "feature_metadata": feature_metadata,
    }


def train_model(data: Dict[str, Any], save: bool = True) -> Any:
    """
    Train a LightGBM model on Numerai training data.

    Uses era-wise training with early stopping on a held-out
    validation set to prevent overfitting.

    Args:
        data: Dict from load_data() with DataFrames and feature list.
        save: Whether to save the trained model to disk.

    Returns:
        Trained LightGBM model.
    """
    import lightgbm as lgb

    train_df = data["train"]
    features = data["features"]

    log.info(f"Training LightGBM model on {len(features)} features...")
    log.info(f"  Target: {TARGET_COL}")

    # Drop rows with NaN target
    train_mask = train_df[TARGET_COL].notna()
    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, TARGET_COL]
    log.info(f"  Training samples: {len(X_train):,}")

    # Use recent eras as validation for early stopping
    # Eras are strings like "0001", "0002", etc.
    eras = train_df.loc[train_mask, "era"].unique()
    eras_sorted = sorted(eras)
    n_val_eras = max(1, len(eras_sorted) // 10)  # 10% of eras for validation
    val_eras = set(eras_sorted[-n_val_eras:])
    train_eras = set(eras_sorted[:-n_val_eras])

    train_idx = train_df.loc[train_mask, "era"].isin(train_eras)
    val_idx = train_df.loc[train_mask, "era"].isin(val_eras)

    log.info(f"  Train eras: {len(train_eras)}, Validation eras: {len(val_eras)}")

    # Select hyperparameters
    params = LGBM_PARAMS_DEEP if USE_DEEP_MODEL else LGBM_PARAMS_STANDARD
    model_type = "deep" if USE_DEEP_MODEL else "standard"
    log.info(f"  Using {model_type} hyperparameters: n_estimators={params['n_estimators']}")

    # Handle NaN in features - LightGBM handles NaN natively
    model = lgb.LGBMRegressor(**params)

    # Fit with early stopping using validation eras
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=True),
        lgb.log_evaluation(period=500),
    ]

    model.fit(
        X_train[train_idx],
        y_train[train_idx],
        eval_set=[(X_train[val_idx], y_train[val_idx])],
        callbacks=callbacks,
    )

    log.info(f"  Best iteration: {model.best_iteration_}")

    # Save model
    if save:
        import cloudpickle
        model_path = MODEL_DIR / f"lgbm_{model_type}_{DATA_VERSION}.pkl"
        with open(model_path, "wb") as f:
            cloudpickle.dump({"model": model, "features": features}, f)
        log.info(f"  Model saved to {model_path}")

    return model


def train_ensemble(data: Dict[str, Any], targets: list = None, save: bool = True) -> Dict[str, Any]:
    """
    Train one LightGBM model per target for ensemble prediction.
    """
    import lightgbm as lgb

    if targets is None:
        targets = ENSEMBLE_TARGETS

    train_df = data["train"]
    features = data["features"]

    # Era-wise train/val split (same as train_model)
    eras = sorted(train_df["era"].unique())
    n_val = max(1, len(eras) // 10)
    val_eras = set(eras[-n_val:])
    train_mask_eras = train_df["era"].isin(set(eras[:-n_val]))
    val_mask_eras = train_df["era"].isin(val_eras)

    params = LGBM_PARAMS_DEEP if USE_DEEP_MODEL else LGBM_PARAMS_STANDARD
    model_type = "deep" if USE_DEEP_MODEL else "standard"

    models = {}
    for tgt in targets:
        if tgt not in train_df.columns:
            log.warning(f"  Target '{tgt}' not found, skipping.")
            continue

        log.info(f"  Training on target: {tgt}")
        mask = train_df[tgt].notna()
        X = train_df.loc[mask, features]
        y = train_df.loc[mask, tgt]

        t_idx = mask & train_mask_eras
        v_idx = mask & val_mask_eras

        model = lgb.LGBMRegressor(**params)
        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=500),
        ]
        model.fit(
            X[t_idx], y[t_idx],
            eval_set=[(X[v_idx], y[v_idx])],
            callbacks=callbacks,
        )
        log.info(f"    Best iteration: {model.best_iteration_}")
        models[tgt] = model

    log.info(f"  Ensemble trained: {len(models)} models")

    if save:
        import cloudpickle
        model_path = MODEL_DIR / f"lgbm_ensemble_{model_type}_{DATA_VERSION}.pkl"
        with open(model_path, "wb") as f:
            cloudpickle.dump({"models": models, "features": features, "targets": list(models.keys())}, f)
        log.info(f"  Ensemble saved to {model_path}")

    return models


def load_model() -> tuple:
    """
    Load a previously trained model from disk.

    Returns (model, features) tuple.
    """
    import cloudpickle

    model_type = "deep" if USE_DEEP_MODEL else "standard"
    model_path = MODEL_DIR / f"lgbm_{model_type}_{DATA_VERSION}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run with --mode train first."
        )

    with open(model_path, "rb") as f:
        bundle = cloudpickle.load(f)

    log.info(f"Model loaded from {model_path}")
    return bundle["model"], bundle["features"]


def load_ensemble() -> tuple:
    """Load ensemble models from disk."""
    import cloudpickle
    model_type = "deep" if USE_DEEP_MODEL else "standard"
    model_path = MODEL_DIR / f"lgbm_ensemble_{model_type}_{DATA_VERSION}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No ensemble model at {model_path}. Run with --mode train --ensemble first.")

    with open(model_path, "rb") as f:
        bundle = cloudpickle.load(f)

    log.info(f"Ensemble loaded: {len(bundle['models'])} models from {model_path}")
    return bundle["models"], bundle["features"], bundle["targets"]


def predict_and_submit(
    napi,
    model,
    features: list,
    live_df: pd.DataFrame,
    model_name: Optional[str] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions on live data and submit to Numerai.

    Args:
        napi: Authenticated NumerAPI client.
        model: Trained model with .predict() method.
        features: List of feature column names.
        live_df: Live round DataFrame.
        model_name: Your model name on Numerai (for model_id lookup).
        dry_run: If True, generate predictions but don't submit.

    Returns:
        DataFrame with predictions.
    """
    current_round = napi.get_current_round()
    log.info(f"Generating predictions for round {current_round}...")

    # Predict
    live_features = live_df[features]
    raw_predictions = model.predict(live_features)

    # Rank predictions between 0 and 1 (required format)
    # Numerai expects values in (0, 1) representing relative rankings
    ranked = pd.Series(raw_predictions, index=live_df.index).rank(pct=True)
    # Clip to avoid exact 0 or 1
    ranked = ranked.clip(0.0001, 0.9999)

    # Format submission
    submission = ranked.to_frame("prediction")
    log.info(f"  Predictions: min={submission['prediction'].min():.4f}, "
             f"max={submission['prediction'].max():.4f}, "
             f"mean={submission['prediction'].mean():.4f}")
    log.info(f"  Unique stocks: {len(submission):,}")

    # Save to CSV
    submission_path = DATA_DIR / f"prediction_{current_round}.csv"
    submission.to_csv(submission_path)
    log.info(f"  Predictions saved to {submission_path}")

    if dry_run:
        log.info("  DRY RUN: Skipping submission upload.")
        return submission

    # Submit to Numerai
    try:
        # Get model ID
        models = napi.get_models()
        if not models:
            log.error("  No models found. Create a model at https://numer.ai first.")
            return submission

        if model_name and model_name in models:
            model_id = models[model_name]
        else:
            # Use the first model if no name specified
            model_name = list(models.keys())[0]
            model_id = models[model_name]
            log.info(f"  Using model: {model_name} (id: {model_id})")

        napi.upload_predictions(str(submission_path), model_id=model_id)
        log.info(f"  Successfully submitted predictions for round {current_round}!")

    except Exception as e:
        log.error(f"  Submission failed: {e}")
        log.error("  Ensure your API keys have 'Upload submissions' permission.")
        log.error("  You can manually upload the CSV at https://numer.ai")

    return submission


def predict_ensemble(
    napi,
    models: dict,
    features: list,
    live_df: pd.DataFrame,
    model_name: Optional[str] = None,
    do_neutralize: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Generate ensemble predictions and submit."""
    current_round = napi.get_current_round()
    log.info(f"Generating ensemble predictions for round {current_round} ({len(models)} models)...")

    all_preds = []
    for tgt, model in models.items():
        raw = model.predict(live_df[features])
        ranked = pd.Series(raw, index=live_df.index).rank(pct=True).clip(0.0001, 0.9999)
        all_preds.append(ranked)
        log.info(f"  {tgt}: mean={ranked.mean():.4f}")

    # Average ranked predictions
    avg_pred = pd.concat(all_preds, axis=1).mean(axis=1)
    # Re-rank the average to ensure uniform distribution
    final_pred = avg_pred.rank(pct=True).clip(0.0001, 0.9999)

    # Optional neutralization for MMC
    if do_neutralize:
        log.info(f"  Applying neutralization (proportion={NEUTRALIZE_PROPORTION})...")
        # Neutralize against top features to improve MMC
        neut_features = [f for f in features if f in live_df.columns][:50]
        pred_df = live_df[neut_features].copy()
        pred_df["prediction"] = final_pred
        neutralized = neutralize(pred_df, ["prediction"], neut_features, proportion=NEUTRALIZE_PROPORTION)
        final_pred = neutralized["prediction"].rank(pct=True).clip(0.0001, 0.9999)
        log.info(f"  Post-neutralization mean={final_pred.mean():.4f}")

    submission = final_pred.to_frame("prediction")
    log.info(f"  Ensemble predictions: min={submission['prediction'].min():.4f}, "
             f"max={submission['prediction'].max():.4f}, "
             f"mean={submission['prediction'].mean():.4f}")

    submission_path = DATA_DIR / f"prediction_{current_round}.csv"
    submission.to_csv(submission_path)
    log.info(f"  Saved to {submission_path}")

    if dry_run:
        log.info("  DRY RUN: Skipping upload.")
        return submission

    try:
        models_map = napi.get_models()
        if model_name and model_name in models_map:
            model_id = models_map[model_name]
        else:
            model_name = list(models_map.keys())[0]
            model_id = models_map[model_name]

        napi.upload_predictions(str(submission_path), model_id=model_id)
        log.info(f"  Submitted ensemble predictions for round {current_round}!")
    except Exception as e:
        log.error(f"  Submission failed: {e}")

    return submission


def run_diagnostics(model, data: Dict[str, Any]):
    """
    Run diagnostics on validation data to assess model quality.

    Computes per-era correlation (the primary scoring metric),
    mean correlation, Sharpe ratio, and max drawdown of correlations.

    These metrics help predict how your model will perform live.
    """
    from scipy.stats import spearmanr

    val_df = data["validation"]
    features = data["features"]

    log.info("Running diagnostics on validation data...")

    # Predict on validation
    val_predictions = model.predict(val_df[features])
    val_df = val_df.copy()
    val_df["prediction"] = pd.Series(val_predictions, index=val_df.index).rank(pct=True)

    # Per-era Spearman correlation (Numerai's CORR metric)
    eras = val_df["era"].unique()
    era_corrs = []
    for era in sorted(eras):
        era_mask = val_df["era"] == era
        era_data = val_df[era_mask]
        if era_data[TARGET_COL].notna().sum() < 10:
            continue
        corr, _ = spearmanr(era_data["prediction"], era_data[TARGET_COL])
        era_corrs.append({"era": era, "corr": corr})

    corr_df = pd.DataFrame(era_corrs)

    # Summary statistics
    mean_corr = corr_df["corr"].mean()
    std_corr = corr_df["corr"].std()
    sharpe = mean_corr / std_corr if std_corr > 0 else 0
    max_dd = (corr_df["corr"].cumsum() - corr_df["corr"].cumsum().cummax()).min()
    pct_positive = (corr_df["corr"] > 0).mean() * 100

    log.info("=" * 60)
    log.info("DIAGNOSTICS RESULTS")
    log.info("=" * 60)
    log.info(f"  Eras evaluated:    {len(corr_df)}")
    log.info(f"  Mean CORR:         {mean_corr:.6f}")
    log.info(f"  Std CORR:          {std_corr:.6f}")
    log.info(f"  Sharpe (CORR):     {sharpe:.4f}")
    log.info(f"  Max Drawdown:      {max_dd:.6f}")
    log.info(f"  % Positive Eras:   {pct_positive:.1f}%")
    log.info("=" * 60)

    # Interpretation guide
    log.info("\nInterpretation Guide:")
    if mean_corr > 0.03:
        log.info("  CORR > 0.03 : Excellent - strong predictive signal")
    elif mean_corr > 0.02:
        log.info("  CORR > 0.02 : Good - competitive model")
    elif mean_corr > 0.01:
        log.info("  CORR > 0.01 : Decent - room for improvement")
    else:
        log.info("  CORR < 0.01 : Weak - consider feature engineering or ensembling")

    if sharpe > 1.0:
        log.info("  Sharpe > 1.0 : Excellent consistency")
    elif sharpe > 0.5:
        log.info("  Sharpe > 0.5 : Reasonable consistency")
    else:
        log.info("  Sharpe < 0.5 : High variance - model may be unstable")

    return corr_df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def _apply_deep_mode():
    """Enable deep model parameters."""
    global USE_DEEP_MODEL
    USE_DEEP_MODEL = True


def _apply_feature_set(fs: str):
    """Override the feature set."""
    global FEATURE_SET
    FEATURE_SET = fs


def main():
    parser = argparse.ArgumentParser(
        description="Numerai Tournament Starter Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python numerai_starter.py --mode full          # Complete pipeline
  python numerai_starter.py --mode download       # Download data only
  python numerai_starter.py --mode train          # Train model
  python numerai_starter.py --mode submit         # Submit predictions
  python numerai_starter.py --mode diagnostics    # Run diagnostics
  python numerai_starter.py --mode full --dry-run # Full pipeline without submitting
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "download", "train", "submit", "diagnostics"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Numerai model name for submission (default: first model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate predictions but don't submit to Numerai",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deep model parameters (slower, better performance)",
    )
    parser.add_argument(
        "--feature-set",
        choices=["small", "medium", "rain", "sunshine", "fncv3", "all"],
        default=None,
        help="Feature set to use (default: rain)",
    )
    parser.add_argument(
        "--public-id",
        type=str,
        default=None,
        help="Numerai API public ID (or set NUMERAI_PUBLIC_ID env var)",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default=None,
        help="Numerai API secret key (or set NUMERAI_SECRET_KEY env var)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use multi-target ensemble",
    )
    parser.add_argument(
        "--neutralize",
        action="store_true",
        help="Apply feature neutralization (for MMC)",
    )

    args = parser.parse_args()

    # Apply CLI overrides to module-level config
    if args.deep:
        _apply_deep_mode()
    if args.feature_set:
        _apply_feature_set(args.feature_set)

    log.info("=" * 60)
    log.info("NUMERAI TOURNAMENT STARTER")
    log.info(f"  Mode:        {args.mode}")
    log.info(f"  Dataset:     {DATA_VERSION}")
    log.info(f"  Feature set: {FEATURE_SET}")
    log.info(f"  Model type:  {'deep' if USE_DEEP_MODEL else 'standard'}")
    log.info(f"  Ensemble:    {args.ensemble}")
    log.info(f"  Neutralize:  {args.neutralize}")
    log.info(f"  Dry run:     {args.dry_run}")
    log.info("=" * 60)

    # Initialize API client
    napi = get_napi(args.public_id, args.secret_key)

    # --- Download ---
    if args.mode in ("full", "download"):
        file_paths = download_data(napi)
        if args.mode == "download":
            log.info("Download complete.")
            return

    # --- Train ---
    if args.mode in ("full", "train", "diagnostics"):
        if args.mode != "full":
            # Need to find existing data files
            file_paths = {
                "features_json": DATA_DIR / "features.json",
                "train": DATA_DIR / "train.parquet",
                "validation": DATA_DIR / "validation.parquet",
            }
            # Check files exist
            for name, path in file_paths.items():
                if not path.exists():
                    log.error(f"Missing data file: {path}")
                    log.error("Run with --mode download first.")
                    sys.exit(1)

        data = load_data(file_paths)
        if args.ensemble:
            models_dict = train_ensemble(data, save=True)
            model = None  # Not used in ensemble mode
        else:
            model = train_model(data, save=True)

        if args.mode == "train":
            log.info("Training complete.")
            return

    # --- Diagnostics ---
    if args.mode == "diagnostics":
        run_diagnostics(model, data)
        return

    # --- Submit ---
    if args.mode in ("full", "submit"):
        if args.mode == "submit":
            if args.ensemble:
                models_dict, features, targets = load_ensemble()
            else:
                model, features = load_model()
            # Download live data
            current_round = napi.get_current_round()
            live_path = DATA_DIR / "live.parquet"
            if not live_path.exists():
                napi.download_dataset(
                    f"{DATA_VERSION}/live.parquet",
                    str(live_path),
                )
            live_df = pd.read_parquet(live_path)
        else:
            features = data["features"]
            live_df = data["live"]

        if live_df is not None:
            if args.ensemble:
                predict_ensemble(
                    napi,
                    models_dict,
                    features,
                    live_df,
                    model_name=args.model_name,
                    do_neutralize=args.neutralize,
                    dry_run=args.dry_run,
                )
            else:
                predict_and_submit(
                    napi,
                    model,
                    features,
                    live_df,
                    model_name=args.model_name,
                    dry_run=args.dry_run,
                )
        else:
            log.warning("No live data available. Skipping submission.")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
