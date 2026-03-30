"""Model training with walk-forward validation."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from keiba.config import MODELS_DIR, RANDOM_SEED
from keiba.features.pipeline import FEATURE_COLUMNS


# Default LightGBM parameters
DEFAULT_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
}

DEFAULT_BINARY_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
}


def _get_group_sizes(df: pd.DataFrame) -> list[int]:
    """Get group sizes for LambdaRank (number of horses per race)."""
    return df.groupby("race_id").size().tolist()


def train_rank_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train a LambdaRank model for race outcome prediction.

    Args:
        train_df: Training data with features and finish_order.
        valid_df: Validation data (optional).
        params: LightGBM parameters (uses defaults if None).
        num_boost_round: Maximum boosting rounds.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Trained LightGBM Booster.
    """
    params = params or DEFAULT_RANK_PARAMS.copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    # For LambdaRank, labels are relevance scores (higher = better)
    # Convert finish_order to relevance: max_finish - finish_order
    max_finish = train_df["finish_order"].max()
    train_labels = max_finish - train_df["finish_order"]

    train_data = lgb.Dataset(
        train_df[feature_cols],
        label=train_labels,
        group=_get_group_sizes(train_df),
        free_raw_data=False,
    )

    valid_sets = []
    if valid_df is not None and not valid_df.empty:
        valid_labels = max_finish - valid_df["finish_order"]
        valid_data = lgb.Dataset(
            valid_df[feature_cols],
            label=valid_labels,
            group=_get_group_sizes(valid_df),
            reference=train_data,
            free_raw_data=False,
        )
        valid_sets.append(valid_data)

    callbacks = []
    if valid_sets:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    callbacks.append(lgb.log_evaluation(100))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    return model


def train_binary_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    target_col: str = "is_place",
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train a binary classification model (win or place probability).

    Args:
        train_df: Training data.
        valid_df: Validation data.
        target_col: "is_win" or "is_place".
        params: LightGBM parameters.
        num_boost_round: Max boosting rounds.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Trained LightGBM Booster.
    """
    params = params or DEFAULT_BINARY_PARAMS.copy()

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    train_data = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[target_col],
        free_raw_data=False,
    )

    valid_sets = []
    if valid_df is not None and not valid_df.empty:
        valid_data = lgb.Dataset(
            valid_df[feature_cols],
            label=valid_df[target_col],
            reference=train_data,
            free_raw_data=False,
        )
        valid_sets.append(valid_data)

    callbacks = []
    if valid_sets:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    callbacks.append(lgb.log_evaluation(100))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    return model


def walk_forward_train(
    df: pd.DataFrame,
    start_test_year: int,
    end_test_year: int,
    model_type: str = "rank",
    target_col: str = "is_place",
) -> list[dict]:
    """Walk-forward training and evaluation.

    For each test year, trains on all prior data and evaluates on the test year.

    Args:
        df: Full dataset with features, targets, and race_date.
        start_test_year: First year to use as test.
        end_test_year: Last year to use as test.
        model_type: "rank" for LambdaRank, "binary" for classification.
        target_col: Target column for binary model.

    Returns:
        List of dicts with keys: year, model, predictions_df.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["race_date"]).dt.year

    results = []
    for test_year in range(start_test_year, end_test_year + 1):
        train_data = df[df["year"] < test_year]
        test_data = df[df["year"] == test_year]

        if train_data.empty or test_data.empty:
            continue

        # Use last year of training as validation for early stopping
        val_year = test_year - 1
        val_data = train_data[train_data["year"] == val_year]
        pure_train = train_data[train_data["year"] < val_year]

        if pure_train.empty:
            pure_train = train_data
            val_data = None

        if model_type == "rank":
            model = train_rank_model(pure_train, val_data)
        else:
            model = train_binary_model(pure_train, val_data, target_col=target_col)

        # Predict
        feature_cols = [c for c in FEATURE_COLUMNS if c in test_data.columns]
        raw_preds = model.predict(test_data[feature_cols])

        pred_df = test_data[["race_id", "horse_id", "finish_order", "odds"]].copy()
        pred_df["raw_score"] = raw_preds

        # Convert to probability within each race (softmax)
        pred_df["predicted_prob"] = pred_df.groupby("race_id")["raw_score"].transform(
            lambda x: np.exp(x) / np.exp(x).sum()
        )

        results.append(
            {
                "year": test_year,
                "model": model,
                "predictions_df": pred_df,
            }
        )

    return results


def save_model(model: lgb.Booster, name: str) -> Path:
    """Save a trained model to disk."""
    path = MODELS_DIR / f"{name}.txt"
    model.save_model(str(path))

    # Save feature importance
    importance = dict(
        zip(model.feature_name(), model.feature_importance(importance_type="gain"))
    )
    importance_path = MODELS_DIR / f"{name}_importance.json"
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

    return path


def load_model(name: str) -> lgb.Booster:
    """Load a saved model from disk."""
    path = MODELS_DIR / f"{name}.txt"
    return lgb.Booster(model_file=str(path))
