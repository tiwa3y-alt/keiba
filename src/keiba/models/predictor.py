"""Prediction and probability estimation."""

import numpy as np
import pandas as pd

from keiba.features.pipeline import FEATURE_COLUMNS, build_features_for_race
from keiba.models.calibration import CalibratedPredictor


def predict_race(
    model,
    results_df: pd.DataFrame,
    horses_df: pd.DataFrame,
    race_id: str,
    calibrator: CalibratedPredictor | None = None,
) -> pd.DataFrame:
    """Generate predictions for a race.

    Args:
        model: Trained LightGBM model.
        results_df: Full historical results.
        horses_df: Horse profiles.
        race_id: Target race ID.
        calibrator: Optional probability calibrator.

    Returns:
        DataFrame with horse_id, predicted_prob, and rank.
    """
    features = build_features_for_race(results_df, horses_df, race_id)
    if features.empty:
        return pd.DataFrame()

    feature_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    raw_scores = model.predict(features[feature_cols])

    result = features[["horse_id"]].copy()
    result["raw_score"] = raw_scores

    # Softmax normalization within race
    exp_scores = np.exp(raw_scores - raw_scores.max())
    result["predicted_prob"] = exp_scores / exp_scores.sum()

    if calibrator:
        result["predicted_prob"] = calibrator.calibrate(result["predicted_prob"].values)
        # Re-normalize
        result["predicted_prob"] /= result["predicted_prob"].sum()

    result["predicted_rank"] = result["predicted_prob"].rank(ascending=False).astype(int)
    result = result.sort_values("predicted_rank")

    return result


def compute_harville_probs(win_probs: np.ndarray) -> dict[tuple, float]:
    """Compute exacta/trifecta probabilities using the Harville model.

    P(i wins, j 2nd) = P(i) * P(j) / (1 - P(i))
    P(i wins, j 2nd, k 3rd) = P(i) * P(j) / (1 - P(i)) * P(k) / (1 - P(i) - P(j))

    Args:
        win_probs: Array of win probabilities for each horse (index = horse position in array).

    Returns:
        Dict mapping (i, j) or (i, j, k) tuples to probabilities.
    """
    n = len(win_probs)
    probs = {}

    # Exacta (馬単): top 2 in order
    for i in range(n):
        if win_probs[i] <= 0:
            continue
        for j in range(n):
            if i == j or win_probs[j] <= 0:
                continue
            p = win_probs[i] * win_probs[j] / (1 - win_probs[i])
            probs[("exacta", i, j)] = p

    # Quinella (馬連): top 2 any order
    for i in range(n):
        for j in range(i + 1, n):
            if win_probs[i] <= 0 or win_probs[j] <= 0:
                continue
            p = probs.get(("exacta", i, j), 0) + probs.get(("exacta", j, i), 0)
            probs[("quinella", i, j)] = p

    # Trifecta (三連単): top 3 in order
    for i in range(n):
        if win_probs[i] <= 0:
            continue
        remaining_after_i = 1 - win_probs[i]
        if remaining_after_i <= 0:
            continue
        for j in range(n):
            if i == j or win_probs[j] <= 0:
                continue
            p_j_given_i = win_probs[j] / remaining_after_i
            remaining_after_ij = remaining_after_i - win_probs[j]
            if remaining_after_ij <= 0:
                continue
            for k in range(n):
                if k == i or k == j or win_probs[k] <= 0:
                    continue
                p_k_given_ij = win_probs[k] / remaining_after_ij
                p = win_probs[i] * p_j_given_i * p_k_given_ij
                probs[("trifecta", i, j, k)] = p

    # Trio (三連複): top 3 any order
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total = 0
                for perm in [
                    (i, j, k), (i, k, j), (j, i, k),
                    (j, k, i), (k, i, j), (k, j, i),
                ]:
                    total += probs.get(("trifecta", *perm), 0)
                probs[("trio", i, j, k)] = total

    return probs


def find_value_bets(
    predictions: pd.DataFrame,
    payouts_df: pd.DataFrame,
    race_id: str,
    ev_threshold: float = 1.0,
) -> list[dict]:
    """Find value bets where expected value exceeds threshold.

    Args:
        predictions: Prediction results with horse_id and predicted_prob.
        payouts_df: Historical payout data (for odds lookup).
        race_id: Target race ID.
        ev_threshold: Minimum expected value (1.0 = breakeven).

    Returns:
        List of value bet opportunities.
    """
    value_bets = []

    for _, row in predictions.iterrows():
        prob = row["predicted_prob"]
        horse_id = row["horse_id"]

        # Win bet (単勝)
        odds_row = payouts_df[
            (payouts_df["horse_id"] == horse_id) & (payouts_df["race_id"] == race_id)
        ]
        if not odds_row.empty:
            odds = odds_row["odds"].iloc[0]
            if odds and odds > 0:
                ev = prob * odds
                if ev > ev_threshold:
                    value_bets.append(
                        {
                            "race_id": race_id,
                            "horse_id": horse_id,
                            "bet_type": "単勝",
                            "predicted_prob": prob,
                            "odds": odds,
                            "expected_value": ev,
                            "edge": ev - 1.0,
                        }
                    )

    return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)
