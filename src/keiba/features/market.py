"""Market (odds) feature engineering."""

import numpy as np
import pandas as pd


def compute_market_features(results_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Compute market-based features for all entries.

    Args:
        results_df: Full race_results joined with races.
        race_id: Target race ID.

    Returns:
        DataFrame with one row per entry.
    """
    target_race = results_df[results_df["race_id"] == race_id]
    if target_race.empty:
        return pd.DataFrame()

    features = []
    total_inv_odds = 0.0
    odds_list = []

    for _, entry in target_race.iterrows():
        odds = entry.get("odds")
        if odds and odds > 0:
            total_inv_odds += 1.0 / odds
            odds_list.append(odds)

    for _, entry in target_race.iterrows():
        f = {"horse_id": entry["horse_id"]}

        odds = entry.get("odds")
        popularity = entry.get("popularity")

        f["win_odds"] = odds if odds and odds > 0 else np.nan
        f["popularity_rank"] = popularity if popularity else np.nan

        # Market-implied probability (normalized to remove overround)
        if odds and odds > 0 and total_inv_odds > 0:
            raw_prob = 1.0 / odds
            f["odds_implied_prob"] = raw_prob / total_inv_odds
        else:
            f["odds_implied_prob"] = np.nan

        # Log odds (useful feature for models)
        if odds and odds > 0:
            f["log_odds"] = np.log(odds)
        else:
            f["log_odds"] = np.nan

        # Odds rank normalized by field size
        field_size = len(target_race)
        if popularity and field_size > 0:
            f["odds_rank_normalized"] = popularity / field_size
        else:
            f["odds_rank_normalized"] = np.nan

        # Is favorite flag
        f["is_favorite"] = 1 if popularity == 1 else 0

        features.append(f)

    return pd.DataFrame(features)
