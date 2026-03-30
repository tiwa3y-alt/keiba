"""Jockey-level feature engineering."""

import numpy as np
import pandas as pd

from keiba.config import STATS_LOOKBACK_DAYS


def compute_jockey_features(results_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Compute jockey features for all entries in a given race.

    Args:
        results_df: Full race_results joined with races.
        race_id: Target race ID.

    Returns:
        DataFrame with one row per entry (horse_id, jockey_id pair).
    """
    target_race = results_df[results_df["race_id"] == race_id]
    if target_race.empty:
        return pd.DataFrame()

    target_date = target_race["race_date"].iloc[0]
    target_venue = target_race["venue"].iloc[0]
    target_distance = target_race["distance"].iloc[0]
    cutoff = target_date - pd.Timedelta(days=STATS_LOOKBACK_DAYS)

    history = results_df[
        (results_df["race_date"] < target_date)
        & (results_df["race_date"] >= cutoff)
        & (results_df["finish_order"] > 0)
    ]

    features = []
    for _, entry in target_race.iterrows():
        jockey_id = entry["jockey_id"]
        horse_id = entry["horse_id"]
        f = {"horse_id": horse_id, "jockey_id": jockey_id}

        jh = history[history["jockey_id"] == jockey_id]
        f["jockey_rides_1y"] = len(jh)

        if len(jh) > 0:
            f["jockey_win_rate_1y"] = (jh["finish_order"] == 1).mean()
            f["jockey_place_rate_1y"] = (jh["finish_order"] <= 3).mean()
        else:
            f["jockey_win_rate_1y"] = np.nan
            f["jockey_place_rate_1y"] = np.nan

        # Grade race performance
        jh_grade = jh[jh["grade"].isin(["G1", "G2", "G3"])]
        if len(jh_grade) > 0:
            f["jockey_grade_win_rate"] = (jh_grade["finish_order"] == 1).mean()
        else:
            f["jockey_grade_win_rate"] = np.nan

        # Venue specific
        jh_venue = jh[jh["venue"] == target_venue]
        if len(jh_venue) >= 3:
            f["jockey_venue_rate"] = (jh_venue["finish_order"] <= 3).mean()
        else:
            f["jockey_venue_rate"] = np.nan

        # Distance specific
        from keiba.features.horse import _distance_category

        dist_cat = _distance_category(target_distance)
        jh_dist = jh[jh["distance"].apply(_distance_category) == dist_cat]
        if len(jh_dist) >= 3:
            f["jockey_distance_rate"] = (jh_dist["finish_order"] <= 3).mean()
        else:
            f["jockey_distance_rate"] = np.nan

        # Horse-jockey combo history
        all_history = results_df[
            (results_df["race_date"] < target_date) & (results_df["finish_order"] > 0)
        ]
        combo = all_history[
            (all_history["jockey_id"] == jockey_id) & (all_history["horse_id"] == horse_id)
        ]
        f["jockey_horse_combo_count"] = len(combo)
        if len(combo) > 0:
            f["jockey_horse_combo_win_rate"] = (combo["finish_order"] == 1).mean()
        else:
            f["jockey_horse_combo_win_rate"] = np.nan

        # Jockey change flag
        horse_history = all_history[all_history["horse_id"] == horse_id].sort_values(
            "race_date", ascending=False
        )
        if len(horse_history) > 0:
            last_jockey = horse_history["jockey_id"].iloc[0]
            f["jockey_change"] = 1 if last_jockey != jockey_id else 0
        else:
            f["jockey_change"] = 0

        features.append(f)

    return pd.DataFrame(features)
