"""Trainer-level feature engineering."""

import numpy as np
import pandas as pd

from keiba.config import STATS_LOOKBACK_DAYS


def compute_trainer_features(results_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Compute trainer features for all entries in a given race.

    Args:
        results_df: Full race_results joined with races.
        race_id: Target race ID.

    Returns:
        DataFrame with one row per entry.
    """
    target_race = results_df[results_df["race_id"] == race_id]
    if target_race.empty:
        return pd.DataFrame()

    target_date = target_race["race_date"].iloc[0]
    cutoff_1y = target_date - pd.Timedelta(days=STATS_LOOKBACK_DAYS)
    cutoff_1m = target_date - pd.Timedelta(days=30)

    history = results_df[
        (results_df["race_date"] < target_date) & (results_df["finish_order"] > 0)
    ]
    history_1y = history[history["race_date"] >= cutoff_1y]

    features = []
    for _, entry in target_race.iterrows():
        trainer_id = entry["trainer_id"]
        f = {"horse_id": entry["horse_id"], "trainer_id": trainer_id}

        th = history_1y[history_1y["trainer_id"] == trainer_id]
        f["trainer_rides_1y"] = len(th)

        if len(th) > 0:
            f["trainer_win_rate_1y"] = (th["finish_order"] == 1).mean()
            f["trainer_place_rate_1y"] = (th["finish_order"] <= 3).mean()
        else:
            f["trainer_win_rate_1y"] = np.nan
            f["trainer_place_rate_1y"] = np.nan

        # Grade performance
        th_grade = th[th["grade"].isin(["G1", "G2", "G3"])]
        if len(th_grade) > 0:
            f["trainer_grade_rate"] = (th_grade["finish_order"] <= 3).mean()
        else:
            f["trainer_grade_rate"] = np.nan

        # Stable form (last 30 days)
        th_recent = history[
            (history["trainer_id"] == trainer_id) & (history["race_date"] >= cutoff_1m)
        ]
        if len(th_recent) >= 3:
            f["trainer_stable_form"] = (th_recent["finish_order"] <= 3).mean()
        else:
            f["trainer_stable_form"] = np.nan

        features.append(f)

    return pd.DataFrame(features)
