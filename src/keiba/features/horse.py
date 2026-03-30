"""Horse-level feature engineering."""

import numpy as np
import pandas as pd

from keiba.config import RECENT_RACES_WINDOW


def compute_horse_features(results_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Compute horse features for all horses in a given race.

    Uses only data from races that occurred BEFORE the target race (no leakage).

    Args:
        results_df: Full race_results table joined with races (must include race_date).
        race_id: Target race ID.

    Returns:
        DataFrame with one row per horse in the race, indexed by horse_id.
    """
    target_race = results_df[results_df["race_id"] == race_id]
    if target_race.empty:
        return pd.DataFrame()

    target_date = target_race["race_date"].iloc[0]
    target_distance = target_race["distance"].iloc[0]
    target_surface = target_race["course_type"].iloc[0]
    target_venue = target_race["venue"].iloc[0]

    horse_ids = target_race["horse_id"].unique()

    # Historical data: only prior races
    history = results_df[
        (results_df["race_date"] < target_date) & (results_df["finish_order"] > 0)
    ]

    features = []
    for horse_id in horse_ids:
        h = history[history["horse_id"] == horse_id].sort_values("race_date", ascending=False)
        f = {"horse_id": horse_id}

        # --- Recent performance (last N races) ---
        recent = h.head(RECENT_RACES_WINDOW)
        f["run_count"] = len(h)
        f["recent_run_count"] = len(recent)

        if len(recent) > 0:
            f["win_rate_last5"] = (recent["finish_order"] == 1).mean()
            f["place_rate_last5"] = (recent["finish_order"] <= 3).mean()
            f["avg_finish_last5"] = recent["finish_order"].mean()
            f["best_finish_last5"] = recent["finish_order"].min()
            f["last_3f_avg"] = recent["last_3f"].mean()
            f["days_since_last"] = (target_date - recent["race_date"].iloc[0]).days
        else:
            f["win_rate_last5"] = np.nan
            f["place_rate_last5"] = np.nan
            f["avg_finish_last5"] = np.nan
            f["best_finish_last5"] = np.nan
            f["last_3f_avg"] = np.nan
            f["days_since_last"] = np.nan

        # --- Distance aptitude ---
        dist_cat = _distance_category(target_distance)
        same_dist = h[h["distance"].apply(_distance_category) == dist_cat]
        f["distance_run_count"] = len(same_dist)
        if len(same_dist) > 0:
            f["distance_win_rate"] = (same_dist["finish_order"] == 1).mean()
            f["distance_place_rate"] = (same_dist["finish_order"] <= 3).mean()
        else:
            f["distance_win_rate"] = np.nan
            f["distance_place_rate"] = np.nan

        # --- Surface aptitude ---
        same_surface = h[h["course_type"] == target_surface]
        f["surface_run_count"] = len(same_surface)
        if len(same_surface) > 0:
            f["surface_win_rate"] = (same_surface["finish_order"] == 1).mean()
        else:
            f["surface_win_rate"] = np.nan

        # --- Venue aptitude ---
        same_venue = h[h["venue"] == target_venue]
        f["venue_run_count"] = len(same_venue)
        if len(same_venue) > 0:
            f["venue_win_rate"] = (same_venue["finish_order"] == 1).mean()
        else:
            f["venue_win_rate"] = np.nan

        # --- Last 3F rank (among race competitors) ---
        if len(recent) > 0 and "last_3f_rank" in recent.columns:
            f["last_3f_rank_avg"] = recent["last_3f_rank"].mean()
        else:
            f["last_3f_rank_avg"] = np.nan

        # --- Weight trend ---
        if len(h) >= 2 and h["horse_weight"].notna().sum() >= 2:
            weights = h["horse_weight"].dropna().head(3)
            f["weight_trend"] = weights.iloc[0] - weights.iloc[-1]
        else:
            f["weight_trend"] = np.nan

        # --- Class progression ---
        target_prize = target_race["prize_1st"].iloc[0] if "prize_1st" in target_race.columns else 0
        if len(recent) > 0 and "prize_1st" in recent.columns:
            last_prize = recent["prize_1st"].iloc[0]
            f["class_progression"] = (
                1 if target_prize > last_prize else (-1 if target_prize < last_prize else 0)
            )
        else:
            f["class_progression"] = np.nan

        # --- Age ---
        sex_age = target_race[target_race["horse_id"] == horse_id]["sex_age"].iloc[0]
        age = _parse_age(sex_age)
        f["age"] = age
        f["age_peak_offset"] = abs(age - 4.5) if age else np.nan  # Peak around 4-5

        # --- Grade experience ---
        grade_races = h[h["grade"].isin(["G1", "G2", "G3"])]
        f["grade_experience"] = len(grade_races)
        f["grade_win_count"] = (grade_races["finish_order"] == 1).sum() if len(grade_races) > 0 else 0

        features.append(f)

    return pd.DataFrame(features)


def _distance_category(distance: int) -> str:
    """Categorize race distance."""
    if distance <= 1400:
        return "sprint"
    elif distance <= 1800:
        return "mile"
    elif distance <= 2200:
        return "intermediate"
    else:
        return "long"


def _parse_age(sex_age: str) -> int | None:
    """Extract age from sex_age string like '牡3'."""
    if not sex_age:
        return None
    try:
        return int(sex_age[-1]) if sex_age[-1].isdigit() else None
    except (IndexError, ValueError):
        return None
