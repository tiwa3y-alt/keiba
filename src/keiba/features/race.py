"""Race-level feature engineering."""

import numpy as np
import pandas as pd


def compute_race_features(results_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Compute race-context features for all entries.

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
    target_venue = target_race["venue"].iloc[0]
    target_surface = target_race["course_type"].iloc[0]
    target_distance = target_race["distance"].iloc[0]

    history = results_df[
        (results_df["race_date"] < target_date) & (results_df["finish_order"] > 0)
    ]

    features = []
    for _, entry in target_race.iterrows():
        f = {"horse_id": entry["horse_id"]}

        # Field size
        f["field_size"] = len(target_race)

        # Distance category
        f["distance_category"] = _encode_distance(target_distance)
        f["distance"] = target_distance

        # Course type encoding
        f["is_turf"] = 1 if target_surface == "芝" else 0
        f["is_dirt"] = 1 if target_surface == "ダート" else 0

        # Track condition encoding
        tc = entry.get("track_condition", "")
        f["track_good"] = 1 if tc == "良" else 0
        f["track_slightly_heavy"] = 1 if tc == "稍重" else 0
        f["track_heavy"] = 1 if tc in ("重", "不良") else 0

        # Frame/post position
        f["frame_number"] = entry.get("frame_number", np.nan)
        f["horse_number"] = entry.get("horse_number", np.nan)

        # Post position bias (historical win rate by post position at this venue/surface)
        venue_surface = history[
            (history["venue"] == target_venue) & (history["course_type"] == target_surface)
        ]
        horse_num = entry.get("horse_number")
        if len(venue_surface) > 50 and horse_num:
            same_post = venue_surface[venue_surface["horse_number"] == horse_num]
            if len(same_post) >= 5:
                f["post_position_bias"] = (same_post["finish_order"] <= 3).mean()
            else:
                f["post_position_bias"] = np.nan
        else:
            f["post_position_bias"] = np.nan

        # Pace prediction: count of front-runners
        pace_count = 0
        for _, e in target_race.iterrows():
            hh = history[history["horse_id"] == e["horse_id"]]
            if len(hh) > 0:
                recent = hh.sort_values("race_date", ascending=False).head(3)
                passing = recent["passing_order"].dropna()
                for po in passing:
                    positions = po.split("-")
                    if positions and positions[0].isdigit() and int(positions[0]) <= 3:
                        pace_count += 1
                        break
        f["front_runner_count"] = pace_count

        # Weight carried
        f["weight_carried"] = entry.get("weight_carried", np.nan)

        features.append(f)

    return pd.DataFrame(features)


def _encode_distance(distance: int) -> int:
    """Encode distance category as integer."""
    if distance <= 1400:
        return 0  # sprint
    elif distance <= 1800:
        return 1  # mile
    elif distance <= 2200:
        return 2  # intermediate
    else:
        return 3  # long
