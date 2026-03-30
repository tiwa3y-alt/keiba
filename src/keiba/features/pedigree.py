"""Pedigree (bloodline) feature engineering."""

import numpy as np
import pandas as pd


def compute_pedigree_features(
    results_df: pd.DataFrame, horses_df: pd.DataFrame, race_id: str
) -> pd.DataFrame:
    """Compute pedigree-based features for all entries.

    Uses historical performance of sire's and broodmare sire's progeny.

    Args:
        results_df: Full race_results joined with races.
        horses_df: Horses table with pedigree info.
        race_id: Target race ID.

    Returns:
        DataFrame with one row per entry.
    """
    target_race = results_df[results_df["race_id"] == race_id]
    if target_race.empty:
        return pd.DataFrame()

    target_date = target_race["race_date"].iloc[0]
    target_distance = target_race["distance"].iloc[0]
    target_surface = target_race["course_type"].iloc[0]

    history = results_df[
        (results_df["race_date"] < target_date) & (results_df["finish_order"] > 0)
    ]

    # Merge history with horse pedigree info
    history_with_ped = history.merge(
        horses_df[["horse_id", "sire_id", "broodmare_sire_id"]],
        on="horse_id",
        how="left",
    )

    from keiba.features.horse import _distance_category

    dist_cat = _distance_category(target_distance)

    features = []
    for _, entry in target_race.iterrows():
        horse_id = entry["horse_id"]
        f = {"horse_id": horse_id}

        horse_info = horses_df[horses_df["horse_id"] == horse_id]
        if horse_info.empty:
            features.append(f)
            continue

        sire_id = horse_info["sire_id"].iloc[0]
        bms_id = horse_info["broodmare_sire_id"].iloc[0]

        # --- Sire stats ---
        sire_progeny = history_with_ped[history_with_ped["sire_id"] == sire_id]
        if len(sire_progeny) > 0:
            f["sire_win_rate"] = (sire_progeny["finish_order"] == 1).mean()
            f["sire_place_rate"] = (sire_progeny["finish_order"] <= 3).mean()

            # By distance
            sire_dist = sire_progeny[sire_progeny["distance"].apply(_distance_category) == dist_cat]
            if len(sire_dist) >= 5:
                f["sire_distance_rate"] = (sire_dist["finish_order"] <= 3).mean()
            else:
                f["sire_distance_rate"] = np.nan

            # By surface
            sire_surf = sire_progeny[sire_progeny["course_type"] == target_surface]
            if len(sire_surf) >= 5:
                f["sire_surface_rate"] = (sire_surf["finish_order"] <= 3).mean()
            else:
                f["sire_surface_rate"] = np.nan

            # Grade winners
            sire_grade = sire_progeny[sire_progeny["grade"].isin(["G1", "G2", "G3"])]
            f["sire_grade_winners"] = (
                sire_grade[sire_grade["finish_order"] == 1]["horse_id"].nunique()
            )
        else:
            f["sire_win_rate"] = np.nan
            f["sire_place_rate"] = np.nan
            f["sire_distance_rate"] = np.nan
            f["sire_surface_rate"] = np.nan
            f["sire_grade_winners"] = 0

        # --- Broodmare sire stats ---
        bms_progeny = history_with_ped[history_with_ped["broodmare_sire_id"] == bms_id]
        if len(bms_progeny) > 0:
            f["bms_win_rate"] = (bms_progeny["finish_order"] == 1).mean()
            bms_dist = bms_progeny[bms_progeny["distance"].apply(_distance_category) == dist_cat]
            if len(bms_dist) >= 5:
                f["bms_distance_rate"] = (bms_dist["finish_order"] <= 3).mean()
            else:
                f["bms_distance_rate"] = np.nan
        else:
            f["bms_win_rate"] = np.nan
            f["bms_distance_rate"] = np.nan

        features.append(f)

    return pd.DataFrame(features)
