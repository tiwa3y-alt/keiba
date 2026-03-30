"""Feature pipeline - combines all feature modules into a single feature matrix."""

import pandas as pd
from sqlalchemy import text

from keiba.db.connection import get_engine, init_db
from keiba.features.horse import compute_horse_features
from keiba.features.jockey import compute_jockey_features
from keiba.features.market import compute_market_features
from keiba.features.pedigree import compute_pedigree_features
from keiba.features.race import compute_race_features
from keiba.features.trainer import compute_trainer_features


def load_full_results() -> pd.DataFrame:
    """Load all race results joined with race metadata from DB."""
    init_db()
    engine = get_engine()
    query = text("""
        SELECT
            rr.*,
            r.race_name, r.race_date, r.venue, r.course_type, r.distance,
            r.track_condition, r.weather, r.grade, r.race_class,
            r.head_count, r.prize_1st
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        ORDER BY r.race_date
    """)
    df = pd.read_sql(query, engine)
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def load_horses() -> pd.DataFrame:
    """Load all horse profiles from DB."""
    engine = get_engine()
    return pd.read_sql("SELECT * FROM horses", engine)


def build_features_for_race(
    results_df: pd.DataFrame, horses_df: pd.DataFrame, race_id: str
) -> pd.DataFrame:
    """Build complete feature matrix for a single race.

    Args:
        results_df: Full results DataFrame (from load_full_results).
        horses_df: Horses DataFrame (from load_horses).
        race_id: Target race ID.

    Returns:
        DataFrame with all features, one row per entry.
    """
    horse_f = compute_horse_features(results_df, race_id)
    jockey_f = compute_jockey_features(results_df, race_id)
    trainer_f = compute_trainer_features(results_df, race_id)
    race_f = compute_race_features(results_df, race_id)
    pedigree_f = compute_pedigree_features(results_df, horses_df, race_id)
    market_f = compute_market_features(results_df, race_id)

    # Merge all on horse_id
    merged = horse_f
    for df in [jockey_f, trainer_f, race_f, pedigree_f, market_f]:
        if not df.empty and "horse_id" in df.columns:
            # Drop duplicate columns before merge
            overlap = [c for c in df.columns if c in merged.columns and c != "horse_id"]
            df = df.drop(columns=[c for c in overlap if c != "horse_id"])
            merged = merged.merge(df, on="horse_id", how="left")

    return merged


def build_dataset(
    results_df: pd.DataFrame, horses_df: pd.DataFrame, race_ids: list[str]
) -> pd.DataFrame:
    """Build feature dataset for multiple races.

    Args:
        results_df: Full results DataFrame.
        horses_df: Horses DataFrame.
        race_ids: List of race IDs to process.

    Returns:
        DataFrame with features + target columns for all races.
    """
    all_features = []
    for race_id in race_ids:
        features = build_features_for_race(results_df, horses_df, race_id)
        if features.empty:
            continue

        # Add target variables
        race_results = results_df[results_df["race_id"] == race_id]
        targets = race_results[["horse_id", "finish_order", "race_id"]].copy()
        targets["is_win"] = (targets["finish_order"] == 1).astype(int)
        targets["is_place"] = (targets["finish_order"] <= 3).astype(int)

        merged = features.merge(targets, on="horse_id", how="left")
        all_features.append(merged)

    if not all_features:
        return pd.DataFrame()

    return pd.concat(all_features, ignore_index=True)


# Feature columns used for model training
FEATURE_COLUMNS = [
    # Horse features
    "run_count",
    "recent_run_count",
    "win_rate_last5",
    "place_rate_last5",
    "avg_finish_last5",
    "best_finish_last5",
    "last_3f_avg",
    "days_since_last",
    "distance_run_count",
    "distance_win_rate",
    "distance_place_rate",
    "surface_run_count",
    "surface_win_rate",
    "venue_run_count",
    "venue_win_rate",
    "weight_trend",
    "class_progression",
    "age",
    "age_peak_offset",
    "grade_experience",
    "grade_win_count",
    # Jockey features
    "jockey_rides_1y",
    "jockey_win_rate_1y",
    "jockey_place_rate_1y",
    "jockey_grade_win_rate",
    "jockey_venue_rate",
    "jockey_distance_rate",
    "jockey_horse_combo_count",
    "jockey_horse_combo_win_rate",
    "jockey_change",
    # Trainer features
    "trainer_rides_1y",
    "trainer_win_rate_1y",
    "trainer_place_rate_1y",
    "trainer_grade_rate",
    "trainer_stable_form",
    # Race features
    "field_size",
    "distance_category",
    "distance",
    "is_turf",
    "is_dirt",
    "track_good",
    "track_slightly_heavy",
    "track_heavy",
    "frame_number",
    "horse_number",
    "post_position_bias",
    "front_runner_count",
    "weight_carried",
    # Pedigree features
    "sire_win_rate",
    "sire_place_rate",
    "sire_distance_rate",
    "sire_surface_rate",
    "sire_grade_winners",
    "bms_win_rate",
    "bms_distance_rate",
    # Market features
    "win_odds",
    "odds_implied_prob",
    "log_odds",
    "popularity_rank",
    "odds_rank_normalized",
    "is_favorite",
]
