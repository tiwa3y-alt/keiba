"""Run the full pipeline: features -> train -> backtest -> report.

Optimized version that builds features in bulk using vectorized operations.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from keiba.backtest.engine import BacktestEngine, BaselineEngine
from keiba.backtest.metrics import compute_metrics
from keiba.backtest.report import generate_report
from keiba.backtest.strategies import FractionalKellyStrategy, ValueBetStrategy
from keiba.config import MODELS_DIR, RANDOM_SEED
from keiba.db.connection import get_engine, init_db
from keiba.features.pipeline import FEATURE_COLUMNS


def load_data():
    """Load all data from database."""
    engine = get_engine()
    results = pd.read_sql("""
        SELECT rr.*, r.race_name, r.race_date, r.venue, r.course_type,
               r.distance, r.track_condition, r.weather, r.grade,
               r.race_class, r.head_count, r.prize_1st
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        ORDER BY r.race_date, rr.race_id
    """, engine)
    results["race_date"] = pd.to_datetime(results["race_date"])

    horses = pd.read_sql("SELECT * FROM horses", engine)
    payouts = pd.read_sql("SELECT * FROM payouts", engine)
    return results, horses, payouts


def build_features_fast(results_df: pd.DataFrame, horses_df: pd.DataFrame) -> pd.DataFrame:
    """Build features using vectorized operations (much faster than per-race loop)."""
    print("Building features (optimized)...")
    df = results_df.copy()
    df = df[df["finish_order"] > 0].copy()  # Remove scratched horses
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    # --- Pre-compute rolling stats per horse ---
    print("  Computing horse rolling stats...")
    horse_groups = df.groupby("horse_id")

    # Shift to avoid leakage (only use prior races)
    df["prev_finish"] = horse_groups["finish_order"].shift(1)
    df["is_prev_win"] = (df["prev_finish"] == 1).astype(float)
    df["is_prev_place"] = (df["prev_finish"] <= 3).astype(float)

    # Rolling averages (last 5 races)
    for col, new_col in [
        ("finish_order", "avg_finish_last5"),
        ("last_3f", "last_3f_avg"),
    ]:
        df[new_col] = horse_groups[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    # Rolling win/place rates
    df["win_rate_last5"] = horse_groups["is_prev_win"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["place_rate_last5"] = horse_groups["is_prev_place"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Best finish in last 5
    df["best_finish_last5"] = horse_groups["finish_order"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).min()
    )

    # Run count (cumulative prior races)
    df["run_count"] = horse_groups.cumcount()
    df["recent_run_count"] = horse_groups["finish_order"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=0).count()
    )

    # Days since last race
    df["prev_race_date"] = horse_groups["race_date"].shift(1)
    df["days_since_last"] = (df["race_date"] - df["prev_race_date"]).dt.days

    # Weight trend
    df["prev_weight"] = horse_groups["horse_weight"].shift(1)
    df["weight_trend"] = df["horse_weight"] - df["prev_weight"]

    # --- Age features ---
    print("  Computing age/sex features...")
    df["age"] = df["sex_age"].str.extract(r"(\d+)$").astype(float)
    df["age_peak_offset"] = (df["age"] - 4.5).abs()

    # --- Distance/surface aptitude (simplified but fast) ---
    print("  Computing aptitude features...")
    df["distance_category"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Surface encoding
    df["is_turf"] = (df["course_type"] == "芝").astype(int)
    df["is_dirt"] = (df["course_type"] == "ダート").astype(int)

    # Track condition encoding
    df["track_good"] = (df["track_condition"] == "良").astype(int)
    df["track_slightly_heavy"] = (df["track_condition"] == "稍重").astype(int)
    df["track_heavy"] = (df["track_condition"].isin(["重", "不良"])).astype(int)

    # --- Jockey features (rolling by jockey) ---
    print("  Computing jockey features...")
    jockey_groups = df.groupby("jockey_id")
    df["jockey_prev_win"] = (jockey_groups["finish_order"].shift(1) == 1).astype(float)
    df["jockey_prev_place"] = (jockey_groups["finish_order"].shift(1) <= 3).astype(float)

    df["jockey_win_rate_1y"] = jockey_groups["jockey_prev_win"].transform(
        lambda x: x.rolling(50, min_periods=5).mean()
    )
    df["jockey_place_rate_1y"] = jockey_groups["jockey_prev_place"].transform(
        lambda x: x.rolling(50, min_periods=5).mean()
    )
    df["jockey_rides_1y"] = jockey_groups.cumcount()

    # Jockey change detection
    df["prev_jockey"] = horse_groups["jockey_id"].shift(1)
    df["jockey_change"] = (df["jockey_id"] != df["prev_jockey"]).astype(int)
    df.loc[df["run_count"] == 0, "jockey_change"] = 0

    # --- Trainer features ---
    print("  Computing trainer features...")
    trainer_groups = df.groupby("trainer_id")
    df["trainer_prev_win"] = (trainer_groups["finish_order"].shift(1) == 1).astype(float)
    df["trainer_prev_place"] = (trainer_groups["finish_order"].shift(1) <= 3).astype(float)

    df["trainer_win_rate_1y"] = trainer_groups["trainer_prev_win"].transform(
        lambda x: x.rolling(50, min_periods=5).mean()
    )
    df["trainer_place_rate_1y"] = trainer_groups["trainer_prev_place"].transform(
        lambda x: x.rolling(50, min_periods=5).mean()
    )
    df["trainer_rides_1y"] = trainer_groups.cumcount()

    # --- Market features ---
    print("  Computing market features...")
    df["win_odds"] = df["odds"]
    df["log_odds"] = np.log(df["odds"].clip(lower=1.01))
    df["popularity_rank"] = df["popularity"]

    # Implied probability (normalized within race)
    race_groups = df.groupby("race_id")
    df["inv_odds"] = 1.0 / df["odds"].clip(lower=1.01)
    df["total_inv_odds"] = race_groups["inv_odds"].transform("sum")
    df["odds_implied_prob"] = df["inv_odds"] / df["total_inv_odds"]
    df["odds_rank_normalized"] = df["popularity"] / race_groups["horse_number"].transform("count")
    df["is_favorite"] = (df["popularity"] == 1).astype(int)

    # --- Race-level features ---
    df["field_size"] = race_groups["horse_number"].transform("count")

    # --- Pedigree (simplified: use sire stats from prior data) ---
    print("  Computing pedigree features...")
    if "sire_id" in horses_df.columns:
        df = df.merge(
            horses_df[["horse_id", "sire_id", "broodmare_sire_id"]],
            on="horse_id", how="left"
        )

        # Sire win rate (from all prior results - simplified)
        sire_groups = df.groupby("sire_id")
        df["sire_prev_win"] = (sire_groups["finish_order"].shift(1) == 1).astype(float)
        df["sire_win_rate"] = sire_groups["sire_prev_win"].transform(
            lambda x: x.rolling(100, min_periods=10).mean()
        )
        df["sire_place_rate"] = df["sire_win_rate"] * 2.5  # Approximation

        bms_groups = df.groupby("broodmare_sire_id")
        df["bms_prev_win"] = (bms_groups["finish_order"].shift(1) == 1).astype(float)
        df["bms_win_rate"] = bms_groups["bms_prev_win"].transform(
            lambda x: x.rolling(100, min_periods=10).mean()
        )
    else:
        df["sire_win_rate"] = np.nan
        df["sire_place_rate"] = np.nan
        df["bms_win_rate"] = np.nan

    # Fill remaining feature columns with NaN if missing
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # --- Target variables ---
    df["is_win"] = (df["finish_order"] == 1).astype(int)
    df["is_place"] = (df["finish_order"] <= 3).astype(int)

    # Grade experience (cumulative count of grade races per horse)
    df["is_grade"] = df["grade"].isin(["G1", "G2", "G3"]).astype(int)
    horse_groups2 = df.groupby("horse_id")
    df["grade_experience"] = horse_groups2["is_grade"].cumsum() - df["is_grade"]
    df["grade_win_flag"] = ((df["finish_order"] == 1) & df["grade"].isin(["G1", "G2", "G3"])).astype(int)
    df["grade_win_count"] = horse_groups2["grade_win_flag"].cumsum() - df["grade_win_flag"]

    print(f"  Dataset shape: {df.shape}")
    return df


def train_and_predict(df: pd.DataFrame, start_test_year=2021, end_test_year=2025):
    """Walk-forward training with LightGBM."""
    print(f"\nTraining models (walk-forward {start_test_year}-{end_test_year})...")

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df["year"] = df["race_date"].dt.year

    all_predictions = []

    for test_year in range(start_test_year, end_test_year + 1):
        train_data = df[df["year"] < test_year].copy()
        test_data = df[df["year"] == test_year].copy()

        if train_data.empty or test_data.empty:
            continue

        # Validation: previous year
        val_data = train_data[train_data["year"] == test_year - 1]
        pure_train = train_data[train_data["year"] < test_year - 1]
        if pure_train.empty:
            pure_train = train_data
            val_data = pd.DataFrame()

        # Prepare LambdaRank
        max_finish = pure_train["finish_order"].max()
        train_labels = max_finish - pure_train["finish_order"]
        train_groups = pure_train.groupby("race_id").size().tolist()

        train_ds = lgb.Dataset(
            pure_train[feature_cols], label=train_labels,
            group=train_groups, free_raw_data=False,
        )

        valid_sets = []
        callbacks = [lgb.log_evaluation(0)]

        if not val_data.empty:
            val_labels = max_finish - val_data["finish_order"]
            val_groups = val_data.groupby("race_id").size().tolist()
            val_ds = lgb.Dataset(
                val_data[feature_cols], label=val_labels,
                group=val_groups, reference=train_ds, free_raw_data=False,
            )
            valid_sets.append(val_ds)
            callbacks.append(lgb.early_stopping(30))

        params = {
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

        model = lgb.train(
            params, train_ds, num_boost_round=300,
            valid_sets=valid_sets, callbacks=callbacks,
        )

        # Predict
        raw_scores = model.predict(test_data[feature_cols])
        pred_df = test_data[["race_id", "horse_id", "horse_number", "finish_order",
                              "odds", "race_date"]].copy()
        pred_df["raw_score"] = raw_scores
        pred_df["predicted_prob"] = pred_df.groupby("race_id")["raw_score"].transform(
            lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        )

        # Quick accuracy check
        top1 = pred_df.groupby("race_id").apply(
            lambda g: g.loc[g["predicted_prob"].idxmax(), "finish_order"] == 1
        ).mean()
        top3 = pred_df.groupby("race_id").apply(
            lambda g: g.loc[g["predicted_prob"].idxmax(), "finish_order"] <= 3
        ).mean()
        n_races = test_data["race_id"].nunique()
        print(f"  {test_year}: {n_races} races | Top-1: {top1:.1%} | Top-3: {top3:.1%}")

        all_predictions.append(pred_df)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODELS_DIR / f"rank_all_{test_year}.txt"))

    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"  Total predictions: {len(predictions)} entries across {predictions['race_id'].nunique()} races")

    # Feature importance
    importance = dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\n  Top 15 features:")
    for name, imp in top_features:
        print(f"    {name:30s} {imp:10.1f}")

    return predictions


def run_backtest(predictions_df, payouts_df):
    """Run backtests with multiple strategies and bet types."""
    print("\n" + "=" * 60)
    print("RUNNING BACKTESTS")
    print("=" * 60)

    # --- Strategy 1: Value Bet (単勝) ---
    print("\n[1] Value Bet Strategy (単勝, EV > 1.2)")
    engine = BacktestEngine(initial_bankroll=100_000)
    strategy = ValueBetStrategy(ev_threshold=1.2, fixed_bet=1000)
    result_win = engine.run(predictions_df, payouts_df, strategy, bet_types=["単勝"])

    # --- Strategy 2: Value Bet (複勝) ---
    print("\n[2] Value Bet Strategy (複勝, EV > 1.1)")
    strategy_place = ValueBetStrategy(ev_threshold=1.1, fixed_bet=1000)
    result_place = engine.run(predictions_df, payouts_df, strategy_place, bet_types=["複勝"])

    # --- Strategy 3: Fractional Kelly (単勝) ---
    print("\n[3] Fractional Kelly Strategy (単勝)")
    kelly = FractionalKellyStrategy(fraction=0.25, min_edge=0.03, max_fraction=0.05)
    result_kelly = engine.run(predictions_df, payouts_df, kelly, bet_types=["単勝"])

    # --- Strategy 4: Value Bet + Multi bet types ---
    print("\n[4] Value Bet Multi (単勝+馬連+三連複)")
    multi_result = engine.run(
        predictions_df, payouts_df,
        ValueBetStrategy(ev_threshold=1.3, fixed_bet=500),
        bet_types=["単勝", "馬連", "三連複"],
    )

    # --- Baseline: Favorite Only ---
    print("\n[Baseline] Always bet 1st favorite (単勝)")
    baseline_bets = BaselineEngine.favorite_only(predictions_df, payouts_df)
    baseline_metrics = compute_metrics(baseline_bets)
    if baseline_metrics:
        print(f"  ROI: {baseline_metrics.get('roi', 0):.1%}")
        print(f"  Hit rate: {baseline_metrics.get('hit_rate', 0):.1%}")
        print(f"  Profit: {baseline_metrics.get('profit', 0):+,.0f} yen")

    # Generate report for the best strategy
    print("\n--- Generating Report for Value Bet (単勝) ---")
    generate_report(result_win, output_dir="reports/value_win")

    print("\n--- Generating Report for Multi Bet ---")
    generate_report(multi_result, output_dir="reports/value_multi")

    return {
        "value_win": result_win,
        "value_place": result_place,
        "kelly": result_kelly,
        "multi": multi_result,
        "baseline_metrics": baseline_metrics,
    }


def main():
    t0 = time.time()
    init_db()

    # Load data
    print("Loading data...")
    results_df, horses_df, payouts_df = load_data()
    print(f"  {len(results_df)} results, {results_df['race_id'].nunique()} races, {len(horses_df)} horses")

    # Build features
    dataset = build_features_fast(results_df, horses_df)

    # Train and predict
    predictions = train_and_predict(dataset, start_test_year=2021, end_test_year=2025)

    # Run backtests
    results = run_backtest(predictions, payouts_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
