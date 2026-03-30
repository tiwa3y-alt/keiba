"""CLI entry point for keiba prediction system."""

import click
import pandas as pd
from tqdm import tqdm

from keiba.config import GRADES


@click.group()
def main():
    """AI-powered Japanese horse racing prediction system."""
    pass


@main.command()
@click.option("--year", required=True, type=int, help="Year to scrape")
@click.option("--grades", default="G1,G2,G3", help="Comma-separated grades (G1,G2,G3)")
@click.option("--with-horses/--no-horses", default=True, help="Also scrape horse profiles")
def scrape(year: int, grades: str, with_horses: bool):
    """Scrape race data from netkeiba.com."""
    from keiba.db.connection import init_db, get_session
    from keiba.scraper.netkeiba import (
        scrape_grade_race_list_by_search,
        scrape_horse_profile,
        scrape_race_result,
        save_horse,
        save_race_data,
    )

    init_db()
    grade_list = [g.strip() for g in grades.split(",")]

    click.echo(f"Fetching grade race list for {year}...")
    race_ids = scrape_grade_race_list_by_search(year)
    click.echo(f"Found {len(race_ids)} races")

    session = get_session()
    scraped_horses = set()

    for race_id in tqdm(race_ids, desc="Scraping races"):
        try:
            data = scrape_race_result(race_id)
            if data is None:
                continue

            # Filter by requested grades
            race_grade = data["race_info"].get("grade", "")
            if race_grade not in grade_list:
                continue

            save_race_data(data, session)

            # Scrape horse profiles
            if with_horses:
                for result in data["results"]:
                    horse_id = result.get("horse_id")
                    if horse_id and horse_id not in scraped_horses:
                        try:
                            profile = scrape_horse_profile(horse_id)
                            if profile:
                                save_horse(profile, session)
                            scraped_horses.add(horse_id)
                        except Exception as e:
                            click.echo(f"  Warning: Failed to scrape horse {horse_id}: {e}")

        except Exception as e:
            click.echo(f"  Error scraping {race_id}: {e}")

    session.close()
    click.echo("Done!")


@main.command()
@click.option("--rebuild/--no-rebuild", default=False, help="Rebuild all features from scratch")
def features(rebuild: bool):
    """Build feature matrix from collected data."""
    from keiba.db.connection import init_db
    from keiba.features.pipeline import build_dataset, load_full_results, load_horses

    init_db()
    click.echo("Loading data from database...")
    results_df = load_full_results()
    horses_df = load_horses()

    if results_df.empty:
        click.echo("No data found. Run 'keiba scrape' first.")
        return

    race_ids = results_df["race_id"].unique().tolist()
    click.echo(f"Building features for {len(race_ids)} races...")

    dataset = build_dataset(results_df, horses_df, race_ids)
    click.echo(f"Dataset shape: {dataset.shape}")

    output_path = "data/features.parquet"
    dataset.to_parquet(output_path, index=False)
    click.echo(f"Features saved to {output_path}")


@main.command()
@click.option("--surface", default="all", help="Surface type: turf, dirt, or all")
@click.option("--start-test-year", default=2021, type=int, help="First test year")
@click.option("--end-test-year", default=2025, type=int, help="Last test year")
@click.option("--model-type", default="rank", help="Model type: rank or binary")
def train(surface: str, start_test_year: int, end_test_year: int, model_type: str):
    """Train prediction model with walk-forward validation."""
    from keiba.models.trainer import save_model, walk_forward_train

    click.echo("Loading feature dataset...")
    try:
        dataset = pd.read_parquet("data/features.parquet")
    except FileNotFoundError:
        click.echo("Feature dataset not found. Run 'keiba features' first.")
        return

    # Filter by surface if specified
    if surface == "turf":
        dataset = dataset[dataset["is_turf"] == 1]
    elif surface == "dirt":
        dataset = dataset[dataset["is_dirt"] == 1]

    click.echo(f"Training {model_type} model (test years: {start_test_year}-{end_test_year})...")
    results = walk_forward_train(dataset, start_test_year, end_test_year, model_type=model_type)

    for r in results:
        year = r["year"]
        preds = r["predictions_df"]
        model = r["model"]

        # Save model
        model_name = f"{model_type}_{surface}_{year}"
        save_model(model, model_name)
        click.echo(f"  Year {year}: saved model '{model_name}'")

        # Quick accuracy check
        if "finish_order" in preds.columns:
            top1_correct = preds.groupby("race_id").apply(
                lambda g: g.loc[g["predicted_prob"].idxmax(), "finish_order"] == 1
            ).mean()
            top3_correct = preds.groupby("race_id").apply(
                lambda g: g.loc[g["predicted_prob"].idxmax(), "finish_order"] <= 3
            ).mean()
            click.echo(f"    Top-1 accuracy: {top1_correct:.1%}, Top-3: {top3_correct:.1%}")

    click.echo("Training complete!")


@main.command()
@click.option("--start-year", default=2021, type=int, help="Backtest start year")
@click.option("--end-year", default=2025, type=int, help="Backtest end year")
@click.option(
    "--bet-types",
    default="単勝",
    help="Comma-separated bet types: 単勝,複勝,馬連,ワイド,馬単,三連複,三連単",
)
@click.option(
    "--strategy",
    default="value",
    help="Strategy: value, kelly, fractional_kelly, topn",
)
@click.option("--ev-threshold", default=1.2, type=float, help="EV threshold for value strategy")
@click.option("--bankroll", default=100000, type=float, help="Initial bankroll (yen)")
def backtest(
    start_year: int,
    end_year: int,
    bet_types: str,
    strategy: str,
    ev_threshold: float,
    bankroll: float,
):
    """Run backtest simulation."""
    from keiba.backtest.engine import BacktestEngine, BaselineEngine
    from keiba.backtest.report import generate_report
    from keiba.backtest.strategies import (
        FractionalKellyStrategy,
        KellyCriterionStrategy,
        TopNStrategy,
        ValueBetStrategy,
    )
    from keiba.models.trainer import walk_forward_train

    click.echo("Loading feature dataset...")
    try:
        dataset = pd.read_parquet("data/features.parquet")
    except FileNotFoundError:
        click.echo("Feature dataset not found. Run 'keiba features' first.")
        return

    # Load payouts
    from keiba.db.connection import get_engine

    payouts_df = pd.read_sql("SELECT * FROM payouts", get_engine())

    # Train and predict via walk-forward
    click.echo("Running walk-forward training...")
    wf_results = walk_forward_train(dataset, start_year, end_year, model_type="rank")

    # Combine all predictions
    all_preds = []
    for r in wf_results:
        pred_df = r["predictions_df"]
        # Add race_date from the dataset
        date_map = dataset[["race_id", "race_date"]].drop_duplicates()
        pred_df = pred_df.merge(date_map, on="race_id", how="left")
        # Add horse_number
        num_map = dataset[["race_id", "horse_id", "horse_number"]].drop_duplicates()
        pred_df = pred_df.merge(num_map, on=["race_id", "horse_id"], how="left")
        all_preds.append(pred_df)

    predictions_df = pd.concat(all_preds, ignore_index=True)

    # Select strategy
    strategy_map = {
        "value": ValueBetStrategy(ev_threshold=ev_threshold),
        "kelly": KellyCriterionStrategy(),
        "fractional_kelly": FractionalKellyStrategy(),
        "topn": TopNStrategy(),
    }
    strat = strategy_map.get(strategy, ValueBetStrategy(ev_threshold=ev_threshold))

    bet_type_list = [bt.strip() for bt in bet_types.split(",")]

    # Run backtest
    click.echo(f"Running backtest ({strategy} strategy, bet types: {bet_type_list})...")
    engine = BacktestEngine(initial_bankroll=bankroll)
    result = engine.run(predictions_df, payouts_df, strat, bet_types=bet_type_list)

    # Run baseline for comparison
    click.echo("\nRunning baseline (favorite only)...")
    baseline_bets = BaselineEngine.favorite_only(predictions_df, payouts_df)
    from keiba.backtest.metrics import compute_metrics

    baseline_metrics = compute_metrics(baseline_bets)
    click.echo(f"  Baseline ROI: {baseline_metrics.get('roi', 0):.1%}")
    click.echo(f"  Baseline hit rate: {baseline_metrics.get('hit_rate', 0):.1%}")

    # Generate report
    generate_report(result)


@main.command()
@click.argument("race_id")
def predict(race_id: str):
    """Predict outcomes for an upcoming race."""
    from keiba.db.connection import init_db
    from keiba.features.pipeline import load_full_results, load_horses
    from keiba.models.predictor import predict_race
    from keiba.models.trainer import load_model

    init_db()
    click.echo(f"Predicting race {race_id}...")

    results_df = load_full_results()
    horses_df = load_horses()

    # Try to load latest model
    try:
        model = load_model("rank_all_2025")
    except Exception:
        try:
            model = load_model("rank_all_2024")
        except Exception:
            click.echo("No trained model found. Run 'keiba train' first.")
            return

    predictions = predict_race(model, results_df, horses_df, race_id)
    if predictions.empty:
        click.echo("No predictions generated. Check if race data is available.")
        return

    click.echo(f"\n{'Rank':>4} {'Horse':>6} {'Prob':>8} {'Score':>8}")
    click.echo("-" * 32)
    for _, row in predictions.iterrows():
        click.echo(
            f"{int(row['predicted_rank']):>4} "
            f"{row['horse_id']:>6} "
            f"{row['predicted_prob']:>7.1%} "
            f"{row['raw_score']:>8.3f}"
        )


if __name__ == "__main__":
    main()
