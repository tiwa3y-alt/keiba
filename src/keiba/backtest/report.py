"""Backtesting report generation with charts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_report(backtest_result: dict, output_dir: Path | str = "reports") -> Path:
    """Generate a comprehensive backtest report with charts.

    Args:
        backtest_result: Output from BacktestEngine.run().
        output_dir: Directory to save report files.

    Returns:
        Path to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bets_df = backtest_result["bets_df"]
    metrics = backtest_result["metrics"]
    monthly = backtest_result["monthly_roi"]
    bt_summary = backtest_result["bet_type_summary"]

    if bets_df.empty:
        print("No bets to report.")
        return output_dir

    # 1. Print summary metrics
    _print_summary(metrics, bt_summary, backtest_result["final_bankroll"])

    # 2. Cumulative P&L chart
    _plot_cumulative_pnl(bets_df, output_dir)

    # 3. Drawdown chart
    _plot_drawdown(bets_df, output_dir)

    # 4. Monthly ROI heatmap
    if not monthly.empty:
        _plot_monthly_roi(monthly, output_dir)

    # 5. EV distribution
    _plot_ev_distribution(bets_df, output_dir)

    # 6. Bet type ROI comparison
    if not bt_summary.empty:
        _plot_bet_type_comparison(bt_summary, output_dir)

    # 7. Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    if not monthly.empty:
        monthly.to_csv(output_dir / "monthly_roi.csv", index=False)

    print(f"\nReport saved to: {output_dir}")
    return output_dir


def _print_summary(metrics: dict, bt_summary: pd.DataFrame, final_bankroll: float):
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total bets:        {metrics.get('total_bets', 0):,}")
    print(f"  Total wagered:     {metrics.get('total_bet_amount', 0):,.0f} yen")
    print(f"  Total payout:      {metrics.get('total_payout', 0):,.0f} yen")
    print(f"  Profit:            {metrics.get('profit', 0):+,.0f} yen")
    print(f"  ROI:               {metrics.get('roi', 0):.1%}")
    print(f"  Hit rate:          {metrics.get('hit_rate', 0):.1%}")
    print(f"  Profit factor:     {metrics.get('profit_factor', 0):.2f}")
    print(f"  Max drawdown:      {metrics.get('max_drawdown', 0):,.0f} yen")
    print(f"  Sharpe ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Calmar ratio:      {metrics.get('calmar_ratio', 0):.2f}")
    print(f"  Max win streak:    {metrics.get('max_win_streak', 0)}")
    print(f"  Max lose streak:   {metrics.get('max_lose_streak', 0)}")
    print(f"  Final bankroll:    {final_bankroll:,.0f} yen")

    if not bt_summary.empty:
        print("\n--- By Bet Type ---")
        for _, row in bt_summary.iterrows():
            print(
                f"  {row['bet_type']:6s}: "
                f"ROI={row['roi']:.1%}  "
                f"Hit={row['hit_rate']:.1%}  "
                f"Bets={int(row['count'])}  "
                f"P/L={row['profit']:+,.0f}"
            )
    print("=" * 60)


def _plot_cumulative_pnl(bets_df: pd.DataFrame, output_dir: Path):
    """Plot cumulative profit and loss."""
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative = (bets_df["payout"] - bets_df["bet_amount"]).cumsum()
    ax.plot(range(len(cumulative)), cumulative, linewidth=1.5)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative P&L (yen)")
    ax.set_title("Cumulative Profit & Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_pnl.png", dpi=150)
    plt.close(fig)


def _plot_drawdown(bets_df: pd.DataFrame, output_dir: Path):
    """Plot drawdown chart."""
    fig, ax = plt.subplots(figsize=(12, 4))
    cumulative = (bets_df["payout"] - bets_df["bet_amount"]).cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.4, color="red")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Drawdown (yen)")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown.png", dpi=150)
    plt.close(fig)


def _plot_monthly_roi(monthly: pd.DataFrame, output_dir: Path):
    """Plot monthly ROI heatmap."""
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_copy = monthly.copy()
    monthly_copy["month"] = monthly_copy["month"].astype(str)
    ax.bar(monthly_copy["month"], (monthly_copy["roi"] - 1) * 100, color=[
        "green" if x > 1 else "red" for x in monthly_copy["roi"]
    ])
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("ROI - 100% (%)")
    ax.set_title("Monthly ROI")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_roi.png", dpi=150)
    plt.close(fig)


def _plot_ev_distribution(bets_df: pd.DataFrame, output_dir: Path):
    """Plot expected value distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ev_values = bets_df["expected_value"].dropna()
    ax.hist(ev_values, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=1.0, color="red", linestyle="--", label="Breakeven (EV=1.0)")
    ax.set_xlabel("Expected Value")
    ax.set_ylabel("Count")
    ax.set_title("Expected Value Distribution of Bets")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ev_distribution.png", dpi=150)
    plt.close(fig)


def _plot_bet_type_comparison(bt_summary: pd.DataFrame, output_dir: Path):
    """Plot ROI comparison by bet type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROI
    colors = ["green" if x > 1 else "red" for x in bt_summary["roi"]]
    axes[0].barh(bt_summary["bet_type"], (bt_summary["roi"] - 1) * 100, color=colors)
    axes[0].axvline(x=0, color="black", linewidth=0.8)
    axes[0].set_xlabel("ROI - 100% (%)")
    axes[0].set_title("ROI by Bet Type")

    # Hit rate
    axes[1].barh(bt_summary["bet_type"], bt_summary["hit_rate"] * 100, color="steelblue")
    axes[1].set_xlabel("Hit Rate (%)")
    axes[1].set_title("Hit Rate by Bet Type")

    fig.tight_layout()
    fig.savefig(output_dir / "bet_type_comparison.png", dpi=150)
    plt.close(fig)
