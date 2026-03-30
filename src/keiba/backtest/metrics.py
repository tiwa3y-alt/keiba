"""Backtesting evaluation metrics."""

import numpy as np
import pandas as pd


def compute_metrics(bets_df: pd.DataFrame) -> dict:
    """Compute comprehensive backtesting metrics.

    Args:
        bets_df: DataFrame with columns: bet_amount, payout, race_date, bet_type.

    Returns:
        Dict of metric name -> value.
    """
    if bets_df.empty:
        return {}

    total_bet = bets_df["bet_amount"].sum()
    total_payout = bets_df["payout"].sum()
    profit = total_payout - total_bet
    is_hit = bets_df["payout"] > 0

    metrics = {
        "total_bets": len(bets_df),
        "total_bet_amount": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "roi": total_payout / total_bet if total_bet > 0 else 0,
        "hit_rate": is_hit.mean() if len(bets_df) > 0 else 0,
        "avg_odds_when_hit": (
            bets_df.loc[is_hit, "payout"].mean() / bets_df.loc[is_hit, "bet_amount"].mean()
            if is_hit.sum() > 0
            else 0
        ),
    }

    # Profit factor
    wins = bets_df.loc[is_hit, "payout"].sum() - bets_df.loc[is_hit, "bet_amount"].sum()
    losses = bets_df.loc[~is_hit, "bet_amount"].sum()
    metrics["profit_factor"] = wins / losses if losses > 0 else float("inf")

    # Expected value per bet
    metrics["avg_ev"] = profit / len(bets_df) if len(bets_df) > 0 else 0

    # Drawdown analysis
    cumulative = (bets_df["payout"] - bets_df["bet_amount"]).cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    metrics["max_drawdown"] = abs(drawdown.min()) if len(drawdown) > 0 else 0
    metrics["max_drawdown_pct"] = (
        metrics["max_drawdown"] / running_max.max() if running_max.max() > 0 else 0
    )

    # Sharpe ratio (annualized, assuming ~50 race days per year)
    returns = bets_df["payout"] / bets_df["bet_amount"] - 1
    if returns.std() > 0:
        metrics["sharpe_ratio"] = returns.mean() / returns.std() * np.sqrt(50)
    else:
        metrics["sharpe_ratio"] = 0

    # Calmar ratio
    if metrics["max_drawdown"] > 0:
        annualized_return = profit / total_bet
        metrics["calmar_ratio"] = annualized_return / (metrics["max_drawdown"] / total_bet)
    else:
        metrics["calmar_ratio"] = float("inf")

    # Streaks
    streak = 0
    max_win_streak = 0
    max_lose_streak = 0
    current_type = None

    for hit in is_hit:
        if hit == current_type:
            streak += 1
        else:
            if current_type is True:
                max_win_streak = max(max_win_streak, streak)
            elif current_type is False:
                max_lose_streak = max(max_lose_streak, streak)
            current_type = hit
            streak = 1

    if current_type is True:
        max_win_streak = max(max_win_streak, streak)
    elif current_type is False:
        max_lose_streak = max(max_lose_streak, streak)

    metrics["max_win_streak"] = max_win_streak
    metrics["max_lose_streak"] = max_lose_streak

    return metrics


def compute_monthly_roi(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly ROI breakdown.

    Args:
        bets_df: DataFrame with bet_amount, payout, race_date.

    Returns:
        DataFrame with month, bets, bet_amount, payout, roi.
    """
    if bets_df.empty:
        return pd.DataFrame()

    df = bets_df.copy()
    df["month"] = pd.to_datetime(df["race_date"]).dt.to_period("M")

    monthly = df.groupby("month").agg(
        bets=("bet_amount", "count"),
        bet_amount=("bet_amount", "sum"),
        payout=("payout", "sum"),
    )
    monthly["roi"] = monthly["payout"] / monthly["bet_amount"]
    monthly["profit"] = monthly["payout"] - monthly["bet_amount"]
    monthly["cumulative_profit"] = monthly["profit"].cumsum()

    return monthly.reset_index()


def compute_bet_type_summary(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bet-type summary.

    Returns:
        DataFrame with bet_type, count, roi, hit_rate, avg_payout.
    """
    if bets_df.empty:
        return pd.DataFrame()

    summary = bets_df.groupby("bet_type").agg(
        count=("bet_amount", "count"),
        total_bet=("bet_amount", "sum"),
        total_payout=("payout", "sum"),
        hits=("payout", lambda x: (x > 0).sum()),
    )
    summary["roi"] = summary["total_payout"] / summary["total_bet"]
    summary["hit_rate"] = summary["hits"] / summary["count"]
    summary["profit"] = summary["total_payout"] - summary["total_bet"]

    return summary.reset_index()
