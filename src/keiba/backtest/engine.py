"""Backtesting engine - simulates betting over historical data."""

import numpy as np
import pandas as pd
from sqlalchemy import text

from keiba.backtest.metrics import compute_bet_type_summary, compute_metrics, compute_monthly_roi
from keiba.backtest.strategies import BettingStrategy
from keiba.db.connection import get_engine
from keiba.models.predictor import compute_harville_probs


class BacktestEngine:
    """Run backtests on historical race data with various betting strategies."""

    def __init__(self, initial_bankroll: float = 100_000):
        self.initial_bankroll = initial_bankroll

    def run(
        self,
        predictions_df: pd.DataFrame,
        payouts_df: pd.DataFrame,
        strategy: BettingStrategy,
        bet_types: list[str] | None = None,
    ) -> dict:
        """Execute backtest.

        Args:
            predictions_df: Model predictions with columns:
                race_id, horse_id, predicted_prob, finish_order, odds, race_date.
            payouts_df: Historical payouts with columns:
                race_id, bet_type, combination, payout.
            strategy: Betting strategy to use.
            bet_types: Which bet types to simulate. Default: ["単勝"].

        Returns:
            Dict with: bets_df, metrics, monthly_roi, bet_type_summary.
        """
        if bet_types is None:
            bet_types = ["単勝"]

        all_bets = []
        bankroll = self.initial_bankroll
        race_ids = predictions_df["race_id"].unique()

        for race_id in sorted(race_ids):
            race_preds = predictions_df[predictions_df["race_id"] == race_id].copy()
            race_payouts = payouts_df[payouts_df["race_id"] == race_id]
            race_date = race_preds["race_date"].iloc[0] if "race_date" in race_preds else None

            if bankroll <= 0:
                break

            for bet_type in bet_types:
                bets = self._simulate_bets(
                    race_preds, race_payouts, strategy, bet_type, bankroll, race_id, race_date
                )
                for bet in bets:
                    bankroll += bet["payout"] - bet["bet_amount"]
                    bet["bankroll_after"] = bankroll
                all_bets.extend(bets)

        bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

        return {
            "bets_df": bets_df,
            "metrics": compute_metrics(bets_df),
            "monthly_roi": compute_monthly_roi(bets_df),
            "bet_type_summary": compute_bet_type_summary(bets_df),
            "final_bankroll": bankroll,
        }

    def _simulate_bets(
        self,
        race_preds: pd.DataFrame,
        race_payouts: pd.DataFrame,
        strategy: BettingStrategy,
        bet_type: str,
        bankroll: float,
        race_id: str,
        race_date,
    ) -> list[dict]:
        """Simulate bets for a single race and bet type."""

        if bet_type == "単勝":
            return self._simulate_win(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "複勝":
            return self._simulate_place(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "馬連":
            return self._simulate_quinella(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "馬単":
            return self._simulate_exacta(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "三連複":
            return self._simulate_trio(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "三連単":
            return self._simulate_trifecta(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        elif bet_type == "ワイド":
            return self._simulate_wide(
                race_preds, race_payouts, strategy, bankroll, race_id, race_date
            )
        return []

    def _simulate_win(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """単勝: Bet on a horse to win."""
        bets = []
        for _, row in preds.iterrows():
            prob = row["predicted_prob"]
            odds = row.get("odds", 0)
            if not odds or odds <= 0:
                continue

            if strategy.should_bet(prob, odds):
                amount = strategy.bet_size(prob, odds, bankroll)
                if amount <= 0 or amount > bankroll:
                    continue

                horse_num = row.get("horse_number", "")
                payout_row = payouts[
                    (payouts["bet_type"] == "単勝")
                    & (payouts["combination"] == str(int(horse_num)))
                ]
                actual_payout = (
                    payout_row["payout"].iloc[0] / 100 * amount
                    if not payout_row.empty
                    else 0
                )

                bets.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "bet_type": "単勝",
                        "combination": str(int(horse_num)),
                        "predicted_prob": prob,
                        "odds": odds,
                        "expected_value": prob * odds,
                        "bet_amount": amount,
                        "payout": actual_payout,
                    }
                )
        return bets

    def _simulate_place(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """複勝: Bet on a horse to finish top 3."""
        bets = []
        for _, row in preds.iterrows():
            # Use place probability (roughly 3x win probability, capped)
            prob = min(row["predicted_prob"] * 3, 0.95)
            odds = row.get("odds", 0)
            place_odds = odds * 0.3 if odds else 0  # Rough estimate

            if not place_odds or place_odds <= 0:
                continue

            if strategy.should_bet(prob, place_odds):
                amount = strategy.bet_size(prob, place_odds, bankroll)
                if amount <= 0 or amount > bankroll:
                    continue

                horse_num = str(int(row.get("horse_number", 0)))
                payout_row = payouts[
                    (payouts["bet_type"] == "複勝") & (payouts["combination"] == horse_num)
                ]
                actual_payout = (
                    payout_row["payout"].iloc[0] / 100 * amount
                    if not payout_row.empty
                    else 0
                )

                bets.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "bet_type": "複勝",
                        "combination": horse_num,
                        "predicted_prob": prob,
                        "odds": place_odds,
                        "expected_value": prob * place_odds,
                        "bet_amount": amount,
                        "payout": actual_payout,
                    }
                )
        return bets

    def _simulate_quinella(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """馬連: Bet on two horses to finish top 2 (any order)."""
        bets = []
        probs = preds["predicted_prob"].values
        horse_nums = preds["horse_number"].values
        n = len(probs)

        harville = compute_harville_probs(probs)

        for i in range(n):
            for j in range(i + 1, n):
                pair_prob = harville.get(("quinella", i, j), 0)
                if pair_prob <= 0.01:
                    continue

                combo = f"{int(min(horse_nums[i], horse_nums[j]))}-{int(max(horse_nums[i], horse_nums[j]))}"
                payout_row = payouts[
                    (payouts["bet_type"] == "馬連") & (payouts["combination"] == combo)
                ]

                # Estimate odds from payout
                if not payout_row.empty:
                    combo_odds = payout_row["payout"].iloc[0] / 100
                else:
                    combo_odds = 1.0 / pair_prob * 0.75 if pair_prob > 0 else 0

                if strategy.should_bet(pair_prob, combo_odds):
                    amount = strategy.bet_size(pair_prob, combo_odds, bankroll)
                    if amount <= 0 or amount > bankroll:
                        continue

                    actual_payout = (
                        payout_row["payout"].iloc[0] / 100 * amount
                        if not payout_row.empty
                        else 0
                    )

                    bets.append(
                        {
                            "race_id": race_id,
                            "race_date": race_date,
                            "bet_type": "馬連",
                            "combination": combo,
                            "predicted_prob": pair_prob,
                            "odds": combo_odds,
                            "expected_value": pair_prob * combo_odds,
                            "bet_amount": amount,
                            "payout": actual_payout,
                        }
                    )
        return bets

    def _simulate_exacta(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """馬単: Bet on 1st and 2nd in exact order."""
        bets = []
        probs = preds["predicted_prob"].values
        horse_nums = preds["horse_number"].values
        n = len(probs)

        harville = compute_harville_probs(probs)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pair_prob = harville.get(("exacta", i, j), 0)
                if pair_prob <= 0.005:
                    continue

                combo = f"{int(horse_nums[i])}-{int(horse_nums[j])}"
                payout_row = payouts[
                    (payouts["bet_type"] == "馬単") & (payouts["combination"] == combo)
                ]

                if not payout_row.empty:
                    combo_odds = payout_row["payout"].iloc[0] / 100
                else:
                    combo_odds = 1.0 / pair_prob * 0.75 if pair_prob > 0 else 0

                if strategy.should_bet(pair_prob, combo_odds):
                    amount = strategy.bet_size(pair_prob, combo_odds, bankroll)
                    if amount <= 0 or amount > bankroll:
                        continue

                    actual_payout = (
                        payout_row["payout"].iloc[0] / 100 * amount
                        if not payout_row.empty
                        else 0
                    )

                    bets.append(
                        {
                            "race_id": race_id,
                            "race_date": race_date,
                            "bet_type": "馬単",
                            "combination": combo,
                            "predicted_prob": pair_prob,
                            "odds": combo_odds,
                            "expected_value": pair_prob * combo_odds,
                            "bet_amount": amount,
                            "payout": actual_payout,
                        }
                    )
        return bets

    def _simulate_trio(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """三連複: Top 3 in any order."""
        bets = []
        probs = preds["predicted_prob"].values
        horse_nums = preds["horse_number"].values
        n = len(probs)

        harville = compute_harville_probs(probs)

        # Only consider top combinations to limit computation
        top_indices = np.argsort(probs)[::-1][:8]

        for idx_i, i in enumerate(top_indices):
            for idx_j, j in enumerate(top_indices):
                if idx_j <= idx_i:
                    continue
                for idx_k, k in enumerate(top_indices):
                    if idx_k <= idx_j:
                        continue
                    sorted_ijk = tuple(sorted([i, j, k]))
                    trio_prob = harville.get(("trio", *sorted_ijk), 0)
                    if trio_prob <= 0.003:
                        continue

                    nums = sorted([int(horse_nums[i]), int(horse_nums[j]), int(horse_nums[k])])
                    combo = f"{nums[0]}-{nums[1]}-{nums[2]}"
                    payout_row = payouts[
                        (payouts["bet_type"] == "三連複") & (payouts["combination"] == combo)
                    ]

                    if not payout_row.empty:
                        combo_odds = payout_row["payout"].iloc[0] / 100
                    else:
                        combo_odds = 1.0 / trio_prob * 0.75 if trio_prob > 0 else 0

                    if strategy.should_bet(trio_prob, combo_odds):
                        amount = strategy.bet_size(trio_prob, combo_odds, bankroll)
                        if amount <= 0 or amount > bankroll:
                            continue

                        actual_payout = (
                            payout_row["payout"].iloc[0] / 100 * amount
                            if not payout_row.empty
                            else 0
                        )

                        bets.append(
                            {
                                "race_id": race_id,
                                "race_date": race_date,
                                "bet_type": "三連複",
                                "combination": combo,
                                "predicted_prob": trio_prob,
                                "odds": combo_odds,
                                "expected_value": trio_prob * combo_odds,
                                "bet_amount": amount,
                                "payout": actual_payout,
                            }
                        )
        return bets

    def _simulate_trifecta(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """三連単: Top 3 in exact order."""
        bets = []
        probs = preds["predicted_prob"].values
        horse_nums = preds["horse_number"].values
        n = len(probs)

        harville = compute_harville_probs(probs)
        top_indices = np.argsort(probs)[::-1][:6]

        for i in top_indices:
            for j in top_indices:
                if j == i:
                    continue
                for k in top_indices:
                    if k == i or k == j:
                        continue
                    tri_prob = harville.get(("trifecta", i, j, k), 0)
                    if tri_prob <= 0.001:
                        continue

                    combo = f"{int(horse_nums[i])}-{int(horse_nums[j])}-{int(horse_nums[k])}"
                    payout_row = payouts[
                        (payouts["bet_type"] == "三連単") & (payouts["combination"] == combo)
                    ]

                    if not payout_row.empty:
                        combo_odds = payout_row["payout"].iloc[0] / 100
                    else:
                        combo_odds = 1.0 / tri_prob * 0.75 if tri_prob > 0 else 0

                    if strategy.should_bet(tri_prob, combo_odds):
                        amount = strategy.bet_size(tri_prob, combo_odds, bankroll)
                        if amount <= 0 or amount > bankroll:
                            continue

                        actual_payout = (
                            payout_row["payout"].iloc[0] / 100 * amount
                            if not payout_row.empty
                            else 0
                        )

                        bets.append(
                            {
                                "race_id": race_id,
                                "race_date": race_date,
                                "bet_type": "三連単",
                                "combination": combo,
                                "predicted_prob": tri_prob,
                                "odds": combo_odds,
                                "expected_value": tri_prob * combo_odds,
                                "bet_amount": amount,
                                "payout": actual_payout,
                            }
                        )
        return bets

    def _simulate_wide(self, preds, payouts, strategy, bankroll, race_id, race_date):
        """ワイド: Two horses both finish in top 3."""
        bets = []
        probs = preds["predicted_prob"].values
        horse_nums = preds["horse_number"].values
        n = len(probs)

        # Approximate wide probability using place probabilities
        place_probs = np.minimum(probs * 3, 0.95)

        top_indices = np.argsort(probs)[::-1][:6]
        for idx_i, i in enumerate(top_indices):
            for idx_j, j in enumerate(top_indices):
                if idx_j <= idx_i:
                    continue
                # P(both in top 3) approximation
                wide_prob = place_probs[i] * place_probs[j] * 0.5

                if wide_prob <= 0.01:
                    continue

                nums = sorted([int(horse_nums[i]), int(horse_nums[j])])
                combo = f"{nums[0]}-{nums[1]}"
                payout_row = payouts[
                    (payouts["bet_type"] == "ワイド") & (payouts["combination"] == combo)
                ]

                if not payout_row.empty:
                    combo_odds = payout_row["payout"].iloc[0] / 100
                else:
                    combo_odds = 1.0 / wide_prob * 0.75 if wide_prob > 0 else 0

                if strategy.should_bet(wide_prob, combo_odds):
                    amount = strategy.bet_size(wide_prob, combo_odds, bankroll)
                    if amount <= 0 or amount > bankroll:
                        continue

                    actual_payout = (
                        payout_row["payout"].iloc[0] / 100 * amount
                        if not payout_row.empty
                        else 0
                    )

                    bets.append(
                        {
                            "race_id": race_id,
                            "race_date": race_date,
                            "bet_type": "ワイド",
                            "combination": combo,
                            "predicted_prob": wide_prob,
                            "odds": combo_odds,
                            "expected_value": wide_prob * combo_odds,
                            "bet_amount": amount,
                            "payout": actual_payout,
                        }
                    )
        return bets


class BaselineEngine:
    """Baseline strategies for comparison."""

    @staticmethod
    def favorite_only(predictions_df: pd.DataFrame, payouts_df: pd.DataFrame) -> pd.DataFrame:
        """Always bet on the 1st favorite (lowest odds). Fixed 100 yen bet."""
        bets = []
        for race_id in predictions_df["race_id"].unique():
            race = predictions_df[predictions_df["race_id"] == race_id]
            favorite = race.loc[race["odds"].idxmin()] if race["odds"].notna().any() else None
            if favorite is None:
                continue

            horse_num = str(int(favorite.get("horse_number", 0)))
            payout_row = payouts_df[
                (payouts_df["race_id"] == race_id)
                & (payouts_df["bet_type"] == "単勝")
                & (payouts_df["combination"] == horse_num)
            ]
            actual_payout = payout_row["payout"].iloc[0] if not payout_row.empty else 0

            bets.append(
                {
                    "race_id": race_id,
                    "race_date": favorite.get("race_date"),
                    "bet_type": "単勝",
                    "combination": horse_num,
                    "predicted_prob": 1.0 / favorite["odds"] if favorite["odds"] > 0 else 0,
                    "odds": favorite["odds"],
                    "expected_value": 1.0,
                    "bet_amount": 100,
                    "payout": actual_payout,
                }
            )

        return pd.DataFrame(bets)
