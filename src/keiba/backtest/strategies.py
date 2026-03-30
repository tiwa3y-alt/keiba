"""Betting strategies for backtesting."""

from abc import ABC, abstractmethod

import numpy as np


class BettingStrategy(ABC):
    """Base class for betting strategies."""

    @abstractmethod
    def should_bet(self, predicted_prob: float, odds: float) -> bool:
        """Decide whether to place a bet."""
        ...

    @abstractmethod
    def bet_size(self, predicted_prob: float, odds: float, bankroll: float) -> float:
        """Calculate bet amount."""
        ...


class ValueBetStrategy(BettingStrategy):
    """Bet when expected value exceeds a threshold.

    EV = predicted_prob * odds
    Bet if EV > threshold (default 1.0 = breakeven).
    """

    def __init__(self, ev_threshold: float = 1.2, fixed_bet: float = 100):
        self.ev_threshold = ev_threshold
        self.fixed_bet = fixed_bet

    def should_bet(self, predicted_prob: float, odds: float) -> bool:
        return predicted_prob * odds > self.ev_threshold

    def bet_size(self, predicted_prob: float, odds: float, bankroll: float) -> float:
        return self.fixed_bet


class KellyCriterionStrategy(BettingStrategy):
    """Kelly criterion for optimal bet sizing.

    f* = (p * b - q) / b
    where p = win prob, q = 1 - p, b = odds - 1 (net odds)
    """

    def __init__(self, min_edge: float = 0.05, max_fraction: float = 0.25):
        self.min_edge = min_edge
        self.max_fraction = max_fraction

    def should_bet(self, predicted_prob: float, odds: float) -> bool:
        kelly = self._kelly_fraction(predicted_prob, odds)
        return kelly > 0

    def bet_size(self, predicted_prob: float, odds: float, bankroll: float) -> float:
        kelly = self._kelly_fraction(predicted_prob, odds)
        fraction = min(kelly, self.max_fraction)
        return bankroll * fraction

    def _kelly_fraction(self, prob: float, odds: float) -> float:
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0
        b = odds - 1  # net odds
        q = 1 - prob
        f = (prob * b - q) / b
        return max(0, f - self.min_edge)


class FractionalKellyStrategy(KellyCriterionStrategy):
    """Conservative Kelly: use a fraction of full Kelly.

    Reduces variance at the cost of lower expected growth.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        min_edge: float = 0.05,
        max_fraction: float = 0.10,
    ):
        super().__init__(min_edge=min_edge, max_fraction=max_fraction)
        self.kelly_fraction = fraction

    def bet_size(self, predicted_prob: float, odds: float, bankroll: float) -> float:
        kelly = self._kelly_fraction(predicted_prob, odds)
        fraction = min(kelly * self.kelly_fraction, self.max_fraction)
        return bankroll * fraction


class TopNStrategy(BettingStrategy):
    """Bet on the top N horses by predicted probability.

    This strategy is used at the race level, not per-horse.
    The should_bet/bet_size methods are for compatibility;
    use select_bets() for actual race-level selection.
    """

    def __init__(self, top_n: int = 3, fixed_bet: float = 100):
        self.top_n = top_n
        self.fixed_bet = fixed_bet

    def should_bet(self, predicted_prob: float, odds: float) -> bool:
        return True  # Selection happens at race level

    def bet_size(self, predicted_prob: float, odds: float, bankroll: float) -> float:
        return self.fixed_bet

    def select_bets(self, race_predictions: list[dict]) -> list[dict]:
        """Select top N horses from a race to bet on.

        Args:
            race_predictions: List of dicts with predicted_prob, odds, horse_number.

        Returns:
            Selected bets (top N by predicted_prob).
        """
        sorted_preds = sorted(race_predictions, key=lambda x: x["predicted_prob"], reverse=True)
        return sorted_preds[: self.top_n]
