"""Tests for backtesting engine and strategies."""

import numpy as np
import pandas as pd
import pytest

from keiba.backtest.metrics import compute_metrics, compute_monthly_roi
from keiba.backtest.strategies import (
    FractionalKellyStrategy,
    KellyCriterionStrategy,
    ValueBetStrategy,
)
from keiba.models.predictor import compute_harville_probs


class TestValueBetStrategy:
    def test_should_bet_positive_ev(self):
        strategy = ValueBetStrategy(ev_threshold=1.2)
        assert strategy.should_bet(0.3, 5.0)  # EV = 1.5

    def test_should_not_bet_low_ev(self):
        strategy = ValueBetStrategy(ev_threshold=1.2)
        assert not strategy.should_bet(0.1, 5.0)  # EV = 0.5

    def test_bet_size_fixed(self):
        strategy = ValueBetStrategy(fixed_bet=200)
        assert strategy.bet_size(0.3, 5.0, 10000) == 200


class TestKellyCriterion:
    def test_positive_edge(self):
        strategy = KellyCriterionStrategy(min_edge=0.0)
        # p=0.5, odds=3.0 -> b=2.0, f = (0.5*2 - 0.5)/2 = 0.25
        assert strategy.should_bet(0.5, 3.0)

    def test_negative_edge(self):
        strategy = KellyCriterionStrategy(min_edge=0.0)
        # p=0.1, odds=2.0 -> b=1.0, f = (0.1*1 - 0.9)/1 = -0.8
        assert not strategy.should_bet(0.1, 2.0)

    def test_bet_size_capped(self):
        strategy = KellyCriterionStrategy(min_edge=0.0, max_fraction=0.10)
        size = strategy.bet_size(0.5, 3.0, 100000)
        assert size <= 100000 * 0.10

    def test_fractional_kelly(self):
        full = KellyCriterionStrategy(min_edge=0.0, max_fraction=1.0)
        frac = FractionalKellyStrategy(fraction=0.25, min_edge=0.0, max_fraction=1.0)
        full_size = full.bet_size(0.5, 3.0, 100000)
        frac_size = frac.bet_size(0.5, 3.0, 100000)
        assert frac_size < full_size


class TestHarvilleModel:
    def test_probabilities_sum(self):
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        harville = compute_harville_probs(probs)

        # Exacta probabilities for all orderings of first two should sum to ~1
        exacta_sum = sum(v for k, v in harville.items() if k[0] == "exacta")
        assert abs(exacta_sum - 1.0) < 0.01

    def test_quinella_less_than_exacta(self):
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        harville = compute_harville_probs(probs)

        # P(0,1 quinella) = P(0->1 exacta) + P(1->0 exacta)
        q = harville.get(("quinella", 0, 1), 0)
        e01 = harville.get(("exacta", 0, 1), 0)
        e10 = harville.get(("exacta", 1, 0), 0)
        assert abs(q - (e01 + e10)) < 0.001

    def test_favorite_highest_exacta(self):
        probs = np.array([0.5, 0.3, 0.2])
        harville = compute_harville_probs(probs)
        # Favorite winning should have highest individual exacta prob
        e01 = harville.get(("exacta", 0, 1), 0)
        e10 = harville.get(("exacta", 1, 0), 0)
        assert e01 > e10


class TestMetrics:
    def test_basic_metrics(self):
        bets = pd.DataFrame(
            {
                "bet_amount": [100, 100, 100, 100, 100],
                "payout": [300, 0, 0, 500, 0],
                "race_date": pd.date_range("2024-01-01", periods=5),
                "bet_type": ["単勝"] * 5,
            }
        )
        metrics = compute_metrics(bets)
        assert metrics["total_bets"] == 5
        assert metrics["total_bet_amount"] == 500
        assert metrics["total_payout"] == 800
        assert metrics["roi"] == 800 / 500
        assert metrics["hit_rate"] == 2 / 5

    def test_empty_bets(self):
        bets = pd.DataFrame()
        metrics = compute_metrics(bets)
        assert metrics == {}

    def test_all_losses(self):
        bets = pd.DataFrame(
            {
                "bet_amount": [100, 100, 100],
                "payout": [0, 0, 0],
                "race_date": pd.date_range("2024-01-01", periods=3),
                "bet_type": ["単勝"] * 3,
            }
        )
        metrics = compute_metrics(bets)
        assert metrics["roi"] == 0
        assert metrics["hit_rate"] == 0
        assert metrics["max_lose_streak"] == 3

    def test_monthly_roi(self):
        bets = pd.DataFrame(
            {
                "bet_amount": [100, 100, 100, 100],
                "payout": [200, 0, 300, 0],
                "race_date": ["2024-01-15", "2024-01-20", "2024-02-10", "2024-02-20"],
                "bet_type": ["単勝"] * 4,
            }
        )
        monthly = compute_monthly_roi(bets)
        assert len(monthly) == 2
