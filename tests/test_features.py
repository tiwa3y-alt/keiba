"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from keiba.features.horse import _distance_category, _parse_age, compute_horse_features
from keiba.features.market import compute_market_features


def _make_results_df():
    """Create sample results DataFrame for testing."""
    return pd.DataFrame(
        {
            "race_id": ["R001"] * 3 + ["R002"] * 3,
            "horse_id": ["H1", "H2", "H3", "H1", "H2", "H4"],
            "finish_order": [1, 2, 3, 2, 1, 3],
            "race_date": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-02-01"] * 3
            ),
            "distance": [2000] * 6,
            "course_type": ["芝"] * 6,
            "venue": ["東京"] * 6,
            "last_3f": [33.5, 34.0, 34.5, 33.8, 33.2, 35.0],
            "horse_weight": [480, 460, 500, 482, 458, 510],
            "prize_1st": [10000] * 6,
            "grade": ["G1"] * 6,
            "odds": [3.0, 5.0, 10.0, 4.0, 2.5, 15.0],
            "popularity": [1, 2, 3, 2, 1, 3],
            "sex_age": ["牡4", "牝3", "牡5", "牡4", "牝3", "セ6"],
            "frame_number": [1, 2, 3, 1, 2, 3],
            "horse_number": [1, 2, 3, 1, 2, 3],
            "passing_order": ["3-3-2-1", "2-2-3-2", "1-1-1-3",
                              "2-2-1-2", "3-3-3-1", "1-1-2-3"],
            "jockey_id": ["J1", "J2", "J3", "J1", "J2", "J3"],
            "trainer_id": ["T1", "T2", "T3", "T1", "T2", "T3"],
            "track_condition": ["良"] * 6,
            "weight_carried": [57.0, 55.0, 57.0, 57.0, 55.0, 57.0],
        }
    )


class TestDistanceCategory:
    def test_sprint(self):
        assert _distance_category(1200) == "sprint"

    def test_mile(self):
        assert _distance_category(1600) == "mile"

    def test_intermediate(self):
        assert _distance_category(2000) == "intermediate"

    def test_long(self):
        assert _distance_category(2400) == "long"


class TestParseAge:
    def test_normal(self):
        assert _parse_age("牡4") == 4

    def test_female(self):
        assert _parse_age("牝3") == 3

    def test_empty(self):
        assert _parse_age("") is None


class TestHorseFeatures:
    def test_basic_features(self):
        df = _make_results_df()
        features = compute_horse_features(df, "R002")
        assert len(features) == 3  # 3 horses in R002
        h1 = features[features["horse_id"] == "H1"].iloc[0]
        assert h1["run_count"] == 1  # 1 prior race
        assert h1["win_rate_last5"] == 1.0  # Won only prior race

    def test_new_horse(self):
        df = _make_results_df()
        features = compute_horse_features(df, "R002")
        h4 = features[features["horse_id"] == "H4"].iloc[0]
        assert h4["run_count"] == 0
        assert np.isnan(h4["win_rate_last5"])


class TestMarketFeatures:
    def test_implied_prob_sums_to_one(self):
        df = _make_results_df()
        features = compute_market_features(df, "R001")
        total = features["odds_implied_prob"].sum()
        assert abs(total - 1.0) < 0.01

    def test_favorite_flag(self):
        df = _make_results_df()
        features = compute_market_features(df, "R001")
        fav = features[features["is_favorite"] == 1]
        assert len(fav) == 1
