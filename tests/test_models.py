"""Tests for model components."""

import numpy as np
import pytest

from keiba.models.calibration import CalibratedPredictor, evaluate_calibration


class TestCalibratedPredictor:
    def test_fit_and_calibrate(self):
        predictor = CalibratedPredictor()
        # Simulate over-confident model
        predicted = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])

        predictor.fit(predicted, actual)
        assert predictor.is_fitted

        calibrated = predictor.calibrate(predicted)
        assert len(calibrated) == len(predicted)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_not_fitted_returns_input(self):
        predictor = CalibratedPredictor()
        probs = np.array([0.5, 0.3])
        result = predictor.calibrate(probs)
        np.testing.assert_array_equal(result, probs)


class TestEvaluateCalibration:
    def test_perfect_calibration(self):
        # Probabilities that match actual outcomes
        predicted = np.array([0.0, 0.0, 1.0, 1.0])
        actual = np.array([0, 0, 1, 1])
        result = evaluate_calibration(predicted, actual, n_bins=2)
        assert result["brier_score"] == 0.0

    def test_returns_ece(self):
        predicted = np.array([0.1, 0.2, 0.8, 0.9])
        actual = np.array([0, 0, 1, 1])
        result = evaluate_calibration(predicted, actual, n_bins=2)
        assert "ece" in result
        assert "brier_score" in result
        assert "bins" in result
        assert result["ece"] >= 0

    def test_worst_calibration(self):
        predicted = np.array([0.0, 0.0, 1.0, 1.0])
        actual = np.array([1, 1, 0, 0])
        result = evaluate_calibration(predicted, actual, n_bins=2)
        assert result["brier_score"] == 1.0
