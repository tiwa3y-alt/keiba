"""Probability calibration for model outputs."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class CalibratedPredictor:
    """Calibrate raw model probabilities using isotonic regression.

    Isotonic regression is preferred over Platt scaling for non-parametric
    calibration, especially when the raw probabilities may not follow a
    sigmoid relationship.
    """

    def __init__(self):
        self._calibrator = IsotonicRegression(
            y_min=0.001, y_max=0.999, out_of_bounds="clip"
        )
        self._is_fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> "CalibratedPredictor":
        """Fit the calibrator on validation data.

        Args:
            predicted_probs: Raw model probabilities.
            actual_outcomes: Binary outcomes (1 = positive, 0 = negative).

        Returns:
            self
        """
        self._calibrator.fit(predicted_probs, actual_outcomes)
        self._is_fitted = True
        return self

    def calibrate(self, predicted_probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities.

        Args:
            predicted_probs: Raw model probabilities.

        Returns:
            Calibrated probabilities.
        """
        if not self._is_fitted:
            return predicted_probs
        return self._calibrator.predict(predicted_probs)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def evaluate_calibration(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Evaluate calibration quality.

    Args:
        predicted_probs: Predicted probabilities.
        actual_outcomes: Binary actual outcomes.
        n_bins: Number of bins for calibration analysis.

    Returns:
        Dict with calibration metrics and bin data.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_data.append(
            {
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "count": int(mask.sum()),
                "mean_predicted": float(predicted_probs[mask].mean()),
                "mean_actual": float(actual_outcomes[mask].mean()),
            }
        )

    # Expected Calibration Error (ECE)
    ece = 0.0
    total = len(predicted_probs)
    for bd in bin_data:
        ece += bd["count"] / total * abs(bd["mean_predicted"] - bd["mean_actual"])

    # Brier score
    brier = float(np.mean((predicted_probs - actual_outcomes) ** 2))

    return {
        "ece": ece,
        "brier_score": brier,
        "bins": bin_data,
    }
