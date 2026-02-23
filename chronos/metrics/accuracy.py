"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      CHRONOS FORECAST METRICS                              ║
║              Institutional-Grade Forecast Accuracy Analytics                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Provides comprehensive forecast accuracy metrics used by institutional
forecasting desks, hedge funds, and research organizations.

Supported Metrics:
    ┌────────────┬──────────────────────────────────────────────────┐
    │ Metric     │ Formula                                          │
    ├────────────┼──────────────────────────────────────────────────┤
    │ MAE        │ (1/n)·Σ|yᵢ − ŷᵢ|                               │
    │ RMSE       │ √((1/n)·Σ(yᵢ − ŷᵢ)²)                          │
    │ MAPE       │ (100/n)·Σ|yᵢ − ŷᵢ|/|yᵢ|                       │
    │ SMAPE      │ (200/n)·Σ|yᵢ − ŷᵢ|/(|yᵢ| + |ŷᵢ|)             │
    │ MASE       │ MAE / MAE_naive                                  │
    │ Coverage   │ (1/n)·Σ𝟙{yᵢ ∈ [ŷₗ, ŷᵤ]}                       │
    │ Winkler    │ Width + (2/α)·penalty_for_misses                 │
    │ R²         │ 1 − RSS/TSS                                      │
    └────────────┴──────────────────────────────────────────────────┘

References:
    [1] Hyndman, R. J. & Koehler, A. B. (2006). "Another look at
        measures of forecast accuracy." International Journal of
        Forecasting, 22(4), 679-688.
    [2] Makridakis, S. et al. (2020). "The M5 accuracy competition."
        International Journal of Forecasting.

Complexity:
    All metrics: O(T) — single pass over observations.
"""

import numpy as np
import pandas as pd
from typing import Optional


class ForecastMetrics:
    """
    Comprehensive forecast accuracy analytics.

    Usage:
        >>> metrics = ForecastMetrics(actual, predicted)
        >>> print(metrics.mae())
        >>> print(metrics.report())
    """

    def __init__(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ):
        """
        Args:
            actual: True values y.
            predicted: Point forecasts ŷ.
            lower: Lower prediction interval bounds (optional).
            upper: Upper prediction interval bounds (optional).
        """
        self._actual = np.asarray(actual, dtype=np.float64)
        self._predicted = np.asarray(predicted, dtype=np.float64)
        self._lower = np.asarray(lower) if lower is not None else None
        self._upper = np.asarray(upper) if upper is not None else None
        self._errors = self._actual - self._predicted

        if len(self._actual) != len(self._predicted):
            raise ValueError(
                f"actual and predicted must have same length. "
                f"Got {len(self._actual)} and {len(self._predicted)}."
            )

    def mae(self) -> float:
        """Mean Absolute Error: (1/n)·Σ|yᵢ − ŷᵢ|"""
        return float(np.mean(np.abs(self._errors)))

    def rmse(self) -> float:
        """Root Mean Squared Error: √((1/n)·Σ(yᵢ − ŷᵢ)²)"""
        return float(np.sqrt(np.mean(self._errors ** 2)))

    def mape(self) -> float:
        """
        Mean Absolute Percentage Error: (100/n)·Σ|yᵢ − ŷᵢ|/|yᵢ|

        Warning: Undefined when actual values are zero.
        Returns inf if any actual value is zero.
        """
        mask = np.abs(self._actual) > 1e-10
        if not mask.any():
            return float('inf')
        return float(100 * np.mean(
            np.abs(self._errors[mask]) / np.abs(self._actual[mask])
        ))

    def smape(self) -> float:
        """
        Symmetric MAPE: (200/n)·Σ|yᵢ − ŷᵢ|/(|yᵢ| + |ŷᵢ|)

        Bounded [0, 200]. Symmetric: penalizes over- and under-
        prediction equally.
        """
        denominator = np.abs(self._actual) + np.abs(self._predicted)
        mask = denominator > 1e-10
        if not mask.any():
            return 0.0
        return float(200 * np.mean(
            np.abs(self._errors[mask]) / denominator[mask]
        ))

    def mase(self, seasonality: int = 1) -> float:
        """
        Mean Absolute Scaled Error.

        MASE = MAE / MAE_naive

        where MAE_naive is the MAE of the seasonal naive forecast
        (y_{t-m}). MASE < 1 means the forecast beats naive.

        Args:
            seasonality: Period of the seasonal naive forecast.
        """
        n = len(self._actual)
        if n <= seasonality:
            return float('inf')

        naive_errors = np.abs(
            self._actual[seasonality:] - self._actual[:-seasonality]
        )
        mae_naive = np.mean(naive_errors)

        if mae_naive < 1e-10:
            return float('inf')

        return self.mae() / mae_naive

    def r_squared(self) -> float:
        """
        R² (Coefficient of Determination): 1 − RSS/TSS

        R² = 1 means perfect prediction.
        R² = 0 means no better than predicting the mean.
        R² < 0 means worse than predicting the mean.
        """
        ss_res = np.sum(self._errors ** 2)
        ss_tot = np.sum((self._actual - np.mean(self._actual)) ** 2)
        if ss_tot < 1e-10:
            return 1.0
        return float(1 - ss_res / ss_tot)

    def coverage(self) -> float:
        """
        Prediction Interval Coverage: fraction of actual values
        falling within [lower, upper].

        Returns:
            Coverage rate in [0, 1].
        """
        if self._lower is None or self._upper is None:
            return float('nan')

        within = (
            (self._actual >= self._lower) &
            (self._actual <= self._upper)
        )
        return float(np.mean(within))

    def winkler_score(self, alpha: float = 0.2) -> float:
        """
        Winkler Score: measures interval sharpness + calibration.

        Score = width + (2/α)·penalty_for_misses

        Lower is better. Rewards narrow intervals that still cover
        the actual values.

        Args:
            alpha: Significance level (1 - interval_width).
        """
        if self._lower is None or self._upper is None:
            return float('nan')

        width = self._upper - self._lower
        n = len(self._actual)
        score = 0.0

        for i in range(n):
            if self._actual[i] < self._lower[i]:
                score += width[i] + (2.0 / alpha) * (
                    self._lower[i] - self._actual[i]
                )
            elif self._actual[i] > self._upper[i]:
                score += width[i] + (2.0 / alpha) * (
                    self._actual[i] - self._upper[i]
                )
            else:
                score += width[i]

        return float(score / n)

    def bias(self) -> float:
        """Mean Error (Bias): (1/n)·Σ(yᵢ − ŷᵢ)"""
        return float(np.mean(self._errors))

    def report(self) -> str:
        """
        Generates a formatted accuracy report.

        Returns:
            Multi-line string with all metrics.
        """
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              CHRONOS FORECAST ACCURACY                  ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Observations:     {len(self._actual)}",
            f"  MAE:              {self.mae():.4f}",
            f"  RMSE:             {self.rmse():.4f}",
            f"  MAPE:             {self.mape():.2f}%",
            f"  SMAPE:            {self.smape():.2f}%",
            f"  MASE:             {self.mase():.4f}",
            f"  R²:               {self.r_squared():.4f}",
            f"  Bias:             {self.bias():.4f}",
        ]

        if self._lower is not None and self._upper is not None:
            lines.extend([
                "",
                f"  Coverage:         {self.coverage():.2%}",
                f"  Winkler Score:    {self.winkler_score():.4f}",
            ])

        lines.append("")
        return "\n".join(lines)
