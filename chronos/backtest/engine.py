"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CHRONOS CROSS-VALIDATOR                                ║
║           Walk-Forward, Expanding & Sliding Window Backtesting             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Time series cross-validation strategies for evaluating forecast accuracy
without data leakage. Unlike standard k-fold CV which violates temporal
ordering, these methods respect the arrow of time.

Strategies:
    ┌───────────────┬──────────────────────────────────────────────────┐
    │ Strategy      │ Description                                      │
    ├───────────────┼──────────────────────────────────────────────────┤
    │ Walk-Forward  │ Fixed training window, slides forward            │
    │ Expanding     │ Growing training window, slides forward          │
    │ Sliding       │ Fixed train + test windows slide together        │
    └───────────────┴──────────────────────────────────────────────────┘

Walk-Forward (default):
    ┌────────────────────────────────────────────────────────────────┐
    │ [===TRAIN===]--[TEST]                                          │
    │        [===TRAIN===]--[TEST]                                   │
    │               [===TRAIN===]--[TEST]                            │
    └────────────────────────────────────────────────────────────────┘

Expanding:
    ┌────────────────────────────────────────────────────────────────┐
    │ [===TRAIN===]--[TEST]                                          │
    │ [======TRAIN======]--[TEST]                                    │
    │ [=========TRAIN=========]--[TEST]                              │
    └────────────────────────────────────────────────────────────────┘

References:
    [1] Tashman, L. J. (2000). "Out-of-sample tests of forecasting
        accuracy." International Journal of Forecasting, 16(4).
    [2] Hyndman, R. J. & Athanasopoulos, G. (2021). "Forecasting:
        Principles and Practice." Chapter 5.

Complexity:
    O(K · F(T_train)) where K = number of folds, F = fit complexity.
"""

import numpy as np
import pandas as pd
from typing import Optional


class CrossValidator:
    """
    Time series cross-validation engine.

    Usage:
        >>> from chronos import MasterChronos
        >>> from chronos.backtest import CrossValidator
        >>>
        >>> model = MasterChronos(growth='linear')
        >>> cv = CrossValidator(
        ...     model=model,
        ...     initial=730,    # 2 years training
        ...     horizon=30,     # 30-day forecast
        ...     period=90,      # shift 90 days between folds
        ...     strategy='expanding',
        ... )
        >>> results = cv.run(df)
        >>> print(results.summary())
    """

    _VALID_STRATEGIES = ('walk_forward', 'expanding', 'sliding')

    def __init__(
        self,
        model,
        initial: int = 730,
        horizon: int = 30,
        period: int = 90,
        strategy: str = 'expanding',
    ):
        """
        Configures the cross-validation engine.

        Args:
            model: A MasterChronos instance (will be re-fit each fold).
            initial: Number of days in the initial training window.
            horizon: Number of days to forecast in each fold.
            period: Number of days to shift between folds.
            strategy: 'walk_forward', 'expanding', or 'sliding'.
        """
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: '{strategy}'. "
                f"Must be one of {self._VALID_STRATEGIES}."
            )

        if initial < 2:
            raise ValueError(f"initial must be >= 2, got {initial}.")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}.")

        self.model = model
        self.initial = initial
        self.horizon = horizon
        self.period = period
        self.strategy = strategy

    def run(self, df: pd.DataFrame) -> 'CVResult':
        """
        Runs the cross-validation.

        For each fold:
            1. Split data into train/test based on cutoff date
            2. Fit the model on training data
            3. Predict the horizon
            4. Record actual vs predicted

        Args:
            df: DataFrame with 'ds' and 'y' columns.

        Returns:
            CVResult with forecast, actual, and metrics per fold.
        """
        df = df.copy().sort_values('ds').reset_index(drop=True)
        df['ds'] = pd.to_datetime(df['ds'])

        start_date = df['ds'].min()
        end_date = df['ds'].max()
        total_days = (end_date - start_date).days

        if total_days < self.initial + self.horizon:
            raise ValueError(
                f"Not enough data for cross-validation. "
                f"Need at least {self.initial + self.horizon} days, "
                f"have {total_days}."
            )

        # --- Generate cutoff dates ---
        cutoffs = []
        cutoff = start_date + pd.Timedelta(days=self.initial)
        while cutoff + pd.Timedelta(days=self.horizon) <= end_date:
            cutoffs.append(cutoff)
            cutoff += pd.Timedelta(days=self.period)

        if len(cutoffs) == 0:
            raise ValueError(
                "No valid cutoff dates. Try reducing 'initial' or 'period'."
            )

        # --- Run each fold ---
        all_results = []

        for fold_idx, cutoff in enumerate(cutoffs):
            # Train/test split
            if self.strategy == 'expanding':
                train = df[df['ds'] < cutoff]
            elif self.strategy == 'walk_forward':
                train_start = cutoff - pd.Timedelta(days=self.initial)
                train = df[(df['ds'] >= train_start) & (df['ds'] < cutoff)]
            elif self.strategy == 'sliding':
                train_start = cutoff - pd.Timedelta(days=self.initial)
                train = df[(df['ds'] >= train_start) & (df['ds'] < cutoff)]

            test_end = cutoff + pd.Timedelta(days=self.horizon)
            test = df[(df['ds'] >= cutoff) & (df['ds'] < test_end)]

            if len(train) < 2 or len(test) == 0:
                continue

            # Fit and predict
            try:
                # Create a fresh model with same params
                import copy
                fold_model = copy.deepcopy(self.model)
                fold_model.fit(train)

                forecast = fold_model.predict(periods=len(test),
                                              include_history=False)

                # Align forecast with test
                min_len = min(len(test), len(forecast))
                test_aligned = test.iloc[:min_len].reset_index(drop=True)
                forecast_aligned = forecast.iloc[:min_len].reset_index(drop=True)

                for i in range(min_len):
                    all_results.append({
                        'fold': fold_idx,
                        'cutoff': cutoff,
                        'ds': test_aligned['ds'].iloc[i],
                        'actual': test_aligned['y'].iloc[i],
                        'predicted': forecast_aligned['yhat'].iloc[i],
                        'lower': forecast_aligned['yhat_lower'].iloc[i],
                        'upper': forecast_aligned['yhat_upper'].iloc[i],
                        'horizon_days': i + 1,
                    })

            except Exception as e:
                # Skip failed folds but log
                all_results.append({
                    'fold': fold_idx,
                    'cutoff': cutoff,
                    'ds': cutoff,
                    'actual': float('nan'),
                    'predicted': float('nan'),
                    'lower': float('nan'),
                    'upper': float('nan'),
                    'horizon_days': 0,
                    'error': str(e),
                })

        results_df = pd.DataFrame(all_results)
        return CVResult(results_df, self.strategy)


class CVResult:
    """
    Cross-validation results container.

    Provides aggregated metrics and per-fold analysis.
    """

    def __init__(self, results: pd.DataFrame, strategy: str):
        self.results = results
        self.strategy = strategy

    def accuracy(self) -> dict:
        """
        Computes aggregated accuracy metrics across all folds.

        Returns:
            dict with MAE, RMSE, MAPE, SMAPE, Coverage.
        """
        valid = self.results.dropna(subset=['actual', 'predicted'])
        if len(valid) == 0:
            return {'mae': float('nan'), 'rmse': float('nan')}

        actual = valid['actual'].values
        predicted = valid['predicted'].values
        errors = actual - predicted

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        mask = np.abs(actual) > 1e-10
        if mask.any():
            mape = float(100 * np.mean(
                np.abs(errors[mask]) / np.abs(actual[mask])
            ))
        else:
            mape = float('inf')

        # Coverage
        if 'lower' in valid.columns and 'upper' in valid.columns:
            lower = valid['lower'].values
            upper = valid['upper'].values
            coverage = float(np.mean(
                (actual >= lower) & (actual <= upper)
            ))
        else:
            coverage = float('nan')

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage': coverage,
            'n_folds': int(valid['fold'].nunique()),
            'n_predictions': len(valid),
        }

    def summary(self) -> str:
        """Generates a formatted cross-validation summary."""
        acc = self.accuracy()
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            CHRONOS CROSS-VALIDATION RESULTS             ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Strategy:         {self.strategy}",
            f"  Folds:            {acc.get('n_folds', 0)}",
            f"  Predictions:      {acc.get('n_predictions', 0)}",
            "",
            f"  MAE:              {acc.get('mae', float('nan')):.4f}",
            f"  RMSE:             {acc.get('rmse', float('nan')):.4f}",
            f"  MAPE:             {acc.get('mape', float('nan')):.2f}%",
            f"  Coverage:         {acc.get('coverage', float('nan')):.2%}",
            "",
        ]
        return "\n".join(lines)
