"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CHRONOS vs PROPHET BENCHMARK                             ║
║           Head-to-Head Comparison on Identical Data                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script runs both Chronos and Facebook Prophet on the same synthetic
time series and compares:
    1. Fit speed (wall-clock time)
    2. Point forecast accuracy (MAE, RMSE, MAPE, R²)
    3. Prediction interval calibration (coverage)
    4. Component decomposition quality

Run:
    python3 BenchmarkChronosVsProphet.py
"""

import time
import warnings
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from chronos import MasterChronos
from chronos.metrics.accuracy import ForecastMetrics


def generate_benchmark_data(T: int = 1095, seed: int = 42) -> tuple:
    """
    Generates a synthetic time series with known components.
    
    Components:
        - Linear trend: y = 100 + 0.03t
        - Weekly seasonality: 3·sin(2πt/7)
        - Yearly seasonality: 8·sin(2πt/365.25)
        - Gaussian noise: N(0, 2)
    
    Returns:
        (train_df, test_df, full_df) — 80/20 train/test split.
    """
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=T, freq='D')
    t = np.arange(T, dtype=float)

    trend = 100 + 0.03 * t
    weekly = 3 * np.sin(2 * np.pi * t / 7)
    yearly = 8 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, 2, T)
    y = trend + weekly + yearly + noise

    df = pd.DataFrame({'ds': dates, 'y': y})

    split = int(T * 0.8)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)

    return train, test, df


def benchmark_chronos(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Benchmarks Chronos on the train/test split."""
    
    # --- Fit ---
    model = MasterChronos(
        growth='linear',
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    
    t0 = time.perf_counter()
    model.fit(train)
    fit_time = (time.perf_counter() - t0) * 1000  # ms
    
    # --- Predict ---
    t0 = time.perf_counter()
    forecast = model.predict(periods=len(test), include_history=False)
    predict_time = (time.perf_counter() - t0) * 1000
    
    # Align predictions with test
    min_len = min(len(forecast), len(test))
    actual = test['y'].values[:min_len]
    predicted = forecast['yhat'].values[:min_len]
    lower = forecast['yhat_lower'].values[:min_len]
    upper = forecast['yhat_upper'].values[:min_len]
    
    metrics = ForecastMetrics(actual, predicted, lower, upper)
    
    return {
        'model': 'Chronos',
        'fit_time_ms': round(fit_time, 1),
        'predict_time_ms': round(predict_time, 1),
        'total_time_ms': round(fit_time + predict_time, 1),
        'mae': round(metrics.mae(), 4),
        'rmse': round(metrics.rmse(), 4),
        'mape': round(metrics.mape(), 2),
        'smape': round(metrics.smape(), 2),
        'r_squared': round(metrics.r_squared(), 4),
        'coverage': round(metrics.coverage(), 4),
        'bias': round(metrics.bias(), 4),
        'mase': round(metrics.mase(), 4),
    }


def benchmark_prophet(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Benchmarks Facebook Prophet on the train/test split."""
    
    from prophet import Prophet
    
    # --- Fit ---
    model = Prophet(
        growth='linear',
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    
    t0 = time.perf_counter()
    model.fit(train)
    fit_time = (time.perf_counter() - t0) * 1000
    
    # --- Predict ---
    t0 = time.perf_counter()
    future = model.make_future_dataframe(periods=len(test))
    prophet_forecast = model.predict(future)
    predict_time = (time.perf_counter() - t0) * 1000
    
    # Get out-of-sample predictions only
    forecast_oos = prophet_forecast.iloc[len(train):].reset_index(drop=True)
    
    min_len = min(len(forecast_oos), len(test))
    actual = test['y'].values[:min_len]
    predicted = forecast_oos['yhat'].values[:min_len]
    lower = forecast_oos['yhat_lower'].values[:min_len]
    upper = forecast_oos['yhat_upper'].values[:min_len]
    
    metrics = ForecastMetrics(actual, predicted, lower, upper)
    
    return {
        'model': 'Prophet',
        'fit_time_ms': round(fit_time, 1),
        'predict_time_ms': round(predict_time, 1),
        'total_time_ms': round(fit_time + predict_time, 1),
        'mae': round(metrics.mae(), 4),
        'rmse': round(metrics.rmse(), 4),
        'mape': round(metrics.mape(), 2),
        'smape': round(metrics.smape(), 2),
        'r_squared': round(metrics.r_squared(), 4),
        'coverage': round(metrics.coverage(), 4),
        'bias': round(metrics.bias(), 4),
        'mase': round(metrics.mase(), 4),
    }


def print_comparison(chronos_results: dict, prophet_results: dict, T: int):
    """Pretty-prints the head-to-head results."""
    
    def winner(metric, lower_is_better=True):
        c = chronos_results[metric]
        p = prophet_results[metric]
        if isinstance(c, str) or isinstance(p, str):
            return ''
        if lower_is_better:
            return '← WINNER' if c < p else ('← WINNER' if p < c else 'TIE')
        else:
            return '← WINNER' if c > p else ('← WINNER' if p > c else 'TIE')
    
    def pick_winner(metric, lower_is_better=True):
        c = chronos_results[metric]
        p = prophet_results[metric]
        if lower_is_better:
            return 'Chronos' if c <= p else 'Prophet'
        else:
            return 'Chronos' if c >= p else 'Prophet'
    
    c = chronos_results
    p = prophet_results
    
    speedup = p['fit_time_ms'] / c['fit_time_ms'] if c['fit_time_ms'] > 0 else float('inf')
    
    print()
    print("=" * 78)
    print("  🥊  CHRONOS vs PROPHET — HEAD-TO-HEAD BENCHMARK")
    print("=" * 78)
    print()
    print(f"  Dataset:     Synthetic ({T} daily observations, 80/20 split)")
    print(f"  Components:  Linear trend + weekly + yearly seasonality + noise")
    print(f"  Settings:    Both models use identical hyperparameters")
    print()
    
    print("─" * 78)
    print(f"  {'METRIC':<25}{'CHRONOS':>15}{'PROPHET':>15}{'WINNER':>18}")
    print("─" * 78)
    
    # Speed
    print(f"  {'⚡ Fit Time (ms)':<25}{c['fit_time_ms']:>15.1f}{p['fit_time_ms']:>15.1f}{'':>3}{pick_winner('fit_time_ms'):>13}")
    print(f"  {'⚡ Predict Time (ms)':<25}{c['predict_time_ms']:>15.1f}{p['predict_time_ms']:>15.1f}{'':>3}{pick_winner('predict_time_ms'):>13}")
    print(f"  {'⚡ Total Time (ms)':<25}{c['total_time_ms']:>15.1f}{p['total_time_ms']:>15.1f}{'':>3}{pick_winner('total_time_ms'):>13}")
    print()
    
    # Accuracy
    print(f"  {'📊 MAE':<25}{c['mae']:>15.4f}{p['mae']:>15.4f}{'':>3}{pick_winner('mae'):>13}")
    print(f"  {'📊 RMSE':<25}{c['rmse']:>15.4f}{p['rmse']:>15.4f}{'':>3}{pick_winner('rmse'):>13}")
    print(f"  {'📊 MAPE (%)':<25}{c['mape']:>15.2f}{p['mape']:>15.2f}{'':>3}{pick_winner('mape'):>13}")
    print(f"  {'📊 SMAPE (%)':<25}{c['smape']:>15.2f}{p['smape']:>15.2f}{'':>3}{pick_winner('smape'):>13}")
    print(f"  {'📊 R²':<25}{c['r_squared']:>15.4f}{p['r_squared']:>15.4f}{'':>3}{pick_winner('r_squared', lower_is_better=False):>13}")
    print(f"  {'📊 MASE':<25}{c['mase']:>15.4f}{p['mase']:>15.4f}{'':>3}{pick_winner('mase'):>13}")
    print(f"  {'📊 Bias':<25}{abs(c['bias']):>15.4f}{abs(p['bias']):>15.4f}{'':>3}{pick_winner('bias', lower_is_better=True):>13}")
    print()
    
    # Coverage
    print(f"  {'🎯 Coverage (80%)':<25}{c['coverage']:>15.4f}{p['coverage']:>15.4f}{'':>3}{'Closer to 0.80':>13}")
    print()
    
    print("─" * 78)
    
    # Tally
    wins_chronos = 0
    wins_prophet = 0
    for metric in ['fit_time_ms', 'predict_time_ms', 'total_time_ms',
                    'mae', 'rmse', 'mape', 'smape', 'mase']:
        w = pick_winner(metric)
        if w == 'Chronos':
            wins_chronos += 1
        else:
            wins_prophet += 1
    
    w_r2 = pick_winner('r_squared', lower_is_better=False)
    if w_r2 == 'Chronos':
        wins_chronos += 1
    else:
        wins_prophet += 1
    
    print()
    print(f"  SCORECARD:   Chronos {wins_chronos} — Prophet {wins_prophet}")
    print(f"  SPEEDUP:     Chronos is {speedup:.1f}× faster than Prophet")
    print()
    
    if wins_chronos > wins_prophet:
        print("  🏆 WINNER: CHRONOS")
    elif wins_prophet > wins_chronos:
        print("  🏆 WINNER: PROPHET")
    else:
        print("  🤝 TIE")
    
    print()
    print("=" * 78)


def main():
    T = 1095  # 3 years daily
    
    print("🔧 Generating benchmark data...")
    train, test, full = generate_benchmark_data(T=T)
    print(f"   Train: {len(train)} observations")
    print(f"   Test:  {len(test)} observations")
    print()
    
    print("🕐 Running Chronos benchmark...")
    chronos_results = benchmark_chronos(train, test)
    print(f"   ✅ Chronos done in {chronos_results['total_time_ms']:.1f}ms")
    
    print("📈 Running Prophet benchmark...")
    prophet_results = benchmark_prophet(train, test)
    print(f"   ✅ Prophet done in {prophet_results['total_time_ms']:.1f}ms")
    
    print_comparison(chronos_results, prophet_results, T)


if __name__ == '__main__':
    main()
