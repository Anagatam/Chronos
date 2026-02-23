"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CHRONOS DEMO SCRIPT                                ║
║              End-to-End Time Series Forecasting Pipeline                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script demonstrates the full Chronos pipeline:
    1. Generate synthetic data with trend + seasonality + noise
    2. Fit the model
    3. Generate forecasts with confidence intervals
    4. Print model summary and diagnostics
    5. Compute forecast accuracy metrics

Run:
    python DemoChronos.py
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chronos import MasterChronos
from chronos.metrics import ForecastMetrics


def main():
    print("=" * 70)
    print("  🕐 CHRONOS — Institutional-Grade Time Series Forecasting")
    print("=" * 70)
    print()

    # ─── Step 1: Generate synthetic data ───────────────────────────────
    print("📊 Step 1: Generating synthetic time series...")
    np.random.seed(42)
    T = 1095  # 3 years daily
    dates = pd.date_range('2022-01-01', periods=T, freq='D')
    t = np.arange(T, dtype=float)

    # Components
    trend = 100 + 0.03 * t  # Gentle upward trend
    weekly = 3 * np.sin(2 * np.pi * t / 7)  # Weekly cycle
    yearly = 8 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
    noise = np.random.normal(0, 2, T)

    y = trend + weekly + yearly + noise

    df = pd.DataFrame({'ds': dates, 'y': y})
    print(f"   → {T} observations from {dates[0].date()} to {dates[-1].date()}")
    print(f"   → y range: [{y.min():.1f}, {y.max():.1f}]")
    print()

    # ─── Step 2: Fit the model ─────────────────────────────────────────
    print("⚙️  Step 2: Fitting MasterChronos model...")
    model = MasterChronos(
        growth='linear',
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
    )
    model.fit(df)
    print(f"   → Model fitted successfully!")
    print(f"   → Audit trail: {len(model.audit_log)} entries")
    print()

    # ─── Step 3: Generate forecasts ────────────────────────────────────
    print("🔮 Step 3: Generating 90-day forecast...")
    forecast = model.predict(periods=90)
    future_only = forecast[forecast['ds'] > df['ds'].max()]
    print(f"   → Forecast shape: {forecast.shape}")
    print(f"   → Future predictions: {len(future_only)}")
    print()

    # ─── Step 4: Show forecast summary ─────────────────────────────────
    print("📋 Forecast Preview (last 5 rows):")
    print(future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string(index=False))
    print()

    # ─── Step 5: Model Summary ─────────────────────────────────────────
    print(model.summary())

    # ─── Step 6: Diagnostics ───────────────────────────────────────────
    diag = model.diagnostics()
    print("📐 Model Diagnostics:")
    print(f"   AIC:                {diag['model_selection']['aic']:.1f}")
    print(f"   BIC:                {diag['model_selection']['bic']:.1f}")
    print(f"   Durbin-Watson:      {diag['autocorrelation']['durbin_watson']:.3f}")
    print(f"   Interpretation:     {diag['autocorrelation']['interpretation']}")
    print(f"   N Parameters:       {diag['model_selection']['n_parameters']}")
    print()

    # ─── Step 7: In-Sample Metrics ─────────────────────────────────────
    # Align forecast with actuals for in-sample evaluation
    in_sample = forecast[forecast['ds'].isin(df['ds'])].sort_values('ds')
    df_aligned = df[df['ds'].isin(in_sample['ds'])].sort_values('ds')

    if len(in_sample) > 0 and len(df_aligned) > 0:
        metrics = ForecastMetrics(
            actual=df_aligned['y'].values,
            predicted=in_sample['yhat'].values,
            lower=in_sample['yhat_lower'].values,
            upper=in_sample['yhat_upper'].values,
        )
        print(metrics.report())

    # ─── Step 8: Changepoints ──────────────────────────────────────────
    cp = model.changepoints()
    print(f"🔍 Changepoints: {cp['n_significant']} significant changepoints detected")
    print()

    # ─── Step 9: JSON Audit Export ─────────────────────────────────────
    json_str = model.tojson()
    print(f"📎 JSON audit trail: {len(json_str)} characters")
    print(f"   (Ready for MiFID II / SEC compliance export)")
    print()

    print("=" * 70)
    print("  ✅ Demo complete! Chronos is ready for institutional forecasting.")
    print("=" * 70)


if __name__ == '__main__':
    main()
