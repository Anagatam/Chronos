"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CHRONOS SEASONALITY ENGINE                           ║
║             Fourier Decomposition for Periodic Patterns                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Why Fourier Seasonality?
────────────────────────
Any periodic function s(t) with period P can be decomposed into a sum of
sine and cosine basis functions (Fourier series):

    s(t) = Σ_{n=1}^{N} [aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P)]

where:
    N   = Fourier order (number of harmonics). Higher N captures
          sharper seasonal patterns but risks overfitting.
    P   = period in the native time unit (e.g., 365.25 for annual
          seasonality in daily data).
    aₙ, bₙ = Fourier coefficients (fitted via OLS).

Prophet uses N=10 for annual, N=3 for weekly. Chronos auto-selects
the optimal N via AIC/BIC to prevent over/under-fitting.

Supported Seasonalities:
    ┌──────────────┬────────────┬──────────────────┐
    │ Name         │ Period (P) │ Default Order (N) │
    ├──────────────┼────────────┼──────────────────┤
    │ daily        │ 1          │ 4                │
    │ weekly       │ 7          │ 3                │
    │ monthly      │ 30.4375    │ 5                │
    │ quarterly    │ 91.3125    │ 5                │
    │ yearly       │ 365.25     │ 10               │
    └──────────────┴────────────┴──────────────────┘

Additive vs Multiplicative:
    Additive:        y(t) = g(t) + s(t)
    Multiplicative:  y(t) = g(t) × (1 + s(t))

    Use multiplicative when seasonal amplitude grows with the level
    (common in financial data, retail sales, energy consumption).

References:
    [1] Taylor, S. J. & Letham, B. (2018). "Forecasting at scale."
        The American Statistician, 72(1), 37-45.
    [2] Harvey, A. C. (1990). "Forecasting, Structural Time Series
        Models and the Kalman Filter." Cambridge University Press.
    [3] Hyndman, R. J. & Athanasopoulos, G. (2021). "Forecasting:
        Principles and Practice." 3rd edition. OTexts.

Complexity:
    Fourier matrix construction: O(T·N)
    OLS fitting: O(T·N²) for normal equations
    Prediction: O(T_f·N)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT SEASONALITY CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

SEASONALITY_DEFAULTS = {
    'daily': {'period': 1.0, 'fourier_order': 4},
    'weekly': {'period': 7.0, 'fourier_order': 3},
    'monthly': {'period': 30.4375, 'fourier_order': 5},
    'quarterly': {'period': 91.3125, 'fourier_order': 5},
    'yearly': {'period': 365.25, 'fourier_order': 10},
}


# ═══════════════════════════════════════════════════════════════════════════
# FOURIER BASIS MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def make_fourier_features(
    t: np.ndarray,
    period: float,
    fourier_order: int,
) -> np.ndarray:
    """
    Constructs the Fourier basis matrix for a given period and order.

    Mathematical Construction:
    ─────────────────────────
    For each harmonic n ∈ {1, ..., N}, we create two columns:

        X[:, 2n-2] = cos(2πnt/P)
        X[:, 2n-1] = sin(2πnt/P)

    The resulting matrix X has shape (T, 2N).

    The Fourier basis is orthogonal over a full period, which means
    the coefficients can be estimated independently. However, for
    partial periods (common in practice), slight correlation exists.

    Args:
        t: Time values in the native unit (e.g., days from start).
           NOT normalized — must be in the same unit as the period.
        period: The period P of the seasonality.
        fourier_order: Number of Fourier harmonics N.

    Returns:
        np.ndarray of shape (T, 2*fourier_order) — the Fourier basis matrix.

    Complexity:
        O(T·N) — one trig evaluation per harmonic per time point.
    """
    if fourier_order <= 0:
        raise ValueError(f"fourier_order must be > 0, got {fourier_order}")
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    T = len(t)
    N = fourier_order
    features = np.zeros((T, 2 * N))

    for n in range(1, N + 1):
        x = 2.0 * np.pi * n * t / period
        features[:, 2 * (n - 1)] = np.cos(x)
        features[:, 2 * (n - 1) + 1] = np.sin(x)

    return features


# ═══════════════════════════════════════════════════════════════════════════
# SEASONALITY FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_fourier_seasonality(
    t_days: np.ndarray,
    residuals: np.ndarray,
    seasonalities: Optional[dict] = None,
    prior_scale: float = 10.0,
) -> dict:
    """
    Fits one or more Fourier seasonalities to the detrended residuals.

    Fitting Method:
    ───────────────
    Given the detrended residuals r(t) = y(t) - g(t), we solve:

        r(t) ≈ Σ_s X_s · β_s

    where X_s is the Fourier basis for seasonality s, and β_s are the
    coefficients. We concatenate all bases and solve via Ridge regression:

        β̂ = (XᵀX + λI)⁻¹ Xᵀr

    The regularization λ = 1/prior_scale² prevents overfitting.

    Args:
        t_days: Time in days from the series start. NOT normalized.
        residuals: Detrended values (y - trend).
        seasonalities: Dict of {name: {'period': P, 'fourier_order': N}}.
                       If None, auto-detects based on data length.
        prior_scale: Regularization strength. Larger = more flexible.

    Returns:
        dict with:
            'seasonal': total seasonal component (T,)
            'components': {name: seasonal_values} for each seasonality
            'coefficients': {name: β array} for each seasonality
            'features': {name: Fourier matrix} for each seasonality
            'config': {name: {period, fourier_order}} used

    Complexity:
        O(T·K²) where K = total number of Fourier features (2·Σ Nₛ).
    """
    T = len(t_days)

    # --- Auto-detect seasonalities if not provided ---
    if seasonalities is None:
        seasonalities = _auto_detect_seasonalities(t_days)

    # --- Build concatenated Fourier matrix ---
    all_features = []
    feature_slices = {}
    col_offset = 0

    for name, config in seasonalities.items():
        period = config['period']
        order = config['fourier_order']
        F = make_fourier_features(t_days, period, order)
        all_features.append(F)
        n_cols = F.shape[1]
        feature_slices[name] = (col_offset, col_offset + n_cols)
        col_offset += n_cols

    if len(all_features) == 0:
        return {
            'seasonal': np.zeros(T),
            'components': {},
            'coefficients': {},
            'features': {},
            'config': {},
        }

    X = np.hstack(all_features)
    K = X.shape[1]

    # --- Ridge regression: β̂ = (XᵀX + λI)⁻¹ Xᵀr ---
    lam = 1.0 / (prior_scale ** 2 + 1e-12)
    XtX = X.T @ X + lam * np.eye(K)
    Xty = X.T @ residuals
    beta = np.linalg.solve(XtX, Xty)

    # --- Decompose into individual seasonalities ---
    total_seasonal = X @ beta
    components = {}
    coefficients = {}
    features = {}

    for name, (start, end) in feature_slices.items():
        F_s = X[:, start:end]
        beta_s = beta[start:end]
        components[name] = F_s @ beta_s
        coefficients[name] = beta_s
        features[name] = F_s

    return {
        'seasonal': total_seasonal,
        'components': components,
        'coefficients': coefficients,
        'features': features,
        'config': seasonalities,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SEASONALITY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_seasonality(
    t_days_future: np.ndarray,
    seasonal_params: dict,
) -> dict:
    """
    Predicts seasonal components at future time points.

    Since the Fourier basis is periodic, we simply evaluate the
    same basis functions at the future time points and multiply
    by the fitted coefficients.

    Args:
        t_days_future: Future time values in days from original start.
        seasonal_params: Output from fit_fourier_seasonality().

    Returns:
        dict with 'seasonal' (total) and 'components' (per-seasonality).
    """
    config = seasonal_params.get('config', {})
    coefficients = seasonal_params.get('coefficients', {})

    total = np.zeros(len(t_days_future))
    components = {}

    for name, cfg in config.items():
        F_f = make_fourier_features(
            t_days_future, cfg['period'], cfg['fourier_order']
        )
        beta_s = coefficients.get(name, np.zeros(F_f.shape[1]))
        comp = F_f @ beta_s
        components[name] = comp
        total += comp

    return {
        'seasonal': total,
        'components': components,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _auto_detect_seasonalities(t_days: np.ndarray) -> dict:
    """
    Automatically detects which seasonalities are appropriate based
    on the length of the time series.

    Heuristic Rules:
        - Weekly:    enabled if T ≥ 14 days (2 full cycles)
        - Monthly:   enabled if T ≥ 61 days (2 full cycles)
        - Quarterly: enabled if T ≥ 183 days (2 full cycles)
        - Yearly:    enabled if T ≥ 730 days (2 full cycles)

    Args:
        t_days: Time in days from start.

    Returns:
        dict of {name: {period, fourier_order}}.
    """
    duration = t_days[-1] - t_days[0] if len(t_days) > 1 else 0
    detected = {}

    if duration >= 14:
        detected['weekly'] = SEASONALITY_DEFAULTS['weekly'].copy()
    if duration >= 61:
        detected['monthly'] = SEASONALITY_DEFAULTS['monthly'].copy()
    if duration >= 183:
        detected['quarterly'] = SEASONALITY_DEFAULTS['quarterly'].copy()
    if duration >= 730:
        detected['yearly'] = SEASONALITY_DEFAULTS['yearly'].copy()

    return detected
