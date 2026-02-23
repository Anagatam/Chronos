"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CHRONOS REGRESSOR ENGINE                             ║
║            Exogenous Variable Regression with Regularization               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Why Regularized Regressors?
───────────────────────────
Exogenous regressors x(t) allow the model to incorporate external
drivers (temperature, marketing spend, economic indicators, etc.):

    y(t) = g(t) + s(t) + h(t) + βᵀx(t) + ε(t)

Prophet uses simple OLS for regressor coefficients, which:
    1. Overfits with many correlated regressors
    2. Cannot perform feature selection
    3. Gives no coefficient stability guarantees

Chronos provides four regularization methods:

    ┌──────────────┬────────────────────────────────┬───────────────────┐
    │ Method       │ Penalty                        │ Use Case          │
    ├──────────────┼────────────────────────────────┼───────────────────┤
    │ OLS          │ None                           │ Few regressors    │
    │ Ridge (L2)   │ λ·‖β‖₂²                       │ Correlated X      │
    │ Lasso (L1)   │ λ·‖β‖₁                        │ Feature selection │
    │ ElasticNet   │ α·λ·‖β‖₁ + (1-α)·λ·‖β‖₂²    │ Default. Best mix │
    └──────────────┴────────────────────────────────┴───────────────────┘

References:
    [1] Tibshirani, R. (1996). "Regression Shrinkage and Selection
        via the Lasso." JRSS-B, 58(1), 267-288.
    [2] Zou, H. & Hastie, T. (2005). "Regularization and variable
        selection via the elastic net." JRSS-B, 67(2), 301-320.
    [3] Hoerl, A. E. & Kennard, R. W. (1970). "Ridge regression:
        biased estimation for nonorthogonal problems." Technometrics.

Complexity:
    OLS:        O(T·K²) for normal equations
    Ridge:      O(T·K²) for normal equations
    Lasso:      O(T·K·I) for coordinate descent (I iterations)
    ElasticNet: O(T·K·I) for coordinate descent
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSOR FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_regressors(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'ridge',
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    regressor_names: Optional[list] = None,
) -> dict:
    """
    Fits regressor coefficients with optional regularization.

    Mathematical Formulation:
    ─────────────────────────
    We minimize the penalized loss:

        L(β) = (1/2T)·‖y - Xβ‖₂² + P(β)

    where P(β) depends on the method:

        OLS:        P(β) = 0
        Ridge:      P(β) = α·‖β‖₂²
        Lasso:      P(β) = α·‖β‖₁
        ElasticNet: P(β) = α·[ρ·‖β‖₁ + (1-ρ)·‖β‖₂²/2]
                    where ρ = l1_ratio

    Fitting:
    ────────
    - OLS/Ridge: Closed-form via normal equations
    - Lasso/ElasticNet: Coordinate descent (sklearn)

    Args:
        X: Regressor matrix (T, K).
        y: Target values (residuals after trend/seasonality/holidays).
        method: 'ols', 'ridge', 'lasso', 'elastic_net'.
        alpha: Regularization strength. Larger = stronger.
        l1_ratio: Mixing parameter for ElasticNet [0, 1].
                  l1_ratio=1 → Lasso, l1_ratio=0 → Ridge.
        regressor_names: Optional names for feature importance.

    Returns:
        dict with:
            'coefficients': β array (K,)
            'regressor_effect': X @ β (T,)
            'feature_importance': dict of {name: |βᵢ| / Σ|β|}
            'method': method used
    """
    T, K = X.shape

    if K == 0:
        return {
            'coefficients': np.array([]),
            'regressor_effect': np.zeros(T),
            'feature_importance': {},
            'method': method,
        }

    if regressor_names is None:
        regressor_names = [f'x_{i}' for i in range(K)]

    # --- Standardize regressors ---
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_norm = (X - X_mean) / X_std

    if method == 'ols':
        beta_norm = np.linalg.lstsq(X_norm, y, rcond=None)[0]

    elif method == 'ridge':
        I = np.eye(K)
        beta_norm = np.linalg.solve(
            X_norm.T @ X_norm + alpha * T * I,
            X_norm.T @ y
        )

    elif method == 'lasso':
        beta_norm = _coordinate_descent(
            X_norm, y, alpha * T, l1_ratio=1.0
        )

    elif method == 'elastic_net':
        beta_norm = _coordinate_descent(
            X_norm, y, alpha * T, l1_ratio=l1_ratio
        )

    else:
        raise ValueError(
            f"Unknown regressor method: '{method}'. "
            f"Supported: 'ols', 'ridge', 'lasso', 'elastic_net'."
        )

    # --- Unstandardize ---
    beta = beta_norm / X_std

    # --- Feature importance ---
    abs_beta = np.abs(beta)
    total = abs_beta.sum()
    importance = {}
    if total > 1e-12:
        for i, name in enumerate(regressor_names):
            importance[name] = float(abs_beta[i] / total)
    else:
        for name in regressor_names:
            importance[name] = 1.0 / K

    return {
        'coefficients': beta,
        'regressor_effect': X @ beta,
        'feature_importance': importance,
        'method': method,
    }


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATE DESCENT (Lasso / ElasticNet)
# ═══════════════════════════════════════════════════════════════════════════

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    l1_ratio: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Coordinate descent for Lasso / ElasticNet.

    Update Rule (for each coordinate j):
    ─────────────────────────────────────
    For ElasticNet:

        ρⱼ = Xⱼᵀ(y - X_{-j}β_{-j}) / T

        βⱼ = S(ρⱼ, λ·α) / (‖Xⱼ‖₂²/T + λ·(1-α))

    where S(z, γ) = sign(z)·max(|z| - γ, 0) is the soft-threshold.

    Args:
        X: Standardized regressor matrix (T, K).
        y: Target values.
        lam: Total regularization strength (already scaled by T).
        l1_ratio: L1 vs L2 mixing.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Coefficient vector β (K,).
    """
    T, K = X.shape
    beta = np.zeros(K)
    r = y.copy()  # residuals

    l1_pen = lam * l1_ratio
    l2_pen = lam * (1 - l1_ratio)

    # Precompute column norms
    col_norms_sq = np.sum(X ** 2, axis=0) / T

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(K):
            # Add back contribution of feature j
            r += X[:, j] * beta[j]

            # Compute univariate OLS
            rho = X[:, j].dot(r) / T

            # Soft-threshold (Lasso part)
            if rho > l1_pen / T:
                beta[j] = (rho - l1_pen / T) / (col_norms_sq[j] + l2_pen / T)
            elif rho < -l1_pen / T:
                beta[j] = (rho + l1_pen / T) / (col_norms_sq[j] + l2_pen / T)
            else:
                beta[j] = 0.0

            # Update residuals
            r -= X[:, j] * beta[j]

        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION WITH REGRESSORS
# ═══════════════════════════════════════════════════════════════════════════

def predict_regressors(
    X_future: np.ndarray,
    regressor_params: dict,
) -> np.ndarray:
    """
    Predicts regressor contribution at future time points.

    Args:
        X_future: Future regressor values (T_f, K).
        regressor_params: Output from fit_regressors().

    Returns:
        np.ndarray of regressor effect at each future point.
    """
    beta = regressor_params.get('coefficients', np.array([]))
    if len(beta) == 0 or X_future.shape[1] == 0:
        return np.zeros(X_future.shape[0])
    return X_future @ beta
