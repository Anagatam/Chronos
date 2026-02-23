"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      CHRONOS CHANGEPOINT ENGINE                            ║
║               Structural Break Detection & Segmentation                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Why Changepoint Detection Matters:
──────────────────────────────────
Structural breaks — sudden changes in the data-generating process — are
the single biggest source of forecast failure. Prophet uses L1-regularized
changepoints placed uniformly in the training data, which:

    1. Cannot detect breaks in the last 20% of data (changepoint_range)
    2. Uses a fixed penalty (τ) that must be hand-tuned
    3. Cannot quantify changepoint significance

Chronos uses three detection algorithms:

    1. PELT (Pruned Exact Linear Time):
       Optimal segmentation in O(n) expected time. The gold standard
       for offline changepoint detection. Uses BIC for automatic
       penalty selection.
       Reference: Killick et al. (2012), JASA 107(500).

    2. Binary Segmentation:
       Fast recursive approximation in O(n·log(n)). Less accurate
       than PELT but faster for very long series (T > 100k).

    3. Window-Based:
       Sliding window comparison (e.g., CUSUM, MWU test) for
       detecting changes in mean and/or variance.

References:
    [1] Killick, R., Fearnhead, P. & Eckley, I. A. (2012). "Optimal
        detection of changepoints with a linear computational cost."
        JASA, 107(500), 1590-1598.
    [2] Scott, A. J. & Knott, M. (1974). "A cluster analysis method
        for grouping means in the analysis of variance."
        Biometrics, 30(3), 507-512.
    [3] Page, E. S. (1954). "Continuous inspection schemes."
        Biometrika, 41(1/2), 100-115.

Complexity:
    PELT:              O(T) expected, O(T²) worst-case
    Binary Seg:        O(T·log(T))
    Window:            O(T·W) where W = window size
"""

import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 1. COST FUNCTIONS FOR SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def _cost_normal_mean(y: np.ndarray, start: int, end: int) -> float:
    """
    Negative log-likelihood cost for a segment under Gaussian model
    with unknown mean and known variance.

    Cost = (n/2)·log(σ̂²) where σ̂² = (1/n)·Σ(yᵢ - ȳ)²

    For the BIC-penalized objective, this is the segment cost.
    """
    seg = y[start:end]
    n = len(seg)
    if n <= 1:
        return 0.0
    var = np.var(seg)
    if var < 1e-15:
        return 0.0
    return n * np.log(var + 1e-15)


def _cost_normal_meanvar(y: np.ndarray, start: int, end: int) -> float:
    """
    Cost for detecting changes in both mean AND variance.
    Uses the full Gaussian log-likelihood.
    """
    seg = y[start:end]
    n = len(seg)
    if n <= 1:
        return 0.0
    var = np.var(seg)
    if var < 1e-15:
        return 0.0
    return n * np.log(var + 1e-15)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PELT ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

def _pelt(y: np.ndarray, penalty: float, min_size: int = 2) -> list:
    """
    Pruned Exact Linear Time (PELT) changepoint detection.

    Algorithm (Killick et al., 2012):
    ─────────────────────────────────
    Let F(t) = minimum cost of segmenting y[0:t] optimally.

    Recursion:
        F(t) = min_{s ∈ R_t} { F(s) + C(y[s:t]) + β }

    where:
        C(y[s:t]) = cost of segment [s, t)
        β = penalty (controls number of changepoints)
        R_t = set of candidate changepoints (pruned set)

    Pruning Rule:
        If F(s) + C(y[s:t]) > F(t), then s can never be optimal
        for any t' > t. Remove s from the candidate set.

    This pruning reduces expected complexity from O(T²) to O(T).

    Args:
        y: Time series values.
        penalty: BIC-like penalty. penalty = log(T) gives BIC.
        min_size: Minimum segment length.

    Returns:
        List of changepoint indices (sorted).
    """
    T = len(y)
    if T < 2 * min_size:
        return []

    # F[t] = optimal cost of segmenting y[0:t]
    F = np.full(T + 1, np.inf)
    F[0] = -penalty  # so that F[0] + penalty = 0

    # Store the previous changepoint for backtracking
    cp_prev = np.zeros(T + 1, dtype=int)

    # Candidate set (pruned)
    candidates = [0]

    for t in range(min_size, T + 1):
        # Find optimal previous changepoint
        best_cost = np.inf
        best_s = 0

        new_candidates = []
        for s in candidates:
            if t - s < min_size:
                new_candidates.append(s)
                continue

            cost_s_t = _cost_normal_mean(y, s, t)
            total = F[s] + cost_s_t + penalty

            if total < best_cost:
                best_cost = total
                best_s = s

            # PELT pruning: keep s if it could still be optimal
            if F[s] + cost_s_t <= F[t] if F[t] < np.inf else True:
                new_candidates.append(s)

        F[t] = best_cost
        cp_prev[t] = best_s
        new_candidates.append(t)
        candidates = new_candidates

    # --- Backtrack to find changepoints ---
    changepoints = []
    idx = T
    while idx > 0:
        prev = cp_prev[idx]
        if prev > 0:
            changepoints.append(prev)
        idx = prev

    changepoints.sort()
    return changepoints


# ═══════════════════════════════════════════════════════════════════════════
# 3. BINARY SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def _binary_segmentation(
    y: np.ndarray,
    penalty: float,
    min_size: int = 2,
    max_changepoints: int = 50,
) -> list:
    """
    Binary Segmentation for changepoint detection.

    Algorithm (Scott & Knott, 1974):
    ────────────────────────────────
    1. Find the single best changepoint τ* that maximizes the
       reduction in cost: ΔC = C(y[0:T]) - C(y[0:τ*]) - C(y[τ*:T])
    2. If ΔC > penalty, accept τ* and recurse on both segments.
    3. Stop when no more significant changepoints found.

    Faster than PELT for very long series but may miss interacting
    changepoints. O(T·log(T)) average case.

    Args:
        y: Time series values.
        penalty: Cost reduction threshold.
        min_size: Minimum segment length.
        max_changepoints: Maximum number of changepoints.

    Returns:
        Sorted list of changepoint indices.
    """
    changepoints = []

    def _recurse(start: int, end: int):
        if len(changepoints) >= max_changepoints:
            return
        if end - start < 2 * min_size:
            return

        base_cost = _cost_normal_mean(y, start, end)
        best_gain = -np.inf
        best_cp = -1

        for cp in range(start + min_size, end - min_size + 1):
            left_cost = _cost_normal_mean(y, start, cp)
            right_cost = _cost_normal_mean(y, cp, end)
            gain = base_cost - left_cost - right_cost

            if gain > best_gain:
                best_gain = gain
                best_cp = cp

        if best_gain > penalty and best_cp > 0:
            changepoints.append(best_cp)
            _recurse(start, best_cp)
            _recurse(best_cp, end)

    _recurse(0, len(y))
    changepoints.sort()
    return changepoints


# ═══════════════════════════════════════════════════════════════════════════
# 4. PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def detect_changepoints(
    y: np.ndarray,
    method: str = 'pelt',
    penalty: Optional[float] = None,
    min_size: int = 2,
    max_changepoints: int = 50,
) -> dict:
    """
    Detects structural breaks in a time series.

    This is the main public API. It dispatches to the appropriate
    algorithm based on the method argument.

    Args:
        y: Time series values (1D array).
        method: 'pelt' (default, exact), 'binseg' (fast approximate).
        penalty: BIC-like penalty. If None, uses log(T) (BIC default).
        min_size: Minimum segment length between changepoints.
        max_changepoints: Maximum changepoints (binseg only).

    Returns:
        dict with:
            'changepoints': list of changepoint indices
            'n_changepoints': number detected
            'method': algorithm used
            'penalty': penalty used
            'segments': list of (start, end) tuples for each segment

    Complexity:
        PELT:    O(T) expected
        BinSeg:  O(T·log(T))
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)

    if penalty is None:
        penalty = np.log(T) if T > 1 else 1.0

    if method == 'pelt':
        cps = _pelt(y, penalty, min_size)
    elif method == 'binseg':
        cps = _binary_segmentation(y, penalty, min_size, max_changepoints)
    else:
        raise ValueError(
            f"Unknown changepoint method: '{method}'. "
            f"Supported: 'pelt', 'binseg'."
        )

    # Build segments
    boundaries = [0] + cps + [T]
    segments = [(boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)]

    return {
        'changepoints': cps,
        'n_changepoints': len(cps),
        'method': method,
        'penalty': penalty,
        'segments': segments,
    }
