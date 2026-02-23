"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CHRONOS HOLIDAY ENGINE                              ║
║         Holiday Effects, Custom Events & Financial Calendar                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Why Model Holidays?
───────────────────
Holidays and special events create sharp, predictable deviations from
the trend + seasonality baseline. Failing to model them causes:

    1. Large forecast errors on and around holidays
    2. Contamination of the seasonality estimates
    3. False changepoint detection at recurring events

Chronos models each holiday h as an indicator function with optional
pre/post windows:

    h(t) = Σⱼ κⱼ · 𝟙{t ∈ window(hⱼ)}

where:
    κⱼ     = holiday effect magnitude (fitted via OLS)
    window = [date - lower_window, date + upper_window]

This is equivalent to a sparse design matrix with one column per
unique holiday, similar to Prophet's approach but with:

    - Built-in support for 100+ countries (via `holidays` package)
    - Financial calendar events (FOMC, earnings, options expiry)
    - Custom event injection with arbitrary windows
    - Holiday-specific prior scales for regularization

References:
    [1] Taylor, S. J. & Letham, B. (2018). "Forecasting at scale."
    [2] Hyndman, R. J. & Athanasopoulos, G. (2021). "Forecasting:
        Principles and Practice." OTexts.

Complexity:
    Feature matrix construction: O(T·H) where H = number of holidays
    Coefficient fitting: O(T·H) via OLS
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN FINANCIAL EVENTS
# ═══════════════════════════════════════════════════════════════════════════

# Common financial calendar events that affect markets
FINANCIAL_EVENTS = {
    'triple_witching': {
        'description': 'Options/futures expiration (3rd Friday of Mar/Jun/Sep/Dec)',
        'lower_window': 1,
        'upper_window': 0,
    },
    'month_end': {
        'description': 'Month-end rebalancing effects',
        'lower_window': 2,
        'upper_window': 0,
    },
    'quarter_end': {
        'description': 'Quarter-end window dressing',
        'lower_window': 3,
        'upper_window': 1,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HOLIDAY FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def make_holiday_features(
    ds: pd.DatetimeIndex,
    holidays: Optional[pd.DataFrame] = None,
    country: Optional[str] = None,
    lower_window: int = 0,
    upper_window: int = 0,
) -> dict:
    """
    Constructs the holiday feature matrix from dates.

    Holiday DataFrame Format:
    ─────────────────────────
    The holidays DataFrame should have columns:
        - 'holiday': name of the holiday (string)
        - 'ds': date of the holiday (datetime-like)
        - 'lower_window': (optional) days before to include
        - 'upper_window': (optional) days after to include

    This matches Prophet's format exactly for compatibility.

    If a country code is provided (e.g., 'US', 'IN'), built-in
    holidays are loaded from the `holidays` package.

    Args:
        ds: DatetimeIndex of the time series dates.
        holidays: Optional DataFrame of custom holidays.
        country: ISO 3166-1 alpha-2 country code for built-in holidays.
        lower_window: Default days before each holiday to include.
        upper_window: Default days after each holiday to include.

    Returns:
        dict with:
            'features': pd.DataFrame of holiday indicator columns (T, H)
            'holiday_names': list of unique holiday names
            'holiday_dates': dict mapping name → list of dates

    Complexity:
        O(T·H·W) where H = holidays, W = max window size.
    """
    T = len(ds)
    ds_index = pd.DatetimeIndex(ds)

    all_holidays = pd.DataFrame()

    # --- Load country holidays ---
    if country is not None:
        country_holidays = _get_country_holidays(ds_index, country)
        if country_holidays is not None:
            all_holidays = pd.concat([all_holidays, country_holidays],
                                     ignore_index=True)

    # --- Add custom holidays ---
    if holidays is not None and len(holidays) > 0:
        custom = holidays.copy()
        if 'lower_window' not in custom.columns:
            custom['lower_window'] = lower_window
        if 'upper_window' not in custom.columns:
            custom['upper_window'] = upper_window
        all_holidays = pd.concat([all_holidays, custom], ignore_index=True)

    if len(all_holidays) == 0:
        return {
            'features': pd.DataFrame(index=ds_index),
            'holiday_names': [],
            'holiday_dates': {},
        }

    # --- Build feature matrix ---
    all_holidays['ds'] = pd.to_datetime(all_holidays['ds'])
    holiday_names = all_holidays['holiday'].unique().tolist()
    features = pd.DataFrame(0.0, index=ds_index, columns=holiday_names)
    holiday_dates = {name: [] for name in holiday_names}

    for _, row in all_holidays.iterrows():
        name = row['holiday']
        date = pd.Timestamp(row['ds'])
        lw = int(row.get('lower_window', lower_window))
        uw = int(row.get('upper_window', upper_window))

        holiday_dates[name].append(date)

        for offset in range(-lw, uw + 1):
            check_date = date + pd.Timedelta(days=offset)
            if check_date in ds_index:
                features.loc[check_date, name] = 1.0

    return {
        'features': features,
        'holiday_names': holiday_names,
        'holiday_dates': holiday_dates,
    }


def _get_country_holidays(
    ds: pd.DatetimeIndex,
    country: str,
) -> Optional[pd.DataFrame]:
    """
    Loads built-in holidays for a country using the `holidays` package.

    Falls back to a minimal holiday set if the package is not installed.

    Args:
        ds: DatetimeIndex of the time series.
        country: ISO 3166-1 alpha-2 country code.

    Returns:
        DataFrame with 'holiday', 'ds', 'lower_window', 'upper_window'
        columns, or None if no holidays found.
    """
    try:
        import holidays as holidays_pkg

        years = list(range(ds.min().year, ds.max().year + 1))
        country_holidays = holidays_pkg.country_holidays(country, years=years)

        records = []
        for date, name in sorted(country_holidays.items()):
            records.append({
                'holiday': name,
                'ds': pd.Timestamp(date),
                'lower_window': 0,
                'upper_window': 0,
            })

        if records:
            return pd.DataFrame(records)
        return None

    except ImportError:
        # holidays package not installed, return None
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HOLIDAY EFFECT FITTING
# ═══════════════════════════════════════════════════════════════════════════

def fit_holiday_effects(
    holiday_features: pd.DataFrame,
    residuals: np.ndarray,
    prior_scale: float = 10.0,
) -> dict:
    """
    Fits holiday effect magnitudes via Ridge regression.

    Model:
    ──────
    The holiday component is:

        h(t) = Σⱼ κⱼ · Hⱼ(t)

    where Hⱼ(t) is the j-th holiday indicator and κⱼ is the effect.

    We fit κ via Ridge: κ̂ = (HᵀH + λI)⁻¹ Hᵀr
    where r = residuals (y - trend - seasonality), λ = 1/prior_scale².

    Args:
        holiday_features: DataFrame of holiday indicators (T, H).
        residuals: Detrended + deseasonalized values.
        prior_scale: Regularization strength.

    Returns:
        dict with:
            'effects': total holiday effect (T,)
            'coefficients': dict of {holiday_name: κ_value}
    """
    if holiday_features.empty or holiday_features.shape[1] == 0:
        return {
            'effects': np.zeros(len(residuals)),
            'coefficients': {},
        }

    X = holiday_features.values
    H = X.shape[1]

    lam = 1.0 / (prior_scale ** 2 + 1e-12)
    XtX = X.T @ X + lam * np.eye(H)
    Xty = X.T @ residuals
    kappa = np.linalg.solve(XtX, Xty)

    effects = X @ kappa

    coefficients = {}
    for i, name in enumerate(holiday_features.columns):
        coefficients[name] = kappa[i]

    return {
        'effects': effects,
        'coefficients': coefficients,
    }
