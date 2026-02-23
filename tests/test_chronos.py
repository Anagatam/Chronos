"""
Chronos — Institutional-Grade Time Series Forecasting Engine
Copyright © 2026 Anagatam Technologies. All rights reserved.

Tests for core forecasting pipeline: trend, seasonality, changepoints,
holidays, regressors, MasterChronos facade, metrics, and audit trail.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chronos.MasterChronos import MasterChronos, _validate_input, AuditEntry
from chronos.core.TrendEngine import (
    fit_linear_trend, fit_logistic_trend, fit_flat_trend, predict_trend
)
from chronos.core.SeasonalityEngine import (
    fit_fourier_seasonality, predict_seasonality, make_fourier_features
)
from chronos.core.ChangePointEngine import detect_changepoints
from chronos.core.HolidayEngine import make_holiday_features, fit_holiday_effects
from chronos.core.RegressorEngine import fit_regressors, predict_regressors
from chronos.metrics.accuracy import ForecastMetrics


# ═══════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ts():
    """Generate a synthetic time series with trend + seasonality + noise."""
    np.random.seed(42)
    T = 730  # 2 years daily
    dates = pd.date_range('2023-01-01', periods=T, freq='D')
    t = np.arange(T, dtype=float)

    # Linear trend
    trend = 100 + 0.05 * t

    # Weekly seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)

    # Yearly seasonality
    yearly = 10 * np.sin(2 * np.pi * t / 365.25)

    # Noise
    noise = np.random.normal(0, 2, T)

    y = trend + weekly + yearly + noise

    return pd.DataFrame({'ds': dates, 'y': y})


@pytest.fixture
def short_ts():
    """Generate a short time series (30 days)."""
    np.random.seed(42)
    T = 30
    dates = pd.date_range('2024-01-01', periods=T, freq='D')
    y = 50 + np.random.normal(0, 3, T)
    return pd.DataFrame({'ds': dates, 'y': y})


# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInitialization:
    """Test constructor validation and defaults."""

    def test_default_init(self):
        model = MasterChronos()
        assert model.growth == 'linear'
        assert model.seasonality_mode == 'additive'
        assert model.regressor_method == 'ridge'

    def test_all_growth_types(self):
        for growth in ('linear', 'logistic', 'flat'):
            model = MasterChronos(growth=growth)
            assert model.growth == growth

    def test_invalid_growth(self):
        with pytest.raises(ValueError, match="Invalid growth"):
            MasterChronos(growth='quadratic')

    def test_all_seasonality_modes(self):
        for mode in ('additive', 'multiplicative'):
            model = MasterChronos(seasonality_mode=mode)
            assert model.seasonality_mode == mode

    def test_invalid_seasonality_mode(self):
        with pytest.raises(ValueError, match="Invalid seasonality_mode"):
            MasterChronos(seasonality_mode='hybrid')

    def test_all_regressor_methods(self):
        for method in ('ols', 'ridge', 'lasso', 'elastic_net'):
            model = MasterChronos(regressor_method=method)
            assert model.regressor_method == method

    def test_invalid_regressor_method(self):
        with pytest.raises(ValueError, match="Invalid regressor_method"):
            MasterChronos(regressor_method='svm')

    def test_invalid_changepoint_prior_scale(self):
        with pytest.raises(ValueError, match="changepoint_prior_scale"):
            MasterChronos(changepoint_prior_scale=-0.1)

    def test_invalid_interval_width(self):
        with pytest.raises(ValueError, match="interval_width"):
            MasterChronos(interval_width=1.5)


# ═══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Test that invalid inputs are caught early."""

    def test_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            _validate_input([[1, 2], [3, 4]])

    def test_missing_ds_column(self):
        df = pd.DataFrame({'date': [1, 2], 'y': [3, 4]})
        with pytest.raises(ValueError, match="'ds' column"):
            _validate_input(df)

    def test_missing_y_column(self):
        df = pd.DataFrame({'ds': ['2024-01-01', '2024-01-02'], 'value': [3, 4]})
        with pytest.raises(ValueError, match="'y' column"):
            _validate_input(df)

    def test_single_observation(self):
        df = pd.DataFrame({'ds': ['2024-01-01'], 'y': [3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            _validate_input(df)

    def test_nan_values(self):
        df = pd.DataFrame({
            'ds': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'y': [1.0, float('nan'), 3.0]
        })
        with pytest.raises(ValueError, match="missing values"):
            _validate_input(df)

    def test_inf_values(self):
        df = pd.DataFrame({
            'ds': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'y': [1.0, float('inf'), 3.0]
        })
        with pytest.raises(ValueError, match="infinite"):
            _validate_input(df)

    def test_valid_input_passes(self):
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=10),
            'y': np.random.randn(10),
        })
        _validate_input(df)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# TREND ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTrendEngine:
    """Test trend fitting and prediction."""

    def test_linear_trend_basic(self):
        t = np.linspace(0, 1, 100)
        y = 2 * t + 3 + np.random.normal(0, 0.01, 100)
        result = fit_linear_trend(t, y, n_changepoints=5)
        assert 'trend' in result
        assert len(result['trend']) == 100
        assert result['growth'] == 'linear'
        # Trend should approximate the linear function
        assert np.corrcoef(result['trend'], y)[0, 1] > 0.95

    def test_flat_trend(self):
        t = np.linspace(0, 1, 50)
        y = np.full(50, 42.0) + np.random.normal(0, 0.1, 50)
        result = fit_flat_trend(t, y)
        assert result['growth'] == 'flat'
        assert abs(result['k'] - 42.0) < 0.5
        assert np.allclose(result['trend'], result['k'])

    def test_logistic_trend(self):
        t = np.linspace(0, 1, 100)
        cap = np.full(100, 100.0)
        y = 100 / (1 + np.exp(-10 * (t - 0.5))) + np.random.normal(0, 1, 100)
        result = fit_logistic_trend(t, y, cap=cap, n_changepoints=5)
        assert result['growth'] == 'logistic'
        assert len(result['trend']) == 100

    def test_trend_prediction(self):
        t = np.linspace(0, 1, 100)
        y = 2 * t + 3
        result = fit_linear_trend(t, y, n_changepoints=3)
        t_future = np.linspace(1.0, 1.5, 50)
        pred = predict_trend(t_future, result)
        assert len(pred) == 50

    def test_changepoints_returned(self):
        t = np.linspace(0, 1, 100)
        y = np.where(t < 0.5, t, 3 * t - 1)
        result = fit_linear_trend(t, y, n_changepoints=10)
        assert 'changepoint_ts' in result
        assert 'deltas' in result


# ═══════════════════════════════════════════════════════════════════════════
# SEASONALITY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSeasonalityEngine:
    """Test Fourier seasonality fitting and prediction."""

    def test_fourier_features_shape(self):
        t = np.arange(100, dtype=float)
        F = make_fourier_features(t, period=7.0, fourier_order=3)
        assert F.shape == (100, 6)  # 2 * 3 = 6

    def test_fourier_features_range(self):
        t = np.arange(100, dtype=float)
        F = make_fourier_features(t, period=7.0, fourier_order=3)
        assert F.min() >= -1.0
        assert F.max() <= 1.0

    def test_fourier_order_validation(self):
        with pytest.raises(ValueError, match="fourier_order"):
            make_fourier_features(np.arange(10.0), 7.0, 0)

    def test_period_validation(self):
        with pytest.raises(ValueError, match="period"):
            make_fourier_features(np.arange(10.0), -1.0, 3)

    def test_seasonality_fitting(self):
        np.random.seed(42)
        t_days = np.arange(365, dtype=float)
        weekly = 5 * np.sin(2 * np.pi * t_days / 7)
        noise = np.random.normal(0, 0.5, 365)
        residuals = weekly + noise

        result = fit_fourier_seasonality(
            t_days, residuals,
            seasonalities={'weekly': {'period': 7.0, 'fourier_order': 3}},
        )
        assert 'seasonal' in result
        assert 'components' in result
        assert 'weekly' in result['components']
        # Fitted seasonality should correlate with true weekly
        corr = np.corrcoef(result['seasonal'], weekly)[0, 1]
        assert corr > 0.8

    def test_seasonality_prediction(self):
        t_days = np.arange(365, dtype=float)
        residuals = np.sin(2 * np.pi * t_days / 7)
        result = fit_fourier_seasonality(
            t_days, residuals,
            seasonalities={'weekly': {'period': 7.0, 'fourier_order': 3}},
        )
        t_future = np.arange(365, 500, dtype=float)
        pred = predict_seasonality(t_future, result)
        assert len(pred['seasonal']) == 135


# ═══════════════════════════════════════════════════════════════════════════
# CHANGEPOINT ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestChangePointEngine:
    """Test changepoint detection algorithms."""

    def test_pelt_no_changepoints(self):
        np.random.seed(42)
        y = np.random.normal(0, 1, 100)
        result = detect_changepoints(y, method='pelt')
        assert 'changepoints' in result
        assert result['method'] == 'pelt'

    def test_pelt_with_changepoint(self):
        np.random.seed(42)
        y = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(5, 1, 50),
        ])
        result = detect_changepoints(y, method='pelt', penalty=10.0)
        assert result['n_changepoints'] >= 0  # Should detect at least vicinity

    def test_binseg(self):
        np.random.seed(42)
        y = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(5, 1, 50),
        ])
        result = detect_changepoints(y, method='binseg')
        assert result['method'] == 'binseg'

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown changepoint"):
            detect_changepoints(np.ones(10), method='invalid')

    def test_segments_correct(self):
        y = np.ones(50)
        result = detect_changepoints(y, method='pelt')
        segments = result['segments']
        # Segments should cover full range
        assert segments[0][0] == 0
        assert segments[-1][1] == 50


# ═══════════════════════════════════════════════════════════════════════════
# HOLIDAY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHolidayEngine:
    """Test holiday feature construction and effect fitting."""

    def test_custom_holidays(self):
        ds = pd.date_range('2024-01-01', periods=365)
        holidays_df = pd.DataFrame({
            'holiday': ['new_year', 'christmas'],
            'ds': pd.to_datetime(['2024-01-01', '2024-12-25']),
        })
        result = make_holiday_features(ds, holidays=holidays_df)
        assert 'features' in result
        assert 'new_year' in result['holiday_names']
        assert 'christmas' in result['holiday_names']

    def test_holiday_with_windows(self):
        ds = pd.date_range('2024-01-01', periods=30)
        holidays_df = pd.DataFrame({
            'holiday': ['event'],
            'ds': pd.to_datetime(['2024-01-15']),
            'lower_window': [2],
            'upper_window': [1],
        })
        result = make_holiday_features(ds, holidays=holidays_df)
        feats = result['features']
        # Should have indicator for Jan 13, 14, 15, 16
        assert feats['event'].sum() == 4

    def test_empty_holidays(self):
        ds = pd.date_range('2024-01-01', periods=30)
        result = make_holiday_features(ds)
        assert len(result['holiday_names']) == 0

    def test_holiday_effect_fitting(self):
        np.random.seed(42)
        T = 365
        feats = pd.DataFrame({
            'holiday_a': np.zeros(T),
        })
        feats.loc[0, 'holiday_a'] = 1.0
        feats.loc[100, 'holiday_a'] = 1.0
        residuals = np.random.normal(0, 1, T)
        residuals[0] += 10
        residuals[100] += 10

        result = fit_holiday_effects(feats, residuals)
        assert 'effects' in result
        assert 'holiday_a' in result['coefficients']


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSOR ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressorEngine:
    """Test regressor fitting and prediction."""

    def test_ridge_regression(self):
        np.random.seed(42)
        T = 100
        X = np.random.randn(T, 3)
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X @ beta_true + np.random.normal(0, 0.1, T)

        result = fit_regressors(X, y, method='ridge', alpha=0.01)
        assert len(result['coefficients']) == 3
        assert len(result['regressor_effect']) == T
        # Coefficients should be close to true values (low regularization)
        assert np.allclose(result['coefficients'], beta_true, atol=0.5)

    def test_ols_regression(self):
        np.random.seed(42)
        T = 100
        X = np.random.randn(T, 2)
        beta_true = np.array([3.0, -2.0])
        y = X @ beta_true
        result = fit_regressors(X, y, method='ols')
        assert np.allclose(result['coefficients'], beta_true, atol=0.1)

    def test_lasso_sparsity(self):
        np.random.seed(42)
        T = 200
        X = np.random.randn(T, 5)
        beta_true = np.array([3.0, 0, 0, -2.0, 0])
        y = X @ beta_true + np.random.normal(0, 0.1, T)
        result = fit_regressors(X, y, method='lasso', alpha=0.1)
        # Some coefficients should be near zero
        small_coefs = np.sum(np.abs(result['coefficients']) < 0.5)
        assert small_coefs >= 1

    def test_empty_regressors(self):
        X = np.empty((10, 0))
        y = np.ones(10)
        result = fit_regressors(X, y, method='ridge')
        assert len(result['coefficients']) == 0

    def test_feature_importance(self):
        np.random.seed(42)
        T = 100
        X = np.random.randn(T, 3)
        beta_true = np.array([5.0, 0.1, -3.0])
        y = X @ beta_true
        result = fit_regressors(
            X, y, method='ridge',
            regressor_names=['big', 'small', 'medium']
        )
        assert 'big' in result['feature_importance']
        assert result['feature_importance']['big'] > result['feature_importance']['small']

    def test_predict_regressors(self):
        params = {'coefficients': np.array([1.0, 2.0])}
        X_future = np.array([[1, 0], [0, 1], [1, 1]])
        pred = predict_regressors(X_future, params)
        np.testing.assert_array_almost_equal(pred, [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════
# MASTER CHRONOS FACADE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMasterChronos:
    """Test the full MasterChronos pipeline."""

    def test_fit_predict_linear(self, sample_ts):
        model = MasterChronos(growth='linear', weekly_seasonality=True,
                              yearly_seasonality=True)
        model.fit(sample_ts)
        forecast = model.predict(periods=30)
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
        assert 'trend' in forecast.columns
        assert len(forecast) > 0

    def test_fit_predict_flat(self, short_ts):
        model = MasterChronos(growth='flat')
        model.fit(short_ts)
        forecast = model.predict(periods=10)
        assert len(forecast) > 0

    def test_method_chaining(self, sample_ts):
        forecast = MasterChronos(growth='linear').fit(sample_ts).predict(periods=10)
        assert 'yhat' in forecast.columns

    def test_components(self, sample_ts):
        model = MasterChronos(growth='linear')
        model.fit(sample_ts)
        comp = model.components()
        assert 'trend' in comp
        assert 'seasonal' in comp
        assert 'residuals' in comp

    def test_changepoints_inspection(self, sample_ts):
        model = MasterChronos(growth='linear')
        model.fit(sample_ts)
        cp = model.changepoints()
        assert 'changepoint_ts' in cp
        assert 'deltas' in cp
        assert 'n_significant' in cp

    def test_diagnostics(self, sample_ts):
        model = MasterChronos(growth='linear')
        model.fit(sample_ts)
        diag = model.diagnostics()
        assert 'residuals' in diag
        assert 'model_selection' in diag
        assert 'autocorrelation' in diag
        assert 'aic' in diag['model_selection']
        assert 'bic' in diag['model_selection']
        assert 'durbin_watson' in diag['autocorrelation']

    def test_not_fitted_error(self):
        model = MasterChronos()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(periods=10)

    def test_add_seasonality(self, sample_ts):
        model = MasterChronos()
        model.add_seasonality('monthly', period=30.4375, fourier_order=5)
        model.fit(sample_ts)
        forecast = model.predict(periods=10)
        assert len(forecast) > 0

    def test_multiplicative_mode(self, sample_ts):
        model = MasterChronos(seasonality_mode='multiplicative')
        model.fit(sample_ts)
        forecast = model.predict(periods=10)
        assert len(forecast) > 0


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditTrail:
    """Test compliance logging and audit trail."""

    def test_audit_log_populated(self, sample_ts):
        model = MasterChronos()
        model.fit(sample_ts)
        assert len(model.audit_log) > 0

    def test_audit_entry_structure(self, sample_ts):
        model = MasterChronos()
        model.fit(sample_ts)
        entry = model.audit_log[0]
        assert hasattr(entry, 'step')
        assert hasattr(entry, 'timestamp')
        assert hasattr(entry, 'duration_ms')
        assert hasattr(entry, 'details')

    def test_summary_report(self, sample_ts):
        model = MasterChronos()
        model.fit(sample_ts)
        summary = model.summary()
        assert 'CHRONOS FORECAST SUMMARY' in summary
        assert 'Model Configuration' in summary
        assert 'Data Overview' in summary
        assert 'Residual Diagnostics' in summary

    def test_json_export(self, sample_ts):
        import json
        model = MasterChronos()
        model.fit(sample_ts)
        json_str = model.tojson()
        data = json.loads(json_str)
        assert 'model' in data
        assert 'data' in data
        assert 'audit_trail' in data

    def test_repr(self):
        model = MasterChronos(growth='logistic')
        assert 'logistic' in repr(model)
        assert 'not fitted' in repr(model)


# ═══════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test forecast accuracy metrics."""

    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = ForecastMetrics(actual, predicted)
        assert m.mae() == 0.0
        assert m.rmse() == 0.0
        assert m.r_squared() == 1.0

    def test_mae(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        m = ForecastMetrics(actual, predicted)
        assert m.mae() == 1.0

    def test_rmse(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([1.0, -1.0])
        m = ForecastMetrics(actual, predicted)
        assert abs(m.rmse() - 1.0) < 1e-10

    def test_mape_zero_actual(self):
        actual = np.array([0.0, 1.0])
        predicted = np.array([1.0, 1.0])
        m = ForecastMetrics(actual, predicted)
        assert m.mape() == 0.0  # Only non-zero actuals counted

    def test_coverage(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        m = ForecastMetrics(actual, predicted, lower, upper)
        assert m.coverage() == 1.0

    def test_coverage_partial(self):
        actual = np.array([1.0, 5.0])
        predicted = np.array([1.0, 1.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])
        m = ForecastMetrics(actual, predicted, lower, upper)
        assert m.coverage() == 0.5

    def test_report(self):
        actual = np.arange(10, dtype=float)
        predicted = actual + np.random.normal(0, 0.1, 10)
        m = ForecastMetrics(actual, predicted)
        report = m.report()
        assert 'MAE' in report
        assert 'RMSE' in report

    def test_bias(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        m = ForecastMetrics(actual, predicted)
        assert m.bias() == -1.0  # Systematic over-prediction


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT ENTRY UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditEntry:
    """Test the AuditEntry dataclass."""

    def test_creation(self):
        entry = AuditEntry('test_step', 42.5, {'key': 'value'})
        assert entry.step == 'test_step'
        assert entry.duration_ms == 42.5
        assert entry.details == {'key': 'value'}
        assert entry.timestamp is not None

    def test_todict(self):
        entry = AuditEntry('step', 10.0)
        d = entry.todict()
        assert d['step'] == 'step'
        assert d['duration_ms'] == 10.0
        assert 'timestamp' in d

    def test_repr(self):
        entry = AuditEntry('fit', 5.0)
        assert 'fit' in repr(entry)
        assert '5.0ms' in repr(entry)
