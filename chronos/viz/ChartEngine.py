"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CHRONOS CHART ENGINE                                 ║
║              Interactive Plotly Visualizations for Forecasting              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Provides 6 interactive Plotly charts for forecast analysis:

    1. Forecast Plot — actual vs predicted with confidence intervals
    2. Component Plot — trend, seasonality, holidays decomposed
    3. Changepoint Plot — detected structural breaks
    4. Residual Plot — diagnostics (histogram + time series)
    5. Cross-Validation Plot — accuracy over time
    6. Seasonality Heatmap — day-of-week × month patterns

Design Philosophy:
    - Dark theme by default (institutional aesthetic)
    - Interactive (zoom, hover, export)
    - Publication-quality (LaTeX-ready)
    - Consistent color palette

References:
    [1] Canopy ChartEngine pattern (Anagatam Technologies, 2026).

Complexity:
    All charts: O(T) where T = number of data points.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    'primary': '#6366F1',       # Indigo
    'secondary': '#8B5CF6',     # Violet
    'accent': '#EC4899',        # Pink
    'success': '#10B981',       # Emerald
    'warning': '#F59E0B',       # Amber
    'danger': '#EF4444',        # Red
    'info': '#3B82F6',          # Blue
    'background': '#0F172A',    # Slate 900
    'surface': '#1E293B',       # Slate 800
    'text': '#F1F5F9',          # Slate 100
    'text_dim': '#94A3B8',      # Slate 400
    'grid': '#334155',          # Slate 700
    'ci_fill': 'rgba(99, 102, 241, 0.15)',  # Indigo 15%
}


class ChartEngine:
    """
    Interactive Plotly chart generator for Chronos forecasts.

    Usage:
        >>> from chronos.viz import ChartEngine
        >>> charts = ChartEngine(model, forecast, df)
        >>> charts.plot_forecast().show()
        >>> charts.plot_components().show()
    """

    def __init__(
        self,
        model=None,
        forecast: Optional[pd.DataFrame] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.forecast = forecast
        self.df = df

    def plot_forecast(
        self,
        title: str = 'Chronos Forecast',
        width: int = 1100,
        height: int = 550,
    ) -> 'go.Figure':
        """
        Plots actual data, forecast, and confidence intervals.

        Returns:
            Plotly Figure object.
        """
        self._check_plotly()

        fig = go.Figure()

        # Confidence interval (shaded band)
        if 'yhat_upper' in self.forecast.columns:
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=COLORS['ci_fill'],
                name='Prediction Interval',
                hoverinfo='skip',
            ))

        # Actual data
        if self.df is not None:
            fig.add_trace(go.Scatter(
                x=self.df['ds'],
                y=self.df['y'],
                mode='markers',
                marker=dict(size=3, color=COLORS['text_dim'], opacity=0.6),
                name='Actual',
            ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['yhat'],
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            name='Forecast',
        ))

        # Trend line
        if 'trend' in self.forecast.columns:
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['trend'],
                mode='lines',
                line=dict(color=COLORS['accent'], width=1.5, dash='dash'),
                name='Trend',
                visible='legendonly',
            ))

        self._apply_layout(fig, title, width, height)
        return fig

    def plot_components(
        self,
        title: str = 'Forecast Components',
        width: int = 1100,
        height: int = 800,
    ) -> 'go.Figure':
        """
        Plots the decomposed forecast components (trend, seasonality, etc.).

        Returns:
            Plotly Figure with subplots for each component.
        """
        self._check_plotly()

        components = ['trend']
        available = self.forecast.columns.tolist()

        # Find seasonal components
        seasonal_cols = [c for c in available if c.startswith('seasonal_')]
        if seasonal_cols:
            components.extend(seasonal_cols)
        elif 'seasonal' in available:
            components.append('seasonal')

        if 'holidays' in available:
            components.append('holidays')

        n_components = len(components)

        fig = make_subplots(
            rows=n_components,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[c.replace('_', ' ').title() for c in components],
        )

        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
                  COLORS['success'], COLORS['warning'], COLORS['info']]

        for i, comp in enumerate(components):
            if comp in self.forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.forecast['ds'],
                        y=self.forecast[comp],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        name=comp.replace('_', ' ').title(),
                    ),
                    row=i + 1,
                    col=1,
                )

        self._apply_layout(fig, title, width, height)
        return fig

    def plot_changepoints(
        self,
        title: str = 'Changepoint Detection',
        width: int = 1100,
        height: int = 500,
    ) -> 'go.Figure':
        """
        Plots the trend with detected changepoints marked.

        Returns:
            Plotly Figure with vertical lines at changepoints.
        """
        self._check_plotly()

        fig = go.Figure()

        # Actual data
        if self.df is not None:
            fig.add_trace(go.Scatter(
                x=self.df['ds'],
                y=self.df['y'],
                mode='markers',
                marker=dict(size=3, color=COLORS['text_dim'], opacity=0.5),
                name='Actual',
            ))

        # Trend
        if 'trend' in self.forecast.columns:
            fig.add_trace(go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['trend'],
                mode='lines',
                line=dict(color=COLORS['primary'], width=2),
                name='Trend',
            ))

        # Changepoints
        if self.model is not None and self.model._fitted:
            cp_info = self.model.changepoints()
            for date in cp_info.get('changepoint_dates', []):
                idx = cp_info['changepoint_dates'].index(date)
                if idx < len(cp_info['deltas']):
                    delta = cp_info['deltas'][idx]
                    if abs(delta) > 0.01:
                        fig.add_vline(
                            x=date,
                            line_dash='dot',
                            line_color=COLORS['accent'],
                            opacity=min(abs(delta) * 5, 0.8),
                            annotation_text=f'Δ={delta:.3f}',
                        )

        self._apply_layout(fig, title, width, height)
        return fig

    def plot_residuals(
        self,
        title: str = 'Residual Diagnostics',
        width: int = 1100,
        height: int = 600,
    ) -> 'go.Figure':
        """
        Plots residual diagnostics (time series + histogram).

        Returns:
            Plotly Figure with two subplots.
        """
        self._check_plotly()

        if self.model is None or not self.model._fitted:
            raise RuntimeError("Model must be fitted for residual analysis.")

        residuals = self.model._residuals

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Residuals Over Time', 'Residual Distribution'],
            vertical_spacing=0.12,
        )

        # Residuals over time
        ds = self.model._df['ds']
        fig.add_trace(
            go.Scatter(
                x=ds,
                y=residuals,
                mode='markers',
                marker=dict(size=3, color=COLORS['primary'], opacity=0.6),
                name='Residuals',
            ),
            row=1, col=1,
        )
        fig.add_hline(y=0, line_dash='dash', line_color=COLORS['warning'],
                      row=1, col=1)

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=50,
                marker_color=COLORS['primary'],
                opacity=0.7,
                name='Distribution',
            ),
            row=2, col=1,
        )

        self._apply_layout(fig, title, width, height)
        return fig

    def plot_cv_results(
        self,
        cv_results: pd.DataFrame,
        title: str = 'Cross-Validation Results',
        width: int = 1100,
        height: int = 500,
    ) -> 'go.Figure':
        """
        Plots cross-validation actual vs predicted over time.

        Args:
            cv_results: DataFrame from CrossValidator.run().results

        Returns:
            Plotly Figure.
        """
        self._check_plotly()

        fig = go.Figure()

        valid = cv_results.dropna(subset=['actual', 'predicted'])

        fig.add_trace(go.Scatter(
            x=valid['ds'],
            y=valid['actual'],
            mode='markers',
            marker=dict(size=4, color=COLORS['text_dim']),
            name='Actual',
        ))

        fig.add_trace(go.Scatter(
            x=valid['ds'],
            y=valid['predicted'],
            mode='markers',
            marker=dict(size=4, color=COLORS['primary']),
            name='Predicted',
        ))

        self._apply_layout(fig, title, width, height)
        return fig

    def plot_seasonality_heatmap(
        self,
        title: str = 'Seasonality Heatmap',
        width: int = 900,
        height: int = 500,
    ) -> 'go.Figure':
        """
        Plots a day-of-week × month heatmap of the target variable.

        Reveals weekly-monthly interaction patterns.

        Returns:
            Plotly Figure with heatmap.
        """
        self._check_plotly()

        if self.df is None:
            raise ValueError("DataFrame required for seasonality heatmap.")

        df = self.df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['day_of_week'] = df['ds'].dt.day_name()
        df['month'] = df['ds'].dt.month_name()

        # Pivot for heatmap
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                      'Friday', 'Saturday', 'Sunday']
        months_order = ['January', 'February', 'March', 'April', 'May',
                        'June', 'July', 'August', 'September', 'October',
                        'November', 'December']

        pivot = df.pivot_table(
            values='y', index='day_of_week', columns='month',
            aggfunc='mean'
        )

        # Reorder
        pivot = pivot.reindex(
            index=[d for d in days_order if d in pivot.index],
            columns=[m for m in months_order if m in pivot.columns],
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[m[:3] for m in pivot.columns],
            y=[d[:3] for d in pivot.index],
            colorscale='Viridis',
            colorbar=dict(title='Mean Value'),
        ))

        self._apply_layout(fig, title, width, height)
        return fig

    # ═══════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════

    def _apply_layout(self, fig, title, width, height):
        """Applies the institutional dark theme layout."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=COLORS['text'])),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'], family='Inter, sans-serif'),
            width=width,
            height=height,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_dim']),
            ),
            hovermode='x unified',
            margin=dict(l=60, r=30, t=60, b=40),
        )
        fig.update_xaxes(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
        )
        fig.update_yaxes(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
        )

    @staticmethod
    def _check_plotly():
        if not HAS_PLOTLY:
            raise ImportError(
                "Plotly is required for ChartEngine. "
                "Install it: pip install plotly"
            )
