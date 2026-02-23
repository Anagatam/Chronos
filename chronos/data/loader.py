"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CHRONOS DATA LOADER                                 ║
║               Zero-Boilerplate Time Series Data Pipeline                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

DataLoader provides a unified interface for loading time series data from
multiple sources, following the Canopy pattern of zero-boilerplate pipelines.

Supported Sources:
    ┌───────────────┬─────────────────────────────────────────────────┐
    │ Source        │ Method                                          │
    ├───────────────┼─────────────────────────────────────────────────┤
    │ Yahoo Finance │ DataLoader.yfinance(ticker, start, end)         │
    │ CSV           │ DataLoader.csv(path, ds_col, y_col)             │
    │ Parquet       │ DataLoader.parquet(path, ds_col, y_col)         │
    │ DataFrame     │ DataLoader.dataframe(df, ds_col, y_col)         │
    └───────────────┴─────────────────────────────────────────────────┘

All methods return a DataFrame with 'ds' and 'y' columns, ready for
MasterChronos.fit().

References:
    [1] Canopy DataLoader pattern (Anagatam Technologies, 2026).

Complexity:
    All methods: O(T) where T = number of observations.
"""

import pandas as pd
import numpy as np
from typing import Optional


class DataLoader:
    """
    Zero-boilerplate data pipeline for time series forecasting.

    All methods are static (class methods) — no instantiation needed.
    Returns DataFrames with 'ds' and 'y' columns compatible with
    MasterChronos.fit().
    """

    @staticmethod
    def yfinance(
        ticker: str,
        start: str = '2020-01-01',
        end: Optional[str] = None,
        column: str = 'Close',
    ) -> pd.DataFrame:
        """
        Fetches stock/index data from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g., 'AAPL', '^NSEI').
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD). Defaults to today.
            column: Price column to use ('Close', 'Open', 'High', 'Low', 'Volume').

        Returns:
            DataFrame with 'ds' and 'y' columns.

        Example:
            >>> df = DataLoader.yfinance('AAPL', start='2020-01-01')
            >>> model = MasterChronos()
            >>> model.fit(df)
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for DataLoader.yfinance(). "
                "Install it: pip install chronos-forecast[data]"
            )

        data = yf.download(ticker, start=start, end=end, progress=False)

        if data.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' "
                f"from {start} to {end or 'today'}."
            )

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if column not in data.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available: {list(data.columns)}"
            )

        df = pd.DataFrame({
            'ds': data.index,
            'y': data[column].values,
        }).dropna().reset_index(drop=True)

        return df

    @staticmethod
    def csv(
        path: str,
        ds_col: Optional[str] = None,
        y_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads time series from a CSV file.

        Auto-detects date and value columns if not specified.

        Args:
            path: Path to CSV file.
            ds_col: Name of the date column. Auto-detected if None.
            y_col: Name of the value column. Auto-detected if None.
            **kwargs: Additional arguments passed to pd.read_csv().

        Returns:
            DataFrame with 'ds' and 'y' columns.
        """
        raw = pd.read_csv(path, **kwargs)
        return DataLoader._standardize(raw, ds_col, y_col)

    @staticmethod
    def parquet(
        path: str,
        ds_col: Optional[str] = None,
        y_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads time series from a Parquet file.

        Args:
            path: Path to Parquet file.
            ds_col: Name of the date column.
            y_col: Name of the value column.

        Returns:
            DataFrame with 'ds' and 'y' columns.
        """
        raw = pd.read_parquet(path)
        return DataLoader._standardize(raw, ds_col, y_col)

    @staticmethod
    def dataframe(
        df: pd.DataFrame,
        ds_col: Optional[str] = None,
        y_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardizes an existing DataFrame to 'ds'/'y' format.

        Args:
            df: Input DataFrame.
            ds_col: Name of the date column.
            y_col: Name of the value column.

        Returns:
            DataFrame with 'ds' and 'y' columns.
        """
        return DataLoader._standardize(df.copy(), ds_col, y_col)

    @staticmethod
    def _standardize(
        df: pd.DataFrame,
        ds_col: Optional[str] = None,
        y_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardizes a DataFrame to have 'ds' and 'y' columns.

        Auto-detection heuristics:
            - Date column: looks for 'ds', 'date', 'datetime', 'timestamp',
              or the first datetime-like column.
            - Value column: looks for 'y', 'value', 'target', or the first
              numeric column (after excluding the date column).
        """
        # --- Auto-detect date column ---
        if ds_col is None:
            for candidate in ['ds', 'date', 'datetime', 'timestamp', 'Date',
                              'DS', 'Datetime', 'Timestamp']:
                if candidate in df.columns:
                    ds_col = candidate
                    break

            if ds_col is None:
                # Try to find a datetime column
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col])
                        ds_col = col
                        break
                    except (ValueError, TypeError):
                        continue

            if ds_col is None:
                # Use index if it looks like dates
                try:
                    pd.to_datetime(df.index)
                    df = df.reset_index()
                    ds_col = df.columns[0]
                except (ValueError, TypeError):
                    raise ValueError(
                        "Could not auto-detect date column. "
                        "Specify ds_col parameter."
                    )

        # --- Auto-detect value column ---
        if y_col is None:
            for candidate in ['y', 'value', 'target', 'Y', 'Value', 'Target',
                              'close', 'Close', 'CLOSE']:
                if candidate in df.columns and candidate != ds_col:
                    y_col = candidate
                    break

            if y_col is None:
                # Use first numeric column (not the date column)
                numeric_cols = [
                    c for c in df.select_dtypes(include=[np.number]).columns
                    if c != ds_col
                ]
                if numeric_cols:
                    y_col = numeric_cols[0]
                else:
                    raise ValueError(
                        "Could not auto-detect value column. "
                        "Specify y_col parameter."
                    )

        result = pd.DataFrame({
            'ds': pd.to_datetime(df[ds_col]),
            'y': pd.to_numeric(df[y_col], errors='coerce'),
        }).dropna().sort_values('ds').reset_index(drop=True)

        return result
