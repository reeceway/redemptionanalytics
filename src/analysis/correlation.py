"""
Correlation analysis for economic indicators and interest rates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings


class CorrelationAnalyzer:
    """Analyze correlations between interest rates and economic indicators"""

    def __init__(self, method: str = 'pearson'):
        """
        Initialize the correlation analyzer

        Args:
            method: Correlation method - 'pearson' or 'spearman'
        """
        if method not in ['pearson', 'spearman']:
            raise ValueError("method must be 'pearson' or 'spearman'")
        self.method = method

    def calculate_correlation(self, x: pd.Series, y: pd.Series,
                            min_periods: int = 30) -> Tuple[float, float]:
        """
        Calculate correlation between two series

        Args:
            x: First time series
            y: Second time series
            min_periods: Minimum number of observations required

        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        # Align the series and drop NaN
        df = pd.DataFrame({'x': x, 'y': y}).dropna()

        if len(df) < min_periods:
            return np.nan, np.nan

        if self.method == 'pearson':
            corr, pval = pearsonr(df['x'], df['y'])
        else:
            corr, pval = spearmanr(df['x'], df['y'])

        return corr, pval

    def correlation_matrix(self, df: pd.DataFrame,
                          min_periods: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix for all columns in DataFrame

        Args:
            df: DataFrame with multiple time series
            min_periods: Minimum number of observations required

        Returns:
            Tuple of (correlation matrix, p-value matrix)
        """
        n = len(df.columns)
        corr_matrix = pd.DataFrame(np.zeros((n, n)),
                                   index=df.columns,
                                   columns=df.columns)
        pval_matrix = pd.DataFrame(np.zeros((n, n)),
                                   index=df.columns,
                                   columns=df.columns)

        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i == j:
                    corr_matrix.iloc[i, j] = 1.0
                    pval_matrix.iloc[i, j] = 0.0
                else:
                    corr, pval = self.calculate_correlation(df[col1], df[col2], min_periods)
                    corr_matrix.iloc[i, j] = corr
                    pval_matrix.iloc[i, j] = pval

        return corr_matrix, pval_matrix

    def lagged_correlation(self, x: pd.Series, y: pd.Series,
                          max_lag: int = 12,
                          min_periods: int = 30) -> pd.DataFrame:
        """
        Calculate correlation at different time lags

        Args:
            x: First time series (independent variable)
            y: Second time series (dependent variable)
            max_lag: Maximum lag periods to test (both positive and negative)
            min_periods: Minimum number of observations required

        Returns:
            DataFrame with lag, correlation, and p-value
        """
        results = []

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # y leads x (shift x forward)
                x_shifted = x.shift(-lag)
                y_aligned = y
            else:
                # x leads y (shift y forward)
                x_shifted = x
                y_aligned = y.shift(lag)

            corr, pval = self.calculate_correlation(x_shifted, y_aligned, min_periods)

            results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05 if not np.isnan(pval) else False
            })

        df = pd.DataFrame(results)
        return df

    def rolling_correlation(self, x: pd.Series, y: pd.Series,
                           window: int = 12,
                           min_periods: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling (moving window) correlation

        Args:
            x: First time series
            y: Second time series
            window: Size of the rolling window
            min_periods: Minimum periods for calculation (default: window)

        Returns:
            Series of rolling correlations
        """
        if min_periods is None:
            min_periods = window

        # Align the series
        df = pd.DataFrame({'x': x, 'y': y})

        # Calculate rolling correlation
        if self.method == 'pearson':
            rolling_corr = df['x'].rolling(window=window, min_periods=min_periods).corr(df['y'])
        else:
            # For Spearman, we need to calculate manually
            def spearman_corr(window_data):
                if len(window_data.dropna()) < min_periods:
                    return np.nan
                try:
                    corr, _ = spearmanr(window_data['x'], window_data['y'])
                    return corr
                except:
                    return np.nan

            rolling_corr = df.rolling(window=window).apply(
                lambda w: spearman_corr(w.dropna()), raw=False
            )['x']

        return rolling_corr

    def analyze_interest_rate_impact(self, interest_rate: pd.Series,
                                    indicators: Dict[str, pd.Series],
                                    lag_analysis: bool = True,
                                    max_lag: int = 12) -> Dict[str, pd.DataFrame]:
        """
        Analyze the correlation between an interest rate and multiple economic indicators

        Args:
            interest_rate: Interest rate time series
            indicators: Dictionary of indicator name to Series
            lag_analysis: Whether to perform lagged correlation analysis
            max_lag: Maximum lag for lag analysis

        Returns:
            Dictionary with analysis results
        """
        results = {}

        # Current (no lag) correlations
        current_corrs = []
        for name, series in indicators.items():
            corr, pval = self.calculate_correlation(interest_rate, series)
            current_corrs.append({
                'indicator': name,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05 if not np.isnan(pval) else False,
                'strength': self._interpret_correlation(corr)
            })

        results['current_correlations'] = pd.DataFrame(current_corrs).sort_values(
            'correlation', key=abs, ascending=False
        )

        # Lagged correlations for each indicator
        if lag_analysis:
            lagged_results = {}
            for name, series in indicators.items():
                lag_df = self.lagged_correlation(interest_rate, series, max_lag=max_lag)
                lagged_results[name] = lag_df

            results['lagged_correlations'] = lagged_results

            # Find optimal lags
            optimal_lags = []
            for name, lag_df in lagged_results.items():
                # Find lag with highest absolute correlation
                max_idx = lag_df['correlation'].abs().idxmax()
                if not np.isnan(max_idx):
                    optimal = lag_df.loc[max_idx]
                    optimal_lags.append({
                        'indicator': name,
                        'optimal_lag': int(optimal['lag']),
                        'correlation': optimal['correlation'],
                        'p_value': optimal['p_value'],
                        'lag_interpretation': self._interpret_lag(int(optimal['lag']))
                    })

            results['optimal_lags'] = pd.DataFrame(optimal_lags).sort_values(
                'correlation', key=abs, ascending=False
            )

        return results

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret the strength of a correlation coefficient"""
        if np.isnan(corr):
            return "undefined"

        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"

        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"

        return f"{strength} {direction}"

    def _interpret_lag(self, lag: int) -> str:
        """Interpret the meaning of a lag value"""
        if lag == 0:
            return "contemporaneous (no lag)"
        elif lag > 0:
            return f"interest rate leads by {lag} periods"
        else:
            return f"indicator leads interest rate by {-lag} periods"

    def summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for all series in DataFrame

        Args:
            df: DataFrame with time series

        Returns:
            DataFrame with summary statistics
        """
        stats_list = []

        for col in df.columns:
            series = df[col].dropna()
            stats_list.append({
                'variable': col,
                'count': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            })

        return pd.DataFrame(stats_list)

    def test_stationarity(self, series: pd.Series) -> Dict[str, float]:
        """
        Perform Augmented Dickey-Fuller test for stationarity

        Args:
            series: Time series to test

        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            series_clean = series.dropna()
            result = adfuller(series_clean, autolag='AIC')

            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'lags_used': result[2],
                'n_observations': result[3],
                'is_stationary': result[1] < 0.05,
                'critical_value_1%': result[4]['1%'],
                'critical_value_5%': result[4]['5%'],
                'critical_value_10%': result[4]['10%']
            }
        except ImportError:
            warnings.warn("statsmodels not available, skipping stationarity test")
            return {}
        except Exception as e:
            warnings.warn(f"Stationarity test failed: {e}")
            return {}
