"""
Visualization tools for economic correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings


class EconomicPlotter:
    """Create visualizations for economic correlation analysis"""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the plotter

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        # Set style
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                pass  # Use default style

        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)

    def plot_time_series(self, data: Union[pd.Series, pd.DataFrame],
                        title: str = "Economic Time Series",
                        ylabel: str = "Value",
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot time series data

        Args:
            data: Series or DataFrame with time series
            title: Plot title
            ylabel: Y-axis label
            figsize: Figure size (uses default if None)

        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(data, pd.Series):
            ax.plot(data.index, data.values, linewidth=2, label=data.name)
        elif isinstance(data, pd.DataFrame):
            for i, col in enumerate(data.columns):
                ax.plot(data.index, data[col], linewidth=2,
                       label=col, color=self.color_palette[i % len(self.color_palette)])

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_dual_axis(self, left_series: pd.Series, right_series: pd.Series,
                      title: str = "Dual Axis Comparison",
                      figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot two series on dual y-axes

        Args:
            left_series: Series for left y-axis
            right_series: Series for right y-axis
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.figsize
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot first series
        color1 = self.color_palette[0]
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel(left_series.name or 'Left Axis', color=color1, fontsize=12)
        ax1.plot(left_series.index, left_series.values, color=color1, linewidth=2,
                label=left_series.name)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Create second y-axis
        ax2 = ax1.twinx()
        color2 = self.color_palette[1]
        ax2.set_ylabel(right_series.name or 'Right Axis', color=color2, fontsize=12)
        ax2.plot(right_series.index, right_series.values, color=color2, linewidth=2,
                label=right_series.name)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Title
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame,
                               pval_matrix: Optional[pd.DataFrame] = None,
                               title: str = "Correlation Matrix",
                               figsize: Optional[Tuple[int, int]] = None,
                               annot: bool = True,
                               mask_insignificant: bool = True,
                               significance_level: float = 0.05) -> plt.Figure:
        """
        Plot correlation matrix as a heatmap

        Args:
            corr_matrix: Correlation matrix
            pval_matrix: P-value matrix (optional, for significance masking)
            title: Plot title
            figsize: Figure size
            annot: Whether to annotate cells with values
            mask_insignificant: Whether to mask insignificant correlations
            significance_level: Significance level for masking

        Returns:
            matplotlib Figure
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)

        # Create mask for insignificant values
        mask = None
        if mask_insignificant and pval_matrix is not None:
            mask = pval_matrix > significance_level

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=annot, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax, mask=mask,
                   fmt='.2f' if annot else None)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        return fig

    def plot_scatter_with_regression(self, x: pd.Series, y: pd.Series,
                                    title: Optional[str] = None,
                                    figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot scatter plot with regression line

        Args:
            x: X-axis series
            y: Y-axis series
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Align data and remove NaN
        df = pd.DataFrame({'x': x, 'y': y}).dropna()

        # Scatter plot
        ax.scatter(df['x'], df['y'], alpha=0.6, s=50, color=self.color_palette[0])

        # Regression line
        if len(df) > 2:
            z = np.polyfit(df['x'], df['y'], 1)
            p = np.poly1d(z)
            ax.plot(df['x'], p(df['x']), "r-", linewidth=2, alpha=0.8,
                   label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

            # Calculate R-squared
            from scipy.stats import pearsonr
            r, pval = pearsonr(df['x'], df['y'])
            ax.text(0.05, 0.95, f'R² = {r**2:.3f}\np-value = {pval:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Labels
        title = title or f'{y.name} vs {x.name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x.name or 'X', fontsize=12)
        ax.set_ylabel(y.name or 'Y', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        plt.tight_layout()
        return fig

    def plot_lagged_correlation(self, lag_df: pd.DataFrame,
                               indicator_name: str = "Indicator",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot correlation vs lag

        Args:
            lag_df: DataFrame with 'lag', 'correlation', and 'p_value' columns
            indicator_name: Name of the indicator
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Plot correlation vs lag
        colors = ['red' if p < 0.05 else 'gray' for p in lag_df['p_value']]
        ax.bar(lag_df['lag'], lag_df['correlation'], color=colors, alpha=0.7)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Highlight optimal lag
        max_idx = lag_df['correlation'].abs().idxmax()
        if not np.isnan(max_idx):
            optimal_lag = lag_df.loc[max_idx, 'lag']
            optimal_corr = lag_df.loc[max_idx, 'correlation']
            ax.axvline(x=optimal_lag, color='green', linestyle='--', linewidth=2,
                      label=f'Optimal lag: {int(optimal_lag)} (r={optimal_corr:.3f})')

        # Labels and title
        ax.set_xlabel('Lag (periods)', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title(f'Lagged Correlation: Interest Rate vs {indicator_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add note about significance
        ax.text(0.02, 0.98, 'Red bars: p < 0.05 (significant)',
               transform=ax.transAxes, verticalalignment='top',
               fontsize=9, style='italic')

        plt.tight_layout()
        return fig

    def plot_rolling_correlation(self, rolling_corr: pd.Series,
                                 series1_name: str = "Series 1",
                                 series2_name: str = "Series 2",
                                 window: int = 12,
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot rolling correlation over time

        Args:
            rolling_corr: Series of rolling correlation values
            series1_name: Name of first series
            series2_name: Name of second series
            window: Window size used for rolling correlation
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        # Plot rolling correlation
        ax.plot(rolling_corr.index, rolling_corr.values, linewidth=2,
               color=self.color_palette[2])

        # Add horizontal lines for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong positive')
        ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Strong negative')

        # Fill areas
        ax.fill_between(rolling_corr.index, 0, rolling_corr.values,
                        where=rolling_corr.values >= 0, alpha=0.3,
                        color='green', interpolate=True)
        ax.fill_between(rolling_corr.index, 0, rolling_corr.values,
                        where=rolling_corr.values < 0, alpha=0.3,
                        color='red', interpolate=True)

        # Labels
        ax.set_title(f'{window}-Period Rolling Correlation: {series1_name} vs {series2_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_ylim(-1, 1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_summary(self, results_df: pd.DataFrame,
                                figsize: Optional[Tuple[int, int]] = None,
                                top_n: int = 10) -> plt.Figure:
        """
        Plot horizontal bar chart of correlations

        Args:
            results_df: DataFrame with 'indicator' and 'correlation' columns
            figsize: Figure size
            top_n: Number of top correlations to show

        Returns:
            matplotlib Figure
        """
        figsize = figsize or (10, 6)
        fig, ax = plt.subplots(figsize=figsize)

        # Get top N by absolute correlation
        df_sorted = results_df.copy()
        df_sorted['abs_corr'] = df_sorted['correlation'].abs()
        df_sorted = df_sorted.nlargest(top_n, 'abs_corr')

        # Color by positive/negative
        colors = ['green' if c > 0 else 'red' for c in df_sorted['correlation']]

        # Create horizontal bar chart
        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted['correlation'], color=colors, alpha=0.7)

        # Add significance markers
        if 'significant' in df_sorted.columns:
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                if row['significant']:
                    ax.text(row['correlation'], i, ' *', va='center',
                           fontsize=12, fontweight='bold')

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['indicator'])
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title(f'Top {top_n} Correlations with Interest Rate',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive correlation'),
            Patch(facecolor='red', alpha=0.7, label='Negative correlation')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        return fig

    def create_dashboard(self, interest_rate: pd.Series,
                        indicator: pd.Series,
                        correlation_results: Dict,
                        figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots

        Args:
            interest_rate: Interest rate time series
            indicator: Economic indicator time series
            correlation_results: Results from CorrelationAnalyzer
            figsize: Figure size

        Returns:
            matplotlib Figure with multiple subplots
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Dual axis time series
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()

        color1 = self.color_palette[0]
        color2 = self.color_palette[1]

        ax1.plot(interest_rate.index, interest_rate.values, color=color1,
                linewidth=2, label=interest_rate.name)
        ax1.set_ylabel(interest_rate.name, color=color1, fontsize=11)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax1_twin.plot(indicator.index, indicator.values, color=color2,
                     linewidth=2, label=indicator.name)
        ax1_twin.set_ylabel(indicator.name, color=color2, fontsize=11)
        ax1_twin.tick_params(axis='y', labelcolor=color2)

        ax1.set_title('Time Series Comparison', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        df = pd.DataFrame({'x': interest_rate, 'y': indicator}).dropna()
        ax2.scatter(df['x'], df['y'], alpha=0.6, s=30, color=self.color_palette[0])

        if len(df) > 2:
            z = np.polyfit(df['x'], df['y'], 1)
            p = np.poly1d(z)
            ax2.plot(df['x'], p(df['x']), "r-", linewidth=2, alpha=0.8)

            from scipy.stats import pearsonr
            r, pval = pearsonr(df['x'], df['y'])
            ax2.text(0.05, 0.95, f'R² = {r**2:.3f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)

        ax2.set_xlabel(interest_rate.name, fontsize=10)
        ax2.set_ylabel(indicator.name, fontsize=10)
        ax2.set_title('Correlation Scatter Plot', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Lagged correlation
        if 'lagged_correlations' in correlation_results:
            ax3 = fig.add_subplot(gs[1, 1])
            lag_df = list(correlation_results['lagged_correlations'].values())[0]

            colors = ['red' if p < 0.05 else 'gray' for p in lag_df['p_value']]
            ax3.bar(lag_df['lag'], lag_df['correlation'], color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            ax3.set_xlabel('Lag (periods)', fontsize=10)
            ax3.set_ylabel('Correlation', fontsize=10)
            ax3.set_title('Lagged Correlation Analysis', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Distribution plots
        ax4 = fig.add_subplot(gs[2, 0])
        interest_rate.dropna().hist(bins=30, alpha=0.7, color=color1, ax=ax4)
        ax4.set_xlabel(interest_rate.name, fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title(f'{interest_rate.name} Distribution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 1])
        indicator.dropna().hist(bins=30, alpha=0.7, color=color2, ax=ax5)
        ax5.set_xlabel(indicator.name, fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title(f'{indicator.name} Distribution', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        return fig
