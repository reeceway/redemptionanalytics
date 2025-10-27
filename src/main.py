"""
Main script for analyzing correlation between interest rates and economic growth
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import EconomicDataLoader
from analysis.correlation import CorrelationAnalyzer
from visualization.plots import EconomicPlotter


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def analyze_interest_rate_and_growth():
    """
    Main analysis: Correlation between interest rates and economic growth
    """
    print_section("Economic Correlation Analysis: Interest Rates & Growth")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize components
    print("Initializing data loader...")
    loader = EconomicDataLoader(start_date='2000-01-01')

    print("Initializing correlation analyzer...")
    analyzer = CorrelationAnalyzer(method='pearson')

    print("Initializing plotter...")
    plotter = EconomicPlotter()

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load data
    print_section("Loading Economic Data")

    try:
        print("Loading Federal Funds Rate...")
        fed_funds = loader.get_series('fed_funds_rate', 'Federal Funds Rate')
        print(f"  ✓ Loaded {len(fed_funds)} observations from {fed_funds.index[0]} to {fed_funds.index[-1]}")
    except Exception as e:
        print(f"  ✗ Error loading Federal Funds Rate: {e}")
        print("\nNote: To use real FRED data, you need to:")
        print("  1. Install required packages: pip install fredapi pandas-datareader")
        print("  2. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  3. Set environment variable: export FRED_API_KEY='your_key_here'")
        print("\nUsing sample synthetic data for demonstration...\n")

        # Create synthetic data for demonstration
        dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='M')
        import numpy as np
        np.random.seed(42)

        # Synthetic Federal Funds Rate (declining trend with volatility)
        trend = np.linspace(6.5, 4.5, len(dates))
        fed_funds = pd.Series(
            trend + np.random.randn(len(dates)) * 0.5 +
            np.sin(np.arange(len(dates)) * 0.2) * 1.5,
            index=dates,
            name='Federal Funds Rate'
        )
        fed_funds = fed_funds.clip(lower=0)

    try:
        print("Loading GDP Growth Rate...")
        gdp_growth = loader.get_gdp_growth()
        print(f"  ✓ Loaded {len(gdp_growth)} observations")
    except Exception as e:
        print(f"  ✗ Error loading GDP Growth: {e}")
        print("  Using synthetic data for demonstration...")

        # Synthetic GDP growth (with inverse relationship to rates)
        gdp_growth = pd.Series(
            3.0 - 0.3 * fed_funds + np.random.randn(len(fed_funds)) * 0.8,
            index=fed_funds.index,
            name='GDP Growth Rate'
        )

    try:
        print("Loading Unemployment Rate...")
        unemployment = loader.get_unemployment()
        print(f"  ✓ Loaded {len(unemployment)} observations")
    except Exception as e:
        print(f"  ✗ Error loading Unemployment: {e}")
        print("  Using synthetic data for demonstration...")

        # Synthetic unemployment (correlated with rates)
        unemployment = pd.Series(
            5.5 + 0.2 * fed_funds + np.random.randn(len(fed_funds)) * 0.5,
            index=fed_funds.index,
            name='Unemployment Rate'
        )
        unemployment = unemployment.clip(lower=3, upper=10)

    try:
        print("Loading Inflation Rate...")
        inflation = loader.get_inflation()
        print(f"  ✓ Loaded {len(inflation)} observations")
    except Exception as e:
        print(f"  ✗ Error loading Inflation: {e}")
        print("  Using synthetic data for demonstration...")

        # Synthetic inflation
        inflation = pd.Series(
            2.5 + 0.15 * fed_funds + np.random.randn(len(fed_funds)) * 0.6,
            index=fed_funds.index,
            name='Inflation Rate'
        )

    # Summary statistics
    print_section("Summary Statistics")

    all_data = pd.DataFrame({
        'Federal Funds Rate': fed_funds,
        'GDP Growth': gdp_growth,
        'Unemployment': unemployment,
        'Inflation': inflation
    })

    summary_stats = analyzer.summary_statistics(all_data)
    print(summary_stats.to_string(index=False))

    # Correlation analysis
    print_section("Correlation Analysis: Interest Rates vs Economic Indicators")

    indicators = {
        'GDP Growth': gdp_growth,
        'Unemployment': unemployment,
        'Inflation': inflation
    }

    results = analyzer.analyze_interest_rate_impact(
        fed_funds,
        indicators,
        lag_analysis=True,
        max_lag=12
    )

    print("\nCurrent (Zero-Lag) Correlations:")
    print("-" * 80)
    print(results['current_correlations'].to_string(index=False))

    print("\n\nOptimal Lag Analysis:")
    print("-" * 80)
    if 'optimal_lags' in results:
        print(results['optimal_lags'].to_string(index=False))

    # Create visualizations
    print_section("Generating Visualizations")

    # 1. Time series plot
    print("Creating time series plot...")
    fig1 = plotter.plot_time_series(
        all_data,
        title="Economic Indicators Over Time",
        ylabel="Value (%)"
    )
    fig1.savefig(output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'time_series.png'}")

    # 2. Dual axis: Interest Rate vs GDP Growth
    print("Creating dual-axis plot: Interest Rates vs GDP Growth...")
    fig2 = plotter.plot_dual_axis(
        fed_funds,
        gdp_growth,
        title="Federal Funds Rate vs GDP Growth"
    )
    fig2.savefig(output_dir / 'interest_rate_vs_gdp.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'interest_rate_vs_gdp.png'}")

    # 3. Scatter plots
    print("Creating scatter plots...")
    fig3 = plotter.plot_scatter_with_regression(
        fed_funds,
        gdp_growth,
        title="Correlation: Federal Funds Rate vs GDP Growth"
    )
    fig3.savefig(output_dir / 'scatter_gdp.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'scatter_gdp.png'}")

    fig4 = plotter.plot_scatter_with_regression(
        fed_funds,
        unemployment,
        title="Correlation: Federal Funds Rate vs Unemployment"
    )
    fig4.savefig(output_dir / 'scatter_unemployment.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'scatter_unemployment.png'}")

    # 4. Correlation matrix
    print("Creating correlation matrix...")
    corr_matrix, pval_matrix = analyzer.correlation_matrix(all_data)
    fig5 = plotter.plot_correlation_matrix(
        corr_matrix,
        pval_matrix,
        title="Correlation Matrix: Interest Rates & Economic Indicators"
    )
    fig5.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'correlation_matrix.png'}")

    # 5. Lagged correlation plots
    if 'lagged_correlations' in results:
        print("Creating lagged correlation plots...")
        for indicator_name, lag_df in results['lagged_correlations'].items():
            fig = plotter.plot_lagged_correlation(lag_df, indicator_name)
            filename = f"lagged_correlation_{indicator_name.lower().replace(' ', '_')}.png"
            fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_dir / filename}")

    # 6. Rolling correlation
    print("Creating rolling correlation plot...")
    rolling_corr = analyzer.rolling_correlation(fed_funds, gdp_growth, window=12)
    fig6 = plotter.plot_rolling_correlation(
        rolling_corr,
        series1_name="Federal Funds Rate",
        series2_name="GDP Growth",
        window=12
    )
    fig6.savefig(output_dir / 'rolling_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'rolling_correlation.png'}")

    # 7. Correlation summary
    print("Creating correlation summary plot...")
    fig7 = plotter.plot_correlation_summary(results['current_correlations'])
    fig7.savefig(output_dir / 'correlation_summary.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'correlation_summary.png'}")

    # 8. Dashboard
    print("Creating comprehensive dashboard...")
    fig8 = plotter.create_dashboard(
        fed_funds,
        gdp_growth,
        results
    )
    fig8.savefig(output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'dashboard.png'}")

    # Summary report
    print_section("Analysis Summary")

    print("\nKey Findings:")
    print("-" * 80)

    # GDP Growth
    gdp_corr = results['current_correlations'][
        results['current_correlations']['indicator'] == 'GDP Growth'
    ]['correlation'].values[0]
    gdp_pval = results['current_correlations'][
        results['current_correlations']['indicator'] == 'GDP Growth'
    ]['p_value'].values[0]

    print(f"\n1. Interest Rates vs GDP Growth:")
    print(f"   - Correlation: {gdp_corr:.3f}")
    print(f"   - P-value: {gdp_pval:.4f}")
    print(f"   - Interpretation: {analyzer._interpret_correlation(gdp_corr)}")
    if gdp_pval < 0.05:
        print(f"   - Statistical significance: YES (p < 0.05)")
    else:
        print(f"   - Statistical significance: NO (p >= 0.05)")

    # Unemployment
    unemp_corr = results['current_correlations'][
        results['current_correlations']['indicator'] == 'Unemployment'
    ]['correlation'].values[0]

    print(f"\n2. Interest Rates vs Unemployment:")
    print(f"   - Correlation: {unemp_corr:.3f}")
    print(f"   - Interpretation: {analyzer._interpret_correlation(unemp_corr)}")

    # Inflation
    infl_corr = results['current_correlations'][
        results['current_correlations']['indicator'] == 'Inflation'
    ]['correlation'].values[0]

    print(f"\n3. Interest Rates vs Inflation:")
    print(f"   - Correlation: {infl_corr:.3f}")
    print(f"   - Interpretation: {analyzer._interpret_correlation(infl_corr)}")

    print("\n" + "=" * 80)
    print("Analysis complete! Check the 'outputs' folder for visualizations.")
    print("=" * 80 + "\n")

    plt.close('all')


if __name__ == "__main__":
    try:
        analyze_interest_rate_and_growth()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
