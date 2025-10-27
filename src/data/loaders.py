"""
Data loader for economic indicators and interest rates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pandas_datareader import data as pdr
    from pandas_datareader.fred import FredReader
except ImportError:
    pdr = None
    FredReader = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None


class EconomicDataLoader:
    """Load economic data from various sources including FRED and Yahoo Finance"""

    # FRED Series IDs for common economic indicators
    FRED_SERIES = {
        # Interest Rates
        'fed_funds_rate': 'FEDFUNDS',           # Federal Funds Effective Rate
        'treasury_10y': 'DGS10',                # 10-Year Treasury Constant Maturity Rate
        'treasury_2y': 'DGS2',                  # 2-Year Treasury Constant Maturity Rate
        'treasury_5y': 'DGS5',                  # 5-Year Treasury Constant Maturity Rate
        'prime_rate': 'DPRIME',                 # Bank Prime Loan Rate

        # Economic Growth Indicators
        'gdp': 'GDP',                           # Gross Domestic Product
        'gdp_growth': 'A191RL1Q225SBEA',       # Real GDP Growth Rate
        'real_gdp': 'GDPC1',                    # Real Gross Domestic Product

        # Employment & Labor
        'unemployment': 'UNRATE',               # Unemployment Rate
        'employment': 'PAYEMS',                 # Total Nonfarm Payrolls
        'labor_force_participation': 'CIVPART', # Labor Force Participation Rate

        # Inflation
        'cpi': 'CPIAUCSL',                      # Consumer Price Index
        'core_cpi': 'CPILFESL',                 # Core CPI (excluding food & energy)
        'pce': 'PCE',                           # Personal Consumption Expenditures
        'inflation_rate': 'FPCPITOTLZGUSA',     # Inflation Rate

        # Consumer & Business
        'consumer_sentiment': 'UMCSENT',        # University of Michigan Consumer Sentiment
        'retail_sales': 'RSXFS',                # Retail Sales
        'industrial_production': 'INDPRO',      # Industrial Production Index
        'capacity_utilization': 'TCU',          # Capacity Utilization

        # Housing
        'housing_starts': 'HOUST',              # Housing Starts
        'home_prices': 'CSUSHPISA',            # Case-Shiller Home Price Index
    }

    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                 fred_api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the data loader

        Args:
            start_date: Start date in 'YYYY-MM-DD' format (default: 20 years ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            fred_api_key: FRED API key (optional, will check environment)
            cache_dir: Directory to cache downloaded data
        """
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*20)).strftime('%Y-%m-%d')

        # Try to get FRED API key
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred_client = None
        if Fred and self.fred_api_key:
            try:
                self.fred_client = Fred(api_key=self.fred_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize FRED client: {e}")

        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / 'data' / 'raw'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_series(self, series_id: str, name: Optional[str] = None) -> pd.Series:
        """
        Fetch a single economic time series from FRED

        Args:
            series_id: FRED series ID or key from FRED_SERIES
            name: Optional name for the series

        Returns:
            pd.Series with the economic data
        """
        # Check if series_id is a key in our FRED_SERIES dict
        actual_series_id = self.FRED_SERIES.get(series_id, series_id)
        series_name = name or series_id

        # Try using fredapi first (better API)
        if self.fred_client:
            try:
                data = self.fred_client.get_series(actual_series_id,
                                                   observation_start=self.start_date,
                                                   observation_end=self.end_date)
                data.name = series_name
                return data
            except Exception as e:
                print(f"Warning: fredapi failed for {series_id}: {e}")

        # Fall back to pandas_datareader
        if FredReader:
            try:
                data = pdr.DataReader(actual_series_id, 'fred', self.start_date, self.end_date)
                if isinstance(data, pd.DataFrame):
                    data = data.iloc[:, 0]
                data.name = series_name
                return data
            except Exception as e:
                print(f"Warning: pandas_datareader failed for {series_id}: {e}")

        raise ValueError(f"Could not load series {series_id}. Please install fredapi or pandas_datareader, "
                        f"and ensure you have a valid FRED API key.")

    def get_interest_rates(self) -> pd.DataFrame:
        """
        Get major interest rate indicators

        Returns:
            DataFrame with multiple interest rate series
        """
        rates = {}
        rate_series = ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'treasury_5y', 'prime_rate']

        for rate in rate_series:
            try:
                rates[rate] = self.get_series(rate)
            except Exception as e:
                print(f"Could not load {rate}: {e}")

        if not rates:
            raise ValueError("Could not load any interest rate data")

        df = pd.DataFrame(rates)
        return df

    def get_gdp_growth(self) -> pd.Series:
        """Get GDP growth rate"""
        try:
            return self.get_series('gdp_growth', 'GDP Growth Rate')
        except:
            # If growth rate not available, calculate from GDP
            try:
                gdp = self.get_series('real_gdp')
                growth = gdp.pct_change(periods=4) * 100  # Year-over-year growth
                growth.name = 'GDP Growth Rate'
                return growth
            except Exception as e:
                raise ValueError(f"Could not load GDP growth data: {e}")

    def get_unemployment(self) -> pd.Series:
        """Get unemployment rate"""
        return self.get_series('unemployment', 'Unemployment Rate')

    def get_inflation(self) -> pd.Series:
        """Get inflation rate (CPI year-over-year change)"""
        try:
            return self.get_series('inflation_rate', 'Inflation Rate')
        except:
            # Calculate from CPI
            try:
                cpi = self.get_series('cpi')
                inflation = cpi.pct_change(periods=12) * 100  # Year-over-year
                inflation.name = 'Inflation Rate'
                return inflation
            except Exception as e:
                raise ValueError(f"Could not load inflation data: {e}")

    def get_all_indicators(self) -> Dict[str, pd.Series]:
        """
        Get all major economic indicators

        Returns:
            Dictionary of indicator name to Series
        """
        indicators = {}

        # Get interest rates as separate series
        try:
            rates_df = self.get_interest_rates()
            for col in rates_df.columns:
                indicators[col] = rates_df[col]
        except Exception as e:
            print(f"Warning: Could not load interest rates: {e}")

        # Get growth indicators
        for name, getter in [
            ('gdp_growth', self.get_gdp_growth),
            ('unemployment', self.get_unemployment),
            ('inflation', self.get_inflation),
        ]:
            try:
                indicators[name] = getter()
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

        # Get additional indicators
        additional = ['consumer_sentiment', 'industrial_production', 'retail_sales']
        for indicator in additional:
            try:
                indicators[indicator] = self.get_series(indicator)
            except Exception as e:
                print(f"Warning: Could not load {indicator}: {e}")

        return indicators

    def create_aligned_dataframe(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a DataFrame with aligned (same dates) economic indicators

        Args:
            indicators: List of indicator names to include (uses all if None)

        Returns:
            DataFrame with aligned time series
        """
        all_indicators = self.get_all_indicators()

        if indicators:
            all_indicators = {k: v for k, v in all_indicators.items() if k in indicators}

        df = pd.DataFrame(all_indicators)

        # Drop rows where all values are NaN
        df = df.dropna(how='all')

        return df
