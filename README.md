# Redemption Analytics

A comprehensive economic correlation analysis platform for studying the relationship between interest rates and economic growth indicators.

## Overview

This project provides tools to analyze correlations between:
- Various interest rate measures (Federal Funds Rate, Treasury yields, etc.)
- Economic growth indicators (GDP growth, employment, inflation, etc.)
- Market performance metrics

## Features

- **Data Collection**: Automated retrieval from FRED (Federal Reserve Economic Data), Yahoo Finance, and other sources
- **Correlation Analysis**: Statistical correlation between interest rates and economic indicators
- **Time Series Analysis**: Temporal relationship analysis including lead-lag effects
- **Visualization**: Interactive charts and correlation matrices
- **Reporting**: Automated analysis reports with statistical significance

## Project Structure

```
redemptionanalytics/
├── src/
│   ├── data/           # Data loading and processing
│   ├── analysis/       # Statistical analysis modules
│   └── visualization/  # Plotting and visualization
├── data/              # Data directory (raw and processed)
├── notebooks/         # Jupyter notebooks for exploration
└── tests/            # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd redemptionanalytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional, for FRED API):
```bash
cp .env.example .env
# Edit .env and add your FRED_API_KEY
```

## Quick Start

Run the main correlation analysis:

```bash
python -m src.main
```

Or use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/correlation_analysis.ipynb
```

## Data Sources

- **FRED** (Federal Reserve Economic Data): GDP, inflation, unemployment, interest rates
- **Yahoo Finance**: Market indices and financial data
- **U.S. Treasury**: Treasury yield curves

## Analysis Methods

- Pearson and Spearman correlation coefficients
- Time-lagged correlation analysis
- Rolling correlation windows
- Statistical significance testing
- Granger causality tests

## Example Usage

```python
from src.data.loaders import EconomicDataLoader
from src.analysis.correlation import CorrelationAnalyzer

# Load data
loader = EconomicDataLoader()
interest_rates = loader.get_interest_rates()
gdp_growth = loader.get_gdp_growth()

# Analyze correlation
analyzer = CorrelationAnalyzer()
results = analyzer.analyze(interest_rates, gdp_growth)
results.plot()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
