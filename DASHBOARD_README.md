# Stock Analysis Dashboard

An interactive Streamlit dashboard that analyzes stock performance using **Quantum Annealing Principal Component Analysis (QAPCA)**.

## Features

### 1. Price & Returns Analysis
- Historical price charts for any stock ticker
- Log returns distribution analysis
- Normalized price comparison with sector peers
- Performance tracking over 6-month period

### 2. PCA Analysis
- **Classical L1-PCA**: Robust principal component analysis using classical optimization
- **Quantum QAPCA-R**: Quantum annealing-based PCA using Neal's simulated annealing sampler
- Visual embeddings showing stock groupings and correlations
- Automatic sector clustering based on price movements

### 3. Sector Correlation
- Interactive correlation heatmap
- Sector contagion analysis
- Identify stocks that move together or in opposition
- Measure shock propagation between sectors

### 4. Performance Metrics
- Annualized volatility calculations
- Sharpe ratio comparisons
- Mean annual returns
- Risk level assessment (Low/Medium/High)

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Activate virtual environment:
```bash
source .venv/bin/activate
```

## Running the Dashboard

```bash
streamlit run stock_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## How to Use

1. **Enter a Ticker**: Type any stock ticker symbol (e.g., AAPL, TSLA, GOOGL) in the sidebar
   - Input is case-insensitive
   - Works with any ticker available on Yahoo Finance

2. **Select Analysis Method**:
   - Classical L1-PCA: Uses scipy's dual annealing
   - Quantum QAPCA-R: Uses quantum-inspired simulated annealing
   - Both: Compare results from both methods

3. **Choose Comparison Stocks**: Select multiple tickers to analyze correlations and sector relationships

4. **Explore the Tabs**:
   - **Price & Returns**: View price history and return distributions
   - **PCA Analysis**: See quantum and classical PCA embeddings
   - **Sector Correlation**: Analyze correlation matrices and contagion risk
   - **Performance Metrics**: Compare volatility, Sharpe ratios, and risk levels

## Understanding the Results

### PCA Embeddings
- Stocks with the **same sign** (both positive or both negative) tend to move together
- Stocks with **opposite signs** tend to move in opposite directions
- **Magnitude** indicates the strength of the relationship

### Groupings
- **Group 1 (Positive)**: Stocks that are positively correlated with the first principal component
- **Group 2 (Negative)**: Stocks that are negatively correlated with the first principal component

### Risk Assessment
- **Low Risk**: Volatility below average
- **Medium Risk**: Volatility within 1.5x average
- **High Risk**: Volatility above 1.5x average

## Example Use Cases

### 1. Sector Contagion Detection
Enter a tech stock (e.g., AAPL) and compare with energy stocks (XOM) to see if market shocks are spreading across sectors.

### 2. Diversification Analysis
Find stocks that are negatively correlated to reduce portfolio risk.

### 3. Performance Prediction
Use PCA groupings to identify which stocks are likely to move together in future market conditions.

### 4. Volatility Comparison
Compare risk levels across different sectors to make informed investment decisions.

## Technical Details

### Algorithms Used

**Classical L1-PCA:**
- Solves: `argmin b^T(-J)b` where J is the covariance matrix
- Uses scipy's `dual_annealing` for optimization
- Robust to outliers due to L1-norm formulation

**Quantum QAPCA-R:**
- Converts covariance matrix to Ising model
- Uses Neal's `SimulatedAnnealingSampler` (quantum-inspired)
- Provides potentially different solutions due to quantum exploration

### Data Sources
- Real-time stock data from Yahoo Finance via `yfinance`
- 6-month historical period for analysis
- Daily log returns for volatility calculations

## Pre-configured Tickers

The dashboard includes pre-configured sector mappings for:

| Ticker | Sector |
|--------|--------|
| AAPL, AMZN, GOOG, MSFT | Technology |
| XOM | Energy |
| GLD | Finance |
| AEP, DUK, SO | Utility |

However, you can analyze **any ticker** available on Yahoo Finance!

## Performance Notes

- Initial data fetch may take 10-30 seconds
- PCA calculations are cached for 1 hour
- Analysis complexity increases with number of comparison stocks
- Quantum annealing uses 100 reads for better accuracy

## Troubleshooting

**"Could not fetch data for ticker"**
- Check if the ticker symbol is correct
- Verify the stock has 6 months of trading history
- Try using the full ticker symbol (e.g., GOOG instead of GOOGL)

**Analysis taking too long**
- Reduce the number of comparison stocks
- The first run is slower due to data fetching
- Subsequent runs use cached data

**Empty charts**
- Ensure comparison stocks are selected
- Check that the ticker has sufficient trading history
- Try refreshing the page

## Built With

- **Streamlit** - Interactive dashboard framework
- **Plotly** - Interactive visualizations
- **yfinance** - Stock data fetching
- **scipy** - Classical optimization (dual annealing)
- **dwave-neal** - Quantum-inspired annealing
- **numpy/pandas** - Data manipulation

## Related Files

- `stock_dashboard.py` - Main dashboard application
- `challenge_2.ipynb` - Original PCA algorithm implementation
- `shaw_circle_challenge.ipynb` - Challenge notebook with detailed explanations

---

**Built for QuantathonV2 Challenge**

For more information about the QAPCA algorithm, see: [Quantum Annealing for Robust Principal Component Analysis](https://arxiv.org/pdf/2501.10431)
