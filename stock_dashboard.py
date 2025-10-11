import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import dual_annealing
import neal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Quantum PCA Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sector mapping
SECTOR_MAP = {
    'AAPL': 'Technology',
    'AMZN': 'Technology',
    'GOOG': 'Technology',
    'MSFT': 'Technology',
    'XOM': 'Energy',
    'GLD': 'Finance',
    'AEP': 'Utility',
    'DUK': 'Utility',
    'SO': 'Utility'
}

# Available tickers for analysis
ALL_TICKERS = list(SECTOR_MAP.keys())

# ============= PCA FUNCTIONS FROM CHALLENGE =============

def solve_covariance_matrix(sample_data):
    """Calculate covariance matrix for given sample data"""
    stacked_data = np.vstack(sample_data.values)
    return stacked_data @ stacked_data.T, stacked_data.T

def l1_objective(b, J):
    """Compute the L1 PCA objective function."""
    return b.T @ (-J) @ b

def Phi(T):
    """Returns nearest orthonormal matrix via SVD"""
    U, _, Vt = np.linalg.svd(T, full_matrices=False)
    return U @ Vt

def solve_l1_classical_component(J):
    """Solve for a single L1 PCA component using simulated annealing."""
    n = J.shape[0]
    bounds = [(-1, 1) for _ in range(n)]
    x0 = np.random.uniform(-1, 1, n)
    result = dual_annealing(l1_objective, bounds, args=(J,), x0=x0, maxiter=100)

    b_opt = result.x
    r_norm_sqrd = b_opt.T @ J @ b_opt
    bbT = np.outer(b_opt, b_opt)

    J_new = J - ((2/r_norm_sqrd) * J @ bbT @ J) + ((J @ bbT @ J @ bbT @ J) / (r_norm_sqrd**2))
    return b_opt, J_new

def convert_J_to_ising_model(X):
    """Converts covariance matrix J into dict of Ising Model couplings."""
    ising_model = {}
    for i in range(len(X)):
        for j in range(len(X)):
            if i < j:
                ising_model[(i, j)] = -X[i, j]
    return ising_model

def solve_l1_qapca_r_component(J):
    """Solve for a single L1 QAPCA component using quantum annealing."""
    sampler = neal.SimulatedAnnealingSampler()
    h = {i: 0 for i in range(J.shape[0])}
    J_ising = convert_J_to_ising_model(J)

    sampleset = sampler.sample_ising(h, J_ising, num_reads=100)
    best_sample = sampleset.first.sample

    b_opt = np.array([best_sample[i] for i in range(J.shape[0])])
    r_norm_sqrd = b_opt.T @ J @ b_opt
    bbT = np.outer(b_opt, b_opt)

    J_new = J - ((2/r_norm_sqrd) * J @ bbT @ J) + ((J @ bbT @ J @ bbT @ J) / (r_norm_sqrd**2))
    return b_opt, J_new

def do_l1_pca(sample_data, K, get_component_func=solve_l1_classical_component):
    """Gets K principal components"""
    J, X = solve_covariance_matrix(sample_data)
    components = []

    for k in range(K):
        r, J = get_component_func(J)
        components.append(r)

    Bopt = np.vstack(components).T
    X_Bopt = X @ Bopt
    R_L1 = Phi(X_Bopt)
    emb = R_L1.T @ X
    return emb, components

# ============= DATA FETCHING FUNCTIONS =============

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker.upper())
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_multiple_stocks(tickers, period='6mo'):
    """Fetch data for multiple stocks"""
    try:
        data = yf.download(tickers, period=period, group_by='ticker', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            # Restructure to get Close prices
            close_data = pd.DataFrame()
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    close_data[ticker] = data[ticker]['Close']
            return close_data
        return data
    except:
        return None

# ============= ANALYSIS FUNCTIONS =============

def calculate_log_returns(data):
    """Calculate log returns from price data"""
    if isinstance(data, pd.DataFrame):
        log_returns = np.log(data / data.shift(1)).dropna()
    else:
        log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def analyze_stock_with_sector(ticker, comparison_tickers):
    """Analyze stock performance against sector peers"""
    all_tickers = [ticker] + comparison_tickers

    # Fetch data
    data = fetch_multiple_stocks(all_tickers, period='6mo')
    if data is None or data.empty:
        return None

    # Calculate log returns
    log_returns = calculate_log_returns(data)

    # Prepare data for PCA
    log_return_vectors = {}
    for t in all_tickers:
        if t in log_returns.columns:
            returns = log_returns[t].dropna().values
            if len(returns) > 0:
                log_return_vectors[t] = returns

    # Convert to pandas Series for PCA
    log_return_series = pd.Series(log_return_vectors)

    # Perform Classical PCA
    try:
        classical_emb, classical_comp = do_l1_pca(log_return_series, 1, solve_l1_classical_component)
    except:
        classical_emb = None

    # Perform Quantum PCA
    try:
        quantum_emb, quantum_comp = do_l1_pca(log_return_series, 1, solve_l1_qapca_r_component)
    except:
        quantum_emb = None

    return {
        'data': data,
        'log_returns': log_returns,
        'tickers': list(log_return_vectors.keys()),
        'classical_embedding': classical_emb,
        'quantum_embedding': quantum_emb,
        'classical_components': classical_comp,
        'quantum_components': quantum_comp
    }

def calculate_sector_correlations(log_returns, sector_map):
    """Calculate correlation matrix grouped by sectors"""
    correlation_matrix = log_returns.corr()
    return correlation_matrix

def calculate_volatility(log_returns):
    """Calculate annualized volatility"""
    return log_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(log_returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    mean_return = log_returns.mean() * 252
    volatility = calculate_volatility(log_returns)
    return (mean_return - risk_free_rate) / volatility

# ============= VISUALIZATION FUNCTIONS =============

def plot_price_history(data, ticker):
    """Plot price history"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=ticker,
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title=f'{ticker} Price History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def plot_returns_distribution(log_returns, ticker):
    """Plot returns distribution"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=log_returns,
        nbinsx=50,
        name=ticker,
        marker_color='#2ca02c'
    ))
    fig.update_layout(
        title=f'{ticker} Log Returns Distribution',
        xaxis_title='Log Returns',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    return fig

def plot_correlation_heatmap(correlation_matrix):
    """Plot correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(
        title='Stock Correlation Matrix',
        template='plotly_white',
        height=500
    )
    return fig

def plot_pca_embeddings(embeddings, tickers, title="PCA Embeddings"):
    """Plot PCA embeddings as bar chart"""
    if embeddings is None:
        return None

    fig = go.Figure()
    colors = ['#1f77b4' if e >= 0 else '#ff7f0e' for e in embeddings[0]]

    fig.add_trace(go.Bar(
        x=tickers,
        y=embeddings[0],
        marker_color=colors,
        text=embeddings[0].round(3),
        textposition='outside'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Ticker',
        yaxis_title='PCA Component Value',
        template='plotly_white',
        showlegend=False
    )
    return fig

def plot_sector_comparison(data, ticker, sector_tickers):
    """Plot normalized price comparison"""
    fig = go.Figure()

    # Normalize prices to 100
    for t in [ticker] + sector_tickers:
        if t in data.columns:
            normalized = (data[t] / data[t].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=data.index,
                y=normalized,
                mode='lines',
                name=t,
                line=dict(width=3 if t == ticker else 1)
            ))

    fig.update_layout(
        title='Normalized Price Comparison (Base = 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    return fig

def plot_volatility_comparison(volatilities):
    """Plot volatility comparison"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(volatilities.keys()),
        y=list(volatilities.values()),
        marker_color='#d62728',
        text=[f'{v:.2%}' for v in volatilities.values()],
        textposition='outside'
    ))
    fig.update_layout(
        title='Annualized Volatility Comparison',
        xaxis_title='Ticker',
        yaxis_title='Volatility',
        template='plotly_white'
    )
    return fig

# ============= MAIN DASHBOARD =============

def main():
    st.title("📊 Quantum PCA Stock Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes stock performance using **Quantum Annealing Principal Component Analysis (QAPCA)**.
    Enter a ticker symbol to analyze how it correlates with sector peers and predict performance patterns.
    """)

    # Sidebar
    st.sidebar.header("Configuration")

    # Ticker input
    ticker_input = st.sidebar.text_input(
        "Enter Ticker Symbol",
        value="AAPL",
        help="Enter a stock ticker (e.g., AAPL, TSLA, GOOGL)"
    ).upper()

    # Analysis type
    analysis_type = st.sidebar.radio(
        "Analysis Method",
        ["Classical L1-PCA", "Quantum QAPCA-R", "Both"]
    )

    # Sector selection for comparison
    st.sidebar.subheader("Comparison Stocks")
    selected_tickers = st.sidebar.multiselect(
        "Select stocks for comparison",
        options=ALL_TICKERS,
        default=['AAPL', 'AMZN', 'GOOG', 'XOM']
    )

    if not ticker_input:
        st.warning("Please enter a ticker symbol")
        return

    # Fetch stock info
    with st.spinner(f'Fetching data for {ticker_input}...'):
        stock_data = fetch_stock_data(ticker_input)

        if stock_data is None or stock_data.empty:
            st.error(f"Could not fetch data for {ticker_input}. Please check the ticker symbol.")
            return

        # Get sector
        ticker_sector = SECTOR_MAP.get(ticker_input, "Unknown")

        # Get comparison tickers
        if ticker_input not in selected_tickers:
            comparison_tickers = selected_tickers
        else:
            comparison_tickers = [t for t in selected_tickers if t != ticker_input]

    # Display basic info
    col1, col2, col3, col4 = st.columns(4)

    current_price = stock_data['Close'].iloc[-1]
    price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
    pct_change = (price_change / stock_data['Close'].iloc[0]) * 100

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("6M Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
    col3.metric("Sector", ticker_sector)
    col4.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price & Returns",
        "🔬 PCA Analysis",
        "🌐 Sector Correlation",
        "📊 Performance Metrics"
    ])

    # TAB 1: Price & Returns
    with tab1:
        st.header("Price History & Returns Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_price_history(stock_data, ticker_input), use_container_width=True)

        with col2:
            log_returns = calculate_log_returns(stock_data['Close'])
            st.plotly_chart(plot_returns_distribution(log_returns, ticker_input), use_container_width=True)

        # Comparison with sector
        if comparison_tickers:
            st.subheader("Sector Comparison")
            all_tickers_for_comparison = [ticker_input] + comparison_tickers
            comparison_data = fetch_multiple_stocks(all_tickers_for_comparison)

            if comparison_data is not None:
                st.plotly_chart(
                    plot_sector_comparison(comparison_data, ticker_input, comparison_tickers),
                    use_container_width=True
                )

    # TAB 2: PCA Analysis
    with tab2:
        st.header("Quantum PCA Analysis")

        if not comparison_tickers:
            st.warning("Please select comparison stocks in the sidebar for PCA analysis")
        else:
            with st.spinner("Running PCA analysis..."):
                analysis_results = analyze_stock_with_sector(ticker_input, comparison_tickers)

                if analysis_results is None:
                    st.error("Could not perform analysis. Please check your ticker selections.")
                else:
                    st.success("Analysis complete!")

                    # Show results based on analysis type
                    if analysis_type in ["Classical L1-PCA", "Both"]:
                        st.subheader("Classical L1-PCA Results")
                        if analysis_results['classical_embedding'] is not None:
                            fig = plot_pca_embeddings(
                                analysis_results['classical_embedding'],
                                analysis_results['tickers'],
                                "Classical L1-PCA Embeddings"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Partition analysis
                            classical_emb = analysis_results['classical_embedding'][0]
                            group_positive = [t for i, t in enumerate(analysis_results['tickers'])
                                            if classical_emb[i] >= 0]
                            group_negative = [t for i, t in enumerate(analysis_results['tickers'])
                                            if classical_emb[i] < 0]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**Group 1 (Positive):** {', '.join(group_positive)}")
                            with col2:
                                st.info(f"**Group 2 (Negative):** {', '.join(group_negative)}")
                        else:
                            st.warning("Classical PCA analysis failed")

                    if analysis_type in ["Quantum QAPCA-R", "Both"]:
                        st.subheader("Quantum QAPCA-R Results")
                        if analysis_results['quantum_embedding'] is not None:
                            fig = plot_pca_embeddings(
                                analysis_results['quantum_embedding'],
                                analysis_results['tickers'],
                                "Quantum QAPCA-R Embeddings"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Partition analysis
                            quantum_emb = analysis_results['quantum_embedding'][0]
                            group_positive = [t for i, t in enumerate(analysis_results['tickers'])
                                            if quantum_emb[i] >= 0]
                            group_negative = [t for i, t in enumerate(analysis_results['tickers'])
                                            if quantum_emb[i] < 0]

                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"**Group 1 (Positive):** {', '.join(group_positive)}")
                            with col2:
                                st.success(f"**Group 2 (Negative):** {', '.join(group_negative)}")
                        else:
                            st.warning("Quantum PCA analysis failed")

                    # Interpretation
                    st.markdown("---")
                    st.subheader("📝 Interpretation")
                    st.markdown(f"""
                    **What does this mean for {ticker_input}?**

                    - Stocks in the **same group** tend to move together (positive correlation)
                    - Stocks in **different groups** tend to move in opposite directions (negative correlation)
                    - The magnitude indicates the strength of the correlation
                    - This helps identify **sector contagion** and **diversification opportunities**
                    """)

    # TAB 3: Sector Correlation
    with tab3:
        st.header("Sector Correlation Analysis")

        if comparison_tickers:
            all_tickers_analysis = [ticker_input] + comparison_tickers
            correlation_data = fetch_multiple_stocks(all_tickers_analysis)

            if correlation_data is not None:
                log_returns_all = calculate_log_returns(correlation_data)
                correlation_matrix = calculate_sector_correlations(log_returns_all, SECTOR_MAP)

                st.plotly_chart(plot_correlation_heatmap(correlation_matrix), use_container_width=True)

                # Correlation insights
                if ticker_input in correlation_matrix.columns:
                    st.subheader(f"Correlation with {ticker_input}")
                    correlations = correlation_matrix[ticker_input].sort_values(ascending=False)
                    correlations = correlations[correlations.index != ticker_input]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Most Correlated:**")
                        for tick, corr in correlations.head(3).items():
                            st.write(f"- {tick}: {corr:.3f}")

                    with col2:
                        st.markdown("**Least Correlated:**")
                        for tick, corr in correlations.tail(3).items():
                            st.write(f"- {tick}: {corr:.3f}")

    # TAB 4: Performance Metrics
    with tab4:
        st.header("Performance Metrics")

        if comparison_tickers:
            all_tickers_metrics = [ticker_input] + comparison_tickers
            metrics_data = fetch_multiple_stocks(all_tickers_metrics)

            if metrics_data is not None:
                log_returns_metrics = calculate_log_returns(metrics_data)

                # Calculate metrics
                volatilities = {}
                sharpe_ratios = {}
                mean_returns = {}

                for t in all_tickers_metrics:
                    if t in log_returns_metrics.columns:
                        volatilities[t] = calculate_volatility(log_returns_metrics[t])
                        sharpe_ratios[t] = calculate_sharpe_ratio(log_returns_metrics[t])
                        mean_returns[t] = log_returns_metrics[t].mean() * 252

                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(plot_volatility_comparison(volatilities), use_container_width=True)

                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(sharpe_ratios.keys()),
                        y=list(sharpe_ratios.values()),
                        marker_color='#2ca02c',
                        text=[f'{v:.2f}' for v in sharpe_ratios.values()],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title='Sharpe Ratio Comparison',
                        xaxis_title='Ticker',
                        yaxis_title='Sharpe Ratio',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.subheader("Summary Statistics")
                summary_df = pd.DataFrame({
                    'Mean Annual Return': [f"{v:.2%}" for v in mean_returns.values()],
                    'Volatility': [f"{v:.2%}" for v in volatilities.values()],
                    'Sharpe Ratio': [f"{v:.2f}" for v in sharpe_ratios.values()]
                }, index=list(mean_returns.keys()))

                st.dataframe(summary_df, use_container_width=True)

                # Highlight target stock
                if ticker_input in mean_returns:
                    st.markdown("---")
                    st.subheader(f"📊 {ticker_input} Performance Summary")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Annual Return", f"{mean_returns[ticker_input]:.2%}")
                    col2.metric("Volatility", f"{volatilities[ticker_input]:.2%}")
                    col3.metric("Sharpe Ratio", f"{sharpe_ratios[ticker_input]:.2f}")

                    # Risk assessment
                    if volatilities[ticker_input] < np.mean(list(volatilities.values())):
                        risk_level = "Low"
                        risk_color = "🟢"
                    elif volatilities[ticker_input] < 1.5 * np.mean(list(volatilities.values())):
                        risk_level = "Medium"
                        risk_color = "🟡"
                    else:
                        risk_level = "High"
                        risk_color = "🔴"

                    st.info(f"{risk_color} **Risk Level:** {risk_level}")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Dashboard:**
    This dashboard uses Quantum Annealing PCA (QAPCA-R) to analyze stock relationships and sector correlations.
    The analysis is based on 6 months of historical data and uses L1-norm robust PCA formulations.

    *Built for QuantathonV2 Challenge*
    """)

if __name__ == "__main__":
    main()
