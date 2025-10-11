import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import dual_annealing
import neal
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Quantum PCA Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============= DATA LOADER CLASS =============

class StockDataLoader:
    """Flexible data loader for the Kaggle stock market dataset."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.stocks_path = self.base_path / "Stocks"
        self.etfs_path = self.base_path / "ETFs"
        self.meta_path = self.base_path / "symbols_valid_meta.csv"
        
        if self.meta_path.exists():
            self.meta_data = pd.read_csv(self.meta_path)
        else:
            self.meta_data = None
    
    def get_available_tickers(self, include_etfs=True):
        """Get list of all available tickers from the dataset."""
        tickers = []
        
        if self.stocks_path.exists():
            stock_files = list(self.stocks_path.glob("*.csv"))
            tickers.extend([f.stem for f in stock_files])
        
        if include_etfs and self.etfs_path.exists():
            etf_files = list(self.etfs_path.glob("*.csv"))
            tickers.extend([f.stem for f in etf_files])
        
        return sorted(tickers)
    
    def load_ticker_data(self, ticker):
        """Load data for a single ticker."""
        stock_file = self.stocks_path / f"{ticker}.csv"
        if stock_file.exists():
            df = pd.read_csv(stock_file)
            df['Ticker'] = ticker
            return df
        
        etf_file = self.etfs_path / f"{ticker}.csv"
        if etf_file.exists():
            df = pd.read_csv(etf_file)
            df['Ticker'] = ticker
            return df
        
        return None
    
    def load_data(self, tickers, date_range='all', filter_common_dates=True):
        """Load data for specified tickers and date range."""
        all_data = []
        
        for ticker in tickers:
            df = self.load_ticker_data(ticker)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            return None
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data['Date'] = pd.to_datetime(combined_data['Date'])
        
        # Filter by date range if specified
        if date_range != 'all':
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                combined_data = combined_data[
                    (combined_data['Date'] >= start_date) & 
                    (combined_data['Date'] <= end_date)
                ]
        
        # Filter to common dates if requested
        if filter_common_dates:
            date_counts = combined_data.groupby('Date')['Ticker'].nunique()
            n_tickers = combined_data['Ticker'].nunique()
            common_dates = date_counts[date_counts == n_tickers].index
            combined_data = combined_data[combined_data['Date'].isin(common_dates)]
        
        combined_data = combined_data.sort_values(by=['Ticker', 'Date'])
        combined_data = combined_data.reset_index(drop=True)
        
        return combined_data

# ============= PCA FUNCTIONS =============

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

def solve_l1_multi_component(J, K=2, eps=0.2, num_reads=100):
    """Solve for multiple L1 QAPCA components simultaneously using quantum annealing."""
    N = J.shape[0]

    # Construct Ising coupling matrix (Eq. 21)
    I_K = np.eye(K)
    onesK = np.ones((K, K))

    H = np.kron(I_K, (-K) * J) + np.kron(onesK - I_K, -eps * J)

    J_ising = {}
    for i in range(K * N):
        for j in range(i + 1, K * N):
            val = H[i, j]
            if abs(val) > 1e-12:
                J_ising[(i, j)] = val

    h = {i: 0.0 for i in range(K * N)}  # no biases

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h, J_ising, num_reads=num_reads)
    best_sample = sampleset.first.sample

    b_opt = np.array([best_sample[i] for i in range(K * N)])
    B_opt = b_opt.reshape(N, K)

    return B_opt, J

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

def do_l1_multi_qapca(sample_data, K=2, eps=0.2, num_reads=100):
    """Perform multi-component QAPCA"""
    J, X = solve_covariance_matrix(sample_data)
    B_opt, _ = solve_l1_multi_component(J, K=K, eps=eps, num_reads=num_reads)
    
    X_Bopt = X @ B_opt
    R_L1 = Phi(X_Bopt)
    emb = R_L1.T @ X
    
    # Extract individual components
    components = [B_opt[:, k] for k in range(K)]
    
    return emb, components

# ============= ANALYSIS FUNCTIONS =============

def calculate_log_returns(data):
    """Calculate log returns from price data"""
    if isinstance(data, pd.DataFrame):
        log_returns = np.log(data / data.shift(1)).dropna()
    else:
        log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def analyze_stock_with_pca(tickers, stock_data, method='classical', num_components=1, eps=0.2):
    """Analyze stock performance using PCA"""
    
    # Pivot data to get close prices for each ticker
    price_data = stock_data.pivot(index='Date', columns='Ticker', values='Close')
    
    # Calculate log returns
    log_returns = calculate_log_returns(price_data)
    
    # Prepare data for PCA
    log_return_vectors = {}
    for t in tickers:
        if t in log_returns.columns:
            returns = log_returns[t].dropna().values
            if len(returns) > 0:
                log_return_vectors[t] = returns
    
    # Convert to pandas Series for PCA
    log_return_series = pd.Series(log_return_vectors)
    
    # Perform PCA based on method
    try:
        if method == 'classical':
            embedding, components = do_l1_pca(log_return_series, num_components, solve_l1_classical_component)
        elif method == 'quantum':
            embedding, components = do_l1_pca(log_return_series, num_components, solve_l1_qapca_r_component)
        elif method == 'multi_quantum':
            embedding, components = do_l1_multi_qapca(log_return_series, K=num_components, eps=eps, num_reads=100)
        else:
            return None
    except Exception as e:
        st.error(f"PCA analysis failed: {str(e)}")
        return None
    
    return {
        'data': price_data,
        'log_returns': log_returns,
        'tickers': list(log_return_vectors.keys()),
        'embedding': embedding,
        'components': components,
        'method': method
    }

def calculate_volatility(log_returns):
    """Calculate annualized volatility"""
    return log_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(log_returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    mean_return = log_returns.mean() * 252
    volatility = calculate_volatility(log_returns)
    return (mean_return - risk_free_rate) / volatility

# ============= VISUALIZATION FUNCTIONS =============

def create_scatterplot(embeddings, labels, title):
    """
    Create a scatterplot with different color for each label using Plotly.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Array of shape (2, n_samples) or (n_samples, 2) containing 2D embeddings
    labels : list
        List of labels for each sample
    title : str
        Title for the plot
    """
    # Handle different embedding shapes
    if embeddings.shape[0] == 2:
        x = embeddings[0, :]
        y = embeddings[1, :]
    elif embeddings.shape[1] == 2:
        x = embeddings[:, 0]
        y = embeddings[:, 1]
    elif embeddings.shape[0] == 1:
        x = embeddings[0, :]
        y = np.zeros_like(x)
    elif embeddings.shape[1] == 1:
        x = embeddings[:, 0]
        y = np.zeros_like(x)
    else:
        raise ValueError(f'Embedding shape {embeddings.shape} is not supported')
    
    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'Component 1': x,
        'Component 2': y,
        'Ticker': labels
    })
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Component 1',
        y='Component 2',
        text='Ticker',
        color='Ticker',
        title=title,
        width=800,
        height=600
    )
    
    # Update traces for better visibility
    fig.update_traces(
        textposition='top center',
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
        textfont_size=10
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        showlegend=True,
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        hovermode='closest'
    )
    
    return fig

def plot_price_history(data, ticker):
    """Plot price history"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[ticker],
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

def plot_pca_embeddings(embeddings, tickers, title="PCA Embeddings", component_idx=0):
    """Plot PCA embeddings as bar chart"""
    if embeddings is None:
        return None

    fig = go.Figure()
    colors = ['#1f77b4' if e >= 0 else '#ff7f0e' for e in embeddings[component_idx]]

    fig.add_trace(go.Bar(
        x=tickers,
        y=embeddings[component_idx],
        marker_color=colors,
        text=embeddings[component_idx].round(3),
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

def plot_multi_component_embeddings(embeddings, tickers, title="Multi-Component PCA"):
    """Plot multiple PCA components"""
    if embeddings is None or embeddings.shape[0] < 2:
        return None
    
    fig = go.Figure()
    
    for i in range(min(embeddings.shape[0], 3)):  # Plot up to 3 components
        fig.add_trace(go.Bar(
            x=tickers,
            y=embeddings[i],
            name=f'Component {i+1}',
            text=embeddings[i].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Ticker',
        yaxis_title='Component Value',
        template='plotly_white',
        barmode='group',
        height=500
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

# ============= MAIN DASHBOARD =============

@st.cache_resource
def initialize_data_loader(base_path):
    """Initialize data loader (cached)"""
    return StockDataLoader(base_path)

def main():
    st.title("📊 Quantum PCA Stock Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes stock performance using **Quantum Annealing Principal Component Analysis (QAPCA)**.
    Select tickers and date ranges to analyze stock relationships and sector correlations.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source",
        ["Kaggle Dataset (Local)", "Yahoo Finance (Online)"]
    )
    
    if data_source == "Kaggle Dataset (Local)":
        # Path to dataset
        dataset_path = st.sidebar.text_input(
            "Dataset Path",
            value="./stock-market-dataset",
            help="Path to the stock market dataset directory"
        )
        
        # Initialize data loader
        try:
            data_loader = initialize_data_loader(dataset_path)
            available_tickers = data_loader.get_available_tickers(include_etfs=True)
            
            if not available_tickers:
                st.error("No tickers found in the dataset path. Please check the path.")
                return
            
            st.sidebar.success(f"✓ Found {len(available_tickers)} tickers")
            
            # Show sample tickers
            with st.sidebar.expander("View Sample Tickers"):
                st.write(available_tickers[:20])
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
        
        # Ticker selection
        st.sidebar.subheader("Ticker Selection")
        selection_method = st.sidebar.radio(
            "Selection method",
            ["Manual Selection", "Quick Presets"]
        )
        
        if selection_method == "Manual Selection":
            selected_tickers = st.sidebar.multiselect(
                "Select tickers for analysis",
                options=available_tickers,
                default=available_tickers[:5] if len(available_tickers) >= 5 else available_tickers,
                help="Select 3-15 tickers for optimal performance"
            )
        else:
            preset = st.sidebar.selectbox(
                "Choose preset",
                ["Top 10", "Tech Stocks", "Energy Stocks", "Random 10"]
            )
            
            if preset == "Top 10":
                selected_tickers = available_tickers[:10]
            elif preset == "Tech Stocks":
                tech_tickers = [t for t in available_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']]
                selected_tickers = tech_tickers[:10] if tech_tickers else available_tickers[:10]
            elif preset == "Energy Stocks":
                energy_tickers = [t for t in available_tickers if t in ['XOM', 'CVX', 'COP', 'SLB', 'EOG']]
                selected_tickers = energy_tickers[:10] if energy_tickers else available_tickers[:10]
            else:
                import random
                random.seed(42)
                selected_tickers = random.sample(available_tickers, min(10, len(available_tickers)))
            
            st.sidebar.info(f"Selected: {', '.join(selected_tickers)}")
        
        # Date range selection
        st.sidebar.subheader("Date Range")
        date_option = st.sidebar.radio(
            "Select date range",
            ["Last 6 Months", "Last 1 Year", "Last 2 Years", "Custom Range"]
        )
        
        if date_option == "Custom Range":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime(2020, 1, 1))
            with col2:
                end_date = st.date_input("End Date", datetime.now())
            date_range = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        else:
            end_date = datetime.now()
            if date_option == "Last 6 Months":
                start_date = end_date - timedelta(days=180)
            elif date_option == "Last 1 Year":
                start_date = end_date - timedelta(days=365)
            else:  # Last 2 Years
                start_date = end_date - timedelta(days=730)
            date_range = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        # Load data
        if not selected_tickers:
            st.warning("Please select at least one ticker")
            return
        
        if len(selected_tickers) > 20:
            st.warning("⚠️ Selecting more than 20 tickers may slow down the analysis")
        
        with st.spinner(f'Loading data for {len(selected_tickers)} tickers...'):
            stock_data = data_loader.load_data(
                tickers=selected_tickers,
                date_range=date_range,
                filter_common_dates=True
            )
            
            if stock_data is None or stock_data.empty:
                st.error("Could not load data for selected tickers and date range")
                return
            
            st.success(f"✓ Loaded {len(stock_data)} records for {len(selected_tickers)} tickers")
    
    else:  # Yahoo Finance
        st.sidebar.info("Yahoo Finance mode - using original implementation")
        st.warning("Please switch to Kaggle Dataset mode to use the new features")
        return
    
    # Analysis configuration
    st.sidebar.subheader("Analysis Configuration")
    
    analysis_method = st.sidebar.selectbox(
        "PCA Method",
        ["Classical L1-PCA", "Quantum QAPCA-R", "Multi-Component QAPCA", "Compare All"]
    )
    
    # Dimension selection
    embedding_dim = st.sidebar.radio(
        "Embedding Dimension",
        [1, 2],
        help="1D for bar charts, 2D for scatter plots"
    )
    
    if analysis_method == "Multi-Component QAPCA":
        if embedding_dim == 2:
            num_components = 2
        else:
            num_components = st.sidebar.slider("Number of Components", 1, 5, 2)
        eps = st.sidebar.slider("Epsilon (ε)", 0.1, 1.0, 0.2, 0.1)
    else:
        num_components = embedding_dim
        eps = 0.2
    
    # Select target ticker for detailed analysis
    target_ticker = st.sidebar.selectbox(
        "Target Ticker for Detailed Analysis",
        options=selected_tickers,
        index=0
    )
    
    # Display basic info
    st.header(f"📈 Analysis for {target_ticker}")
    
    # Get price data for target ticker
    target_data = stock_data[stock_data['Ticker'] == target_ticker].set_index('Date')['Close']
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = target_data.iloc[-1]
    price_change = target_data.iloc[-1] - target_data.iloc[0]
    pct_change = (price_change / target_data.iloc[0]) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Period Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
    col3.metric("Selected Tickers", len(selected_tickers))
    col4.metric("Data Points", len(target_data))
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price & Returns",
        "🔬 PCA Analysis",
        "🌐 Correlation Analysis",
        "📊 Performance Metrics"
    ])
    
    # TAB 1: Price & Returns
    with tab1:
        st.header("Price History & Comparison")
        
        # Price history
        price_data = stock_data.pivot(index='Date', columns='Ticker', values='Close')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_price_history(price_data, target_ticker), use_container_width=True)
        
        with col2:
            log_returns = calculate_log_returns(price_data)
            target_returns = log_returns[target_ticker].dropna()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=target_returns,
                nbinsx=50,
                name=target_ticker,
                marker_color='#2ca02c'
            ))
            fig.update_layout(
                title=f'{target_ticker} Log Returns Distribution',
                xaxis_title='Log Returns',
                yaxis_title='Frequency',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with all tickers
        st.subheader("Multi-Ticker Comparison")
        comparison_tickers = [t for t in selected_tickers if t != target_ticker]
        st.plotly_chart(
            plot_sector_comparison(price_data, target_ticker, comparison_tickers),
            use_container_width=True
        )
    
    # TAB 2: PCA Analysis
    with tab2:
        st.header("Quantum PCA Analysis")
        
        with st.spinner("Running PCA analysis..."):
            results = {}
            
            if analysis_method == "Classical L1-PCA":
                results['classical'] = analyze_stock_with_pca(
                    selected_tickers, stock_data, method='classical', 
                    num_components=num_components, eps=eps
                )
            elif analysis_method == "Quantum QAPCA-R":
                results['quantum'] = analyze_stock_with_pca(
                    selected_tickers, stock_data, method='quantum', 
                    num_components=num_components, eps=eps
                )
            elif analysis_method == "Multi-Component QAPCA":
                results['multi_quantum'] = analyze_stock_with_pca(
                    selected_tickers, stock_data, method='multi_quantum', 
                    num_components=num_components, eps=eps
                )
            else:  # Compare All
                comp_dim = 2 if embedding_dim == 2 else 1
                results['classical'] = analyze_stock_with_pca(
                    selected_tickers, stock_data, method='classical', 
                    num_components=comp_dim, eps=eps
                )
                results['quantum'] = analyze_stock_with_pca(
                    selected_tickers, stock_data, method='quantum', 
                    num_components=comp_dim, eps=eps
                )
                if embedding_dim == 2:
                    results['multi_quantum'] = analyze_stock_with_pca(
                        selected_tickers, stock_data, method='multi_quantum', 
                        num_components=2, eps=eps
                    )
            
            # Display results
            for method_name, result in results.items():
                if result is None:
                    continue
                
                st.subheader(f"📊 {method_name.replace('_', ' ').title()} Results")
                
                # Check if 2D embedding
                if result['embedding'].shape[0] >= 2 and embedding_dim == 2:
                    # Show 2D scatter plot
                    fig = create_scatterplot(
                        result['embedding'],
                        result['tickers'],
                        f"{method_name.replace('_', ' ').title()} - 2D Embedding"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show partitions for each component
                    for comp_idx in range(min(2, result['embedding'].shape[0])):
                        st.markdown(f"**Component {comp_idx + 1} Partitioning:**")
                        emb = result['embedding'][comp_idx]
                        group_pos = [t for i, t in enumerate(result['tickers']) if emb[i] >= 0]
                        group_neg = [t for i, t in enumerate(result['tickers']) if emb[i] < 0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"**Positive Group:** {', '.join(group_pos)}")
                        with col2:
                            st.info(f"**Negative Group:** {', '.join(group_neg)}")
                
                elif method_name == 'multi_quantum' and result['embedding'].shape[0] > 1:
                    # Multi-component bar chart visualization
                    fig = plot_multi_component_embeddings(
                        result['embedding'],
                        result['tickers'],
                        f"Multi-Component QAPCA ({num_components} components)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show partitions for each component
                    for comp_idx in range(min(result['embedding'].shape[0], 3)):
                        st.markdown(f"**Component {comp_idx + 1} Partitioning:**")
                        emb = result['embedding'][comp_idx]
                        group_pos = [t for i, t in enumerate(result['tickers']) if emb[i] >= 0]
                        group_neg = [t for i, t in enumerate(result['tickers']) if emb[i] < 0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"**Positive Group:** {', '.join(group_pos)}")
                        with col2:
                            st.info(f"**Negative Group:** {', '.join(group_neg)}")
                else:
                    # Single component bar chart visualization
                    fig = plot_pca_embeddings(
                        result['embedding'],
                        result['tickers'],
                        f"{method_name.replace('_', ' ').title()} Embeddings"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Partition analysis
                    emb = result['embedding'][0]
                    group_pos = [t for i, t in enumerate(result['tickers']) if emb[i] >= 0]
                    group_neg = [t for i, t in enumerate(result['tickers']) if emb[i] < 0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Positive Group:** {', '.join(group_pos)}")
                    with col2:
                        st.info(f"**Negative Group:** {', '.join(group_neg)}")
                
                st.markdown("---")
            
            # Interpretation
            st.subheader("📝 Interpretation")
            if embedding_dim == 2:
                st.markdown(f"""
                **What does this mean for {target_ticker}?**
                
                - **2D Embedding** reveals complex relationships between stocks
                - Stocks **close together** in the scatter plot tend to move similarly
                - Stocks **far apart** have different movement patterns
                - **Component 1** (x-axis) captures the primary direction of variation
                - **Component 2** (y-axis) captures the secondary direction
                - This helps identify **diversification opportunities** and **sector clusters**
                """)
            else:
                st.markdown(f"""
                **What does this mean for {target_ticker}?**
                
                - Stocks in the **same group** tend to move together (positive correlation)
                - Stocks in **different groups** tend to move in opposite directions (negative correlation)
                - The magnitude indicates the strength of the relationship
                - **Multi-component analysis** reveals multiple patterns of co-movement
                - This helps identify **diversification opportunities** and **risk factors**
                """)
    
    # TAB 3: Correlation Analysis
    with tab3:
        st.header("Correlation Matrix")
        
        price_data = stock_data.pivot(index='Date', columns='Ticker', values='Close')
        log_returns = calculate_log_returns(price_data)
        correlation_matrix = log_returns.corr()
        
        st.plotly_chart(plot_correlation_heatmap(correlation_matrix), use_container_width=True)
        
        # Correlation insights for target ticker
        if target_ticker in correlation_matrix.columns:
            st.subheader(f"Correlation with {target_ticker}")
            correlations = correlation_matrix[target_ticker].sort_values(ascending=False)
            correlations = correlations[correlations.index != target_ticker]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most Correlated:**")
                for tick, corr in correlations.head(5).items():
                    st.write(f"- {tick}: {corr:.3f}")
            
            with col2:
                st.markdown("**Least Correlated:**")
                for tick, corr in correlations.tail(5).items():
                    st.write(f"- {tick}: {corr:.3f}")
    
    # TAB 4: Performance Metrics
    with tab4:
        st.header("Performance Metrics")
        
        price_data = stock_data.pivot(index='Date', columns='Ticker', values='Close')
        log_returns = calculate_log_returns(price_data)
        
        # Calculate metrics
        volatilities = {}
        sharpe_ratios = {}
        mean_returns = {}
        
        for ticker in selected_tickers:
            if ticker in log_returns.columns:
                volatilities[ticker] = calculate_volatility(log_returns[ticker])
                sharpe_ratios[ticker] = calculate_sharpe_ratio(log_returns[ticker])
                mean_returns[ticker] = log_returns[ticker].mean() * 252
        
        col1, col2 = st.columns(2)
        
        with col1:
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
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Target ticker summary
        if target_ticker in mean_returns:
            st.markdown("---")
            st.subheader(f"📊 {target_ticker} Performance Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Annual Return", f"{mean_returns[target_ticker]:.2%}")
            col2.metric("Volatility", f"{volatilities[target_ticker]:.2%}")
            col3.metric("Sharpe Ratio", f"{sharpe_ratios[target_ticker]:.2f}")
            
            # Risk assessment
            avg_vol = np.mean(list(volatilities.values()))
            if volatilities[target_ticker] < avg_vol:
                risk_level = "Low"
                risk_color = "🟢"
            elif volatilities[target_ticker] < 1.5 * avg_vol:
                risk_level = "Medium"
                risk_color = "🟡"
            else:
                risk_level = "High"
                risk_color = "🔴"
            
            st.info(f"{risk_color} **Risk Level:** {risk_level} (compared to peer average)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Dashboard:**
    - Uses Quantum Annealing PCA (QAPCA-R) for stock relationship analysis
    - **Multi-Component QAPCA** reveals multiple patterns of co-movement simultaneously
    - **2D Embeddings** provide visual insights into stock clustering and relationships
    - Analysis based on historical data with L1-norm robust PCA formulations
    - Data loaded from comprehensive Kaggle stock market dataset
    
    *Enhanced Dashboard for Advanced Stock Analysis*
    """)

if __name__ == "__main__":
    main()
