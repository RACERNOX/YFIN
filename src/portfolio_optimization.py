import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def get_stock_data_for_portfolio(tickers, period='1y', interval='1d'):
    """
    Fetch historical data for a list of stocks for portfolio optimization.
    
    Args:
        tickers (list): List of stock tickers
        period (str): Time period to fetch data for ('1y', '2y', '5y', etc.)
        interval (str): Data interval ('1d', '1wk', '1mo', etc.)
    
    Returns:
        pandas.DataFrame: DataFrame with historical closing prices
    """
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker')
    
    # Format multi-level columns to single level if multiple tickers
    if len(tickers) > 1:
        closes = pd.DataFrame()
        for ticker in tickers:
            closes[ticker] = data[ticker]['Close']
        return closes
    else:
        # If only one ticker, format is different
        return pd.DataFrame({tickers[0]: data['Close']})

def calculate_returns(prices_df):
    """
    Calculate daily returns from historical prices.
    
    Args:
        prices_df (pandas.DataFrame): DataFrame with historical prices
    
    Returns:
        pandas.DataFrame: DataFrame with daily returns
    """
    returns_df = prices_df.pct_change().dropna()
    return returns_df

def calculate_portfolio_performance(weights, returns):
    """
    Calculate portfolio performance metrics (return, volatility, Sharpe ratio).
    
    Args:
        weights (numpy.array): Asset weights in portfolio
        returns (pandas.DataFrame): Asset returns
    
    Returns:
        tuple: (Portfolio return, portfolio volatility, Sharpe ratio)
    """
    # Convert returns DataFrame to numpy array for calculations
    returns_array = returns.values
    
    # Calculate portfolio return and volatility (annualized)
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe_ratio(weights, returns):
    """
    Calculate negative Sharpe ratio for optimization (minimize negative = maximize positive).
    
    Args:
        weights (numpy.array): Asset weights in portfolio
        returns (pandas.DataFrame): Asset returns
    
    Returns:
        float: Negative Sharpe ratio
    """
    return -calculate_portfolio_performance(weights, returns)[2]

def optimize_portfolio(returns, risk_free_rate=0.0, target_return=None, target_volatility=None):
    """
    Optimize portfolio weights using Modern Portfolio Theory.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        risk_free_rate (float): Risk-free rate
        target_return (float, optional): Target portfolio return for minimum volatility
        target_volatility (float, optional): Target volatility for maximum return
    
    Returns:
        dict: Optimization results including optimal weights and performance metrics
    """
    num_assets = len(returns.columns)
    args = (returns,)
    
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: no short selling (weights between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize for maximum Sharpe ratio
    optimal_sharpe = sco.minimize(
        negative_sharpe_ratio, 
        initial_weights, 
        args=args, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    # Get optimal weights
    optimal_weights = optimal_sharpe['x']
    
    # Calculate performance with optimal weights
    optimal_performance = calculate_portfolio_performance(optimal_weights, returns)
    
    # Store results
    optimization_results = {
        'optimal_weights': dict(zip(returns.columns, optimal_weights)),
        'expected_annual_return': optimal_performance[0],
        'annual_volatility': optimal_performance[1],
        'sharpe_ratio': optimal_performance[2]
    }
    
    return optimization_results

def generate_efficient_frontier(returns, num_portfolios=5000):
    """
    Generate the efficient frontier points for plotting.
    
    Args:
        returns (pandas.DataFrame): Asset returns
        num_portfolios (int): Number of random portfolios to generate
    
    Returns:
        pandas.DataFrame: DataFrame with portfolio performances
    """
    num_assets = len(returns.columns)
    results = np.zeros((num_portfolios, 3 + num_assets))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        # Calculate portfolio performance
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(weights, returns)
        
        # Store results
        results[i, 0] = portfolio_return
        results[i, 1] = portfolio_volatility
        results[i, 2] = sharpe_ratio
        
        # Store weights
        for j in range(num_assets):
            results[i, j+3] = weights[j]
    
    # Convert results to DataFrame
    columns = ['return', 'volatility', 'sharpe']
    columns.extend(returns.columns)
    
    return pd.DataFrame(results, columns=columns)

def plot_efficient_frontier(frontier_df, optimal_point=None, risk_free_rate=0.0, theme='dark'):
    """
    Plot the efficient frontier using Plotly.
    
    Args:
        frontier_df (pandas.DataFrame): DataFrame with efficient frontier points
        optimal_point (tuple, optional): Optimal portfolio (volatility, return) for highlighting
        risk_free_rate (float): Risk-free rate for capital market line
        theme (str): 'dark' or 'light' theme for the plot
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with efficient frontier
    """
    # Set colors based on theme
    bg_color = '#121212' if theme == 'dark' else '#ffffff'
    text_color = '#e0e0e0' if theme == 'dark' else '#212529'
    grid_color = 'rgba(255, 255, 255, 0.1)' if theme == 'dark' else 'rgba(0, 0, 0, 0.1)'
    
    # Create figure
    fig = px.scatter(
        frontier_df, 
        x='volatility', 
        y='return', 
        color='sharpe',
        color_continuous_scale='Plasma',
        title='Efficient Frontier',
        template='plotly_dark' if theme == 'dark' else 'plotly_white'
    )
    
    # Add optimal portfolio point if provided
    if optimal_point:
        fig.add_trace(
            go.Scatter(
                x=[optimal_point[0]],
                y=[optimal_point[1]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                name='Optimal Portfolio'
            )
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Annual Volatility',
        yaxis_title='Annual Expected Return',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        legend=dict(
            bgcolor=bg_color,
            bordercolor=text_color,
            orientation='h',
            y=1.02,
            x=0.5,
            xanchor='center'
        ),
        coloraxis_colorbar=dict(
            title='Sharpe Ratio',
            titleside='right'
        ),
        xaxis=dict(
            gridcolor=grid_color,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zeroline=False
        )
    )
    
    return fig

def get_risk_profile_allocation(risk_profile, returns):
    """
    Get portfolio allocation based on user's risk profile.
    
    Args:
        risk_profile (str): Risk profile ('conservative', 'moderate', 'aggressive')
        returns (pandas.DataFrame): Asset returns
    
    Returns:
        dict: Optimized portfolio weights and metrics for the given risk profile
    """
    # Generate efficient frontier
    frontier_df = generate_efficient_frontier(returns, num_portfolios=5000)
    
    # Find the minimum volatility portfolio
    min_vol_idx = frontier_df['volatility'].idxmin()
    min_vol_portfolio = frontier_df.iloc[min_vol_idx]
    
    # Find the maximum Sharpe ratio portfolio
    max_sharpe_idx = frontier_df['sharpe'].idxmax()
    max_sharpe_portfolio = frontier_df.iloc[max_sharpe_idx]
    
    # Find the maximum return portfolio
    max_return_idx = frontier_df['return'].idxmax()
    max_return_portfolio = frontier_df.iloc[max_return_idx]
    
    # Select portfolio based on risk profile
    if risk_profile == 'conservative':
        selected_portfolio = min_vol_portfolio
        description = "Conservative allocation prioritizing lower risk"
    elif risk_profile == 'moderate':
        selected_portfolio = max_sharpe_portfolio
        description = "Moderate allocation with optimal risk-return ratio"
    else:  # aggressive
        # Choose a portfolio with higher returns than max Sharpe but lower than max return
        aggressive_idx = frontier_df[
            (frontier_df['return'] > max_sharpe_portfolio['return']) & 
            (frontier_df['return'] < max_return_portfolio['return'])
        ]['sharpe'].idxmax()
        
        if pd.isna(aggressive_idx):
            # Fallback to max return if no suitable portfolio found
            selected_portfolio = max_return_portfolio
            description = "Aggressive allocation targeting maximum returns"
        else:
            selected_portfolio = frontier_df.iloc[aggressive_idx]
            description = "Aggressive allocation with higher return potential"
    
    # Extract weights and performance metrics
    assets = returns.columns
    weights = {asset: selected_portfolio[3 + i] for i, asset in enumerate(assets)}
    
    result = {
        'risk_profile': risk_profile,
        'description': description,
        'weights': weights,
        'expected_annual_return': selected_portfolio['return'],
        'annual_volatility': selected_portfolio['volatility'],
        'sharpe_ratio': selected_portfolio['sharpe']
    }
    
    return result

def compare_allocations(returns):
    """
    Generate and compare portfolio allocations for different risk profiles.
    
    Args:
        returns (pandas.DataFrame): Asset returns
    
    Returns:
        dict: Comparison of conservative, moderate, and aggressive portfolios
    """
    conservative = get_risk_profile_allocation('conservative', returns)
    moderate = get_risk_profile_allocation('moderate', returns)
    aggressive = get_risk_profile_allocation('aggressive', returns)
    
    return {
        'conservative': conservative,
        'moderate': moderate,
        'aggressive': aggressive
    } 