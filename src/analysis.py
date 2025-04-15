import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df (pandas.DataFrame): Stock price data with OHLC columns
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Check if required columns exist
    required_columns = ['Close', 'High', 'Low', 'Open']
    for col in required_columns:
        if col not in result.columns:
            print(f"Required column {col} not found in dataframe")
            return df
    
    try:
        # Simple Moving Averages
        result['SMA_20'] = ta.trend.sma_indicator(result['Close'], window=20)
        result['SMA_50'] = ta.trend.sma_indicator(result['Close'], window=50)
        result['SMA_200'] = ta.trend.sma_indicator(result['Close'], window=200)
        
        # Exponential Moving Averages
        result['EMA_12'] = ta.trend.ema_indicator(result['Close'], window=12)
        result['EMA_26'] = ta.trend.ema_indicator(result['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(result['Close'])
        result['MACD'] = macd.macd()
        result['MACD_Signal'] = macd.macd_signal()
        result['MACD_Histogram'] = macd.macd_diff()
        
        # RSI
        result['RSI'] = ta.momentum.RSIIndicator(result['Close']).rsi()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(result['Close'])
        result['Bollinger_High'] = bollinger.bollinger_hband()
        result['Bollinger_Low'] = bollinger.bollinger_lband()
        result['Bollinger_Mid'] = bollinger.bollinger_mavg()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(result['High'], result['Low'], result['Close'])
        result['Stoch_K'] = stoch.stoch()
        result['Stoch_D'] = stoch.stoch_signal()
        
        # Average True Range (ATR)
        result['ATR'] = ta.volatility.AverageTrueRange(
            result['High'], result['Low'], result['Close']
        ).average_true_range()
        
        # Volume indicators
        if 'Volume' in result.columns:
            # On-Balance Volume
            result['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                result['Close'], result['Volume']
            ).on_balance_volume()
            
            # Volume Weighted Average Price
            result['VWAP'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
        
        # Calculate daily returns
        result['Daily_Return'] = result['Close'].pct_change() * 100
        
        # Calculate trailing volatility (standard deviation of returns)
        result['Volatility_20d'] = result['Daily_Return'].rolling(window=20).std()
        
        return result
    
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return df  # Return original dataframe on error

def get_support_resistance_levels(df, n_levels=3):
    """
    Identify potential support and resistance levels using K-means clustering
    
    Args:
        df (pandas.DataFrame): Stock price data
        n_levels (int): Number of support/resistance levels to identify
        
    Returns:
        dict: Dictionary with support and resistance levels
    """
    if df.empty or len(df) < 30:
        return {'support': [], 'resistance': []}
    
    # Check if required column exists
    if 'Close' not in df.columns:
        print("'Close' column not found in dataframe")
        return {'support': [], 'resistance': []}
    
    try:
        # Extract closing prices and reshape for KMeans
        prices = df['Close'].values.reshape(-1, 1)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_levels, random_state=42, n_init=10)
        kmeans.fit(scaled_prices)
        
        # Get the centroids and convert back to original scale
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Sort centroids by price
        sorted_centroids = sorted([float(c[0]) for c in centroids])
        
        # Determine current price position
        current_price = df['Close'].iloc[-1]
        
        # Separate into support and resistance levels
        support_levels = [c for c in sorted_centroids if c < current_price]
        resistance_levels = [c for c in sorted_centroids if c > current_price]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    except Exception as e:
        print(f"Error calculating support and resistance levels: {e}")
        return {'support': [], 'resistance': []}

def calculate_performance_metrics(df):
    """
    Calculate various performance metrics for a stock
    
    Args:
        df (pandas.DataFrame): Stock price data
        
    Returns:
        dict: Dictionary with performance metrics including day_change, week_change, 
              month_change, ytd_change, volatility, sharpe_ratio, and max_drawdown
    """
    # Initialize default result with N/A values
    default_result = {
        'day_change': None,
        'week_change': None,
        'month_change': None,
        'ytd_change': None,
        'volatility': None,
        'sharpe_ratio': None,
        'max_drawdown': None
    }
    
    # Check if dataframe is empty or too small
    if df.empty or len(df) < 2:
        return default_result
    
    # Check if required column exists
    if 'Close' not in df.columns:
        print("'Close' column not found in dataframe")
        return default_result
    
    try:
        # Sort by date to ensure chronological order
        if 'Date' in df.columns:
            df = df.sort_values('Date')
        
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate performance metrics
        current_price = df['Close'].iloc[-1]
        
        # Daily return (day change)
        try:
            prev_day_price = df['Close'].iloc[-2]
            day_change = ((current_price / prev_day_price) - 1) * 100
        except IndexError:
            day_change = None
        
        # Weekly return (week change)
        try:
            last_week_idx = max(0, len(df) - 6)
            prev_week_price = df['Close'].iloc[last_week_idx]
            week_change = ((current_price / prev_week_price) - 1) * 100
        except (IndexError, ZeroDivisionError):
            week_change = None
        
        # Monthly return (month change)
        try:
            last_month_idx = max(0, len(df) - 23)
            prev_month_price = df['Close'].iloc[last_month_idx]
            month_change = ((current_price / prev_month_price) - 1) * 100
        except (IndexError, ZeroDivisionError):
            month_change = None
        
        # Year-to-date return (ytd change)
        try:
            last_year_idx = max(0, len(df) - 253)
            prev_year_price = df['Close'].iloc[last_year_idx]
            ytd_change = ((current_price / prev_year_price) - 1) * 100
        except (IndexError, ZeroDivisionError):
            ytd_change = None
        
        # Calculate other metrics
        # Annualized volatility (standard deviation * sqrt(trading days))
        try:
            volatility = df['Daily_Return'].std() * (252 ** 0.5) * 100
        except:
            volatility = None
        
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        try:
            mean_return = df['Daily_Return'].mean()
            std_return = df['Daily_Return'].std()
            if std_return == 0:
                sharpe_ratio = None
            else:
                sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)
        except:
            sharpe_ratio = None
        
        # Maximum Drawdown
        try:
            cumulative_returns = (1 + df['Daily_Return']).cumprod()
            max_return = cumulative_returns.expanding().max()
            drawdown = ((cumulative_returns / max_return) - 1)
            max_drawdown = drawdown.min() * 100
        except:
            max_drawdown = None
        
        # Build result dictionary
        metrics = {
            'day_change': day_change,
            'week_change': week_change,
            'month_change': month_change,
            'ytd_change': ytd_change,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return default_result

def format_number(value):
    """Format number with appropriate precision or return 'N/A'"""
    if value is None or pd.isna(value):
        return 'N/A'
    try:
        # Return the numeric value as is, formatting will be done in the UI
        return value
    except:
        return 'N/A'

def get_trading_signals(df):
    """
    Generate simple trading signals based on technical indicators
    
    Args:
        df (pandas.DataFrame): Stock price data with technical indicators
        
    Returns:
        pandas.DataFrame: DataFrame with added trading signals
    """
    if df.empty or 'SMA_20' not in df.columns or 'SMA_50' not in df.columns:
        return df
    
    result = df.copy()
    
    # SMA Crossover signal
    result['Signal_SMA_Crossover'] = 0
    result.loc[result['SMA_20'] > result['SMA_50'], 'Signal_SMA_Crossover'] = 1
    result.loc[result['SMA_20'] < result['SMA_50'], 'Signal_SMA_Crossover'] = -1
    
    # RSI Overbought/Oversold signal
    if 'RSI' in result.columns:
        result['Signal_RSI'] = 0
        result.loc[result['RSI'] < 30, 'Signal_RSI'] = 1  # Oversold - buy signal
        result.loc[result['RSI'] > 70, 'Signal_RSI'] = -1  # Overbought - sell signal
    
    # MACD Signal
    if 'MACD' in result.columns and 'MACD_Signal' in result.columns:
        result['Signal_MACD'] = 0
        result.loc[result['MACD'] > result['MACD_Signal'], 'Signal_MACD'] = 1  # Bullish
        result.loc[result['MACD'] < result['MACD_Signal'], 'Signal_MACD'] = -1  # Bearish
    
    # Bollinger Band signals
    if 'Bollinger_High' in result.columns and 'Bollinger_Low' in result.columns:
        result['Signal_Bollinger'] = 0
        result.loc[result['Close'] < result['Bollinger_Low'], 'Signal_Bollinger'] = 1  # Oversold
        result.loc[result['Close'] > result['Bollinger_High'], 'Signal_Bollinger'] = -1  # Overbought
    
    # Combined signal (simple average of all signals)
    signal_columns = [col for col in result.columns if col.startswith('Signal_')]
    if signal_columns:
        result['Signal_Combined'] = result[signal_columns].mean(axis=1)
    
    return result

def calculate_correlation_matrix(tickers_data):
    """
    Calculate correlation matrix between multiple stocks
    
    Args:
        tickers_data (dict): Dictionary with ticker symbols as keys and DataFrames as values
        
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    # Extract closing prices for each ticker
    close_prices = {}
    
    for ticker, data in tickers_data.items():
        if not data.empty and 'Close' in data.columns:
            close_prices[ticker] = data['Close']
    
    if not close_prices:
        return pd.DataFrame()
    
    # Create a DataFrame with all closing prices
    prices_df = pd.DataFrame(close_prices)
    
    # Calculate correlation matrix
    return prices_df.corr() 