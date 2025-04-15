import pandas as pd
import numpy as np
from scipy import stats

def calculate_moving_averages(df, windows=[20, 50, 200]):
    """
    Calculate simple moving averages for the given windows.
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        windows: List of window sizes for moving averages
        
    Returns:
        DataFrame with original data and moving averages added
    """
    result = df.copy()
    for window in windows:
        result[f'MA_{window}'] = result['Close'].rolling(window=window).mean()
    return result

def calculate_rsi(df, window=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        window: RSI calculation window
        
    Returns:
        DataFrame with original data and RSI added
    """
    result = df.copy()
    delta = result['Close'].diff()
    
    # Make two series: one for gains and one for losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    result['RSI'] = 100 - (100 / (1 + rs))
    
    return result

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with original data and MACD components added
    """
    result = df.copy()
    
    # Calculate EMAs
    result['EMA_fast'] = result['Close'].ewm(span=fast, adjust=False).mean()
    result['EMA_slow'] = result['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line and signal line
    result['MACD_line'] = result['EMA_fast'] - result['EMA_slow']
    result['MACD_signal'] = result['MACD_line'].ewm(span=signal, adjust=False).mean()
    
    # Calculate MACD histogram
    result['MACD_hist'] = result['MACD_line'] - result['MACD_signal']
    
    return result

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        window: Rolling window for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with original data and Bollinger Bands added
    """
    result = df.copy()
    
    # Calculate middle band (SMA)
    result['BB_middle'] = result['Close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    result['BB_std'] = result['Close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    result['BB_upper'] = result['BB_middle'] + (result['BB_std'] * num_std)
    result['BB_lower'] = result['BB_middle'] - (result['BB_std'] * num_std)
    
    return result

def calculate_volume_metrics(df, window=20):
    """
    Calculate volume-based metrics.
    
    Args:
        df: DataFrame with stock price data (must include 'Volume' column)
        window: Rolling window for volume metrics
        
    Returns:
        DataFrame with original data and volume metrics added
    """
    result = df.copy()
    
    # Calculate volume moving average
    result['Volume_MA'] = result['Volume'].rolling(window=window).mean()
    
    # Calculate volume ratio (current volume / average volume)
    result['Volume_Ratio'] = result['Volume'] / result['Volume_MA']
    
    # Calculate on-balance volume (OBV)
    obv = 0
    obvs = []
    
    for i, row in result.iterrows():
        if i > 0:
            prev_close = result.loc[result.index[i-1], 'Close']
            if row['Close'] > prev_close:
                obv += row['Volume']
            elif row['Close'] < prev_close:
                obv -= row['Volume']
        obvs.append(obv)
    
    result['OBV'] = obvs
    
    return result

def calculate_volatility(df, window=20):
    """
    Calculate various volatility metrics.
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        window: Rolling window for volatility calculations
        
    Returns:
        DataFrame with original data and volatility metrics added
    """
    result = df.copy()
    
    # Calculate daily returns
    result['Daily_Return'] = result['Close'].pct_change() * 100
    
    # Calculate rolling volatility (standard deviation of returns)
    result['Volatility'] = result['Daily_Return'].rolling(window=window).std()
    
    # Calculate average true range (ATR)
    result['TR'] = np.maximum(
        result['High'] - result['Low'],
        np.maximum(
            abs(result['High'] - result['Close'].shift(1)),
            abs(result['Low'] - result['Close'].shift(1))
        )
    )
    result['ATR'] = result['TR'].rolling(window=window).mean()
    
    return result

def calculate_momentum_indicators(df, window=14):
    """
    Calculate momentum indicators.
    
    Args:
        df: DataFrame with stock price data (must include 'Close' column)
        window: Rolling window for momentum calculations
        
    Returns:
        DataFrame with original data and momentum indicators added
    """
    result = df.copy()
    
    # Calculate Rate of Change (ROC)
    result['ROC'] = (result['Close'] / result['Close'].shift(window) - 1) * 100
    
    # Calculate Stochastic Oscillator
    result['Lowest_Low'] = result['Low'].rolling(window=window).min()
    result['Highest_High'] = result['High'].rolling(window=window).max()
    result['Stoch_K'] = 100 * ((result['Close'] - result['Lowest_Low']) / 
                              (result['Highest_High'] - result['Lowest_Low']))
    result['Stoch_D'] = result['Stoch_K'].rolling(window=3).mean()
    
    return result

def calculate_trend_indicators(df):
    """
    Calculate trend indicators.
    
    Args:
        df: DataFrame with stock price data (must include appropriate columns)
        
    Returns:
        DataFrame with original data and trend indicators added
    """
    result = df.copy()
    
    # Calculate price position relative to moving averages
    if 'MA_50' in result.columns and 'MA_200' in result.columns:
        # Golden Cross / Death Cross signal
        result['MA_50_200_Signal'] = np.where(result['MA_50'] > result['MA_200'], 1, -1)
        
        # Price position relative to MAs
        result['Price_Above_MA50'] = np.where(result['Close'] > result['MA_50'], 1, 0)
        result['Price_Above_MA200'] = np.where(result['Close'] > result['MA_200'], 1, 0)
    
    return result

def calculate_all_metrics(df):
    """
    Calculate all metrics in one function.
    
    Args:
        df: DataFrame with stock price data (must include OHLCV columns)
        
    Returns:
        DataFrame with all metrics added
    """
    result = df.copy()
    
    # Apply all metrics calculations
    result = calculate_moving_averages(result)
    result = calculate_rsi(result)
    result = calculate_macd(result)
    result = calculate_bollinger_bands(result)
    result = calculate_volume_metrics(result)
    result = calculate_volatility(result)
    result = calculate_momentum_indicators(result)
    result = calculate_trend_indicators(result)
    
    return result

def get_current_signals(df):
    """
    Get current trading signals based on technical indicators.
    
    Args:
        df: DataFrame with calculated metrics
        
    Returns:
        Dictionary with signal interpretations
    """
    signals = {}
    
    # Get the most recent complete row with data
    latest = df.iloc[-1]
    
    # RSI signals
    if 'RSI' in df.columns:
        if latest['RSI'] < 30:
            signals['RSI'] = 'Oversold (Buy)'
        elif latest['RSI'] > 70:
            signals['RSI'] = 'Overbought (Sell)'
        else:
            signals['RSI'] = 'Neutral'
    
    # MACD signals
    if 'MACD_line' in df.columns and 'MACD_signal' in df.columns:
        # Check for crossover
        if df['MACD_line'].iloc[-1] > df['MACD_signal'].iloc[-1] and df['MACD_line'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
            signals['MACD'] = 'Bullish Crossover (Buy)'
        elif df['MACD_line'].iloc[-1] < df['MACD_signal'].iloc[-1] and df['MACD_line'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
            signals['MACD'] = 'Bearish Crossover (Sell)'
        elif df['MACD_line'].iloc[-1] > df['MACD_signal'].iloc[-1]:
            signals['MACD'] = 'Bullish'
        else:
            signals['MACD'] = 'Bearish'
    
    # Bollinger Bands signals
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        if latest['Close'] > latest['BB_upper']:
            signals['Bollinger'] = 'Above Upper Band (Overbought)'
        elif latest['Close'] < latest['BB_lower']:
            signals['Bollinger'] = 'Below Lower Band (Oversold)'
        else:
            signals['Bollinger'] = 'Within Bands'
    
    # Moving Average signals
    if 'MA_50' in df.columns and 'MA_200' in df.columns:
        if latest['MA_50'] > latest['MA_200']:
            if df['MA_50'].iloc[-2] <= df['MA_200'].iloc[-2]:
                signals['MA_Crossover'] = 'Golden Cross (Bullish)'
            else:
                signals['MA_Crossover'] = 'Bullish Trend'
        else:
            if df['MA_50'].iloc[-2] >= df['MA_200'].iloc[-2]:
                signals['MA_Crossover'] = 'Death Cross (Bearish)'
            else:
                signals['MA_Crossover'] = 'Bearish Trend'
    
    # Stochastic signals
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        if latest['Stoch_K'] < 20:
            signals['Stochastic'] = 'Oversold'
        elif latest['Stoch_K'] > 80:
            signals['Stochastic'] = 'Overbought'
        elif latest['Stoch_K'] > latest['Stoch_D'] and df['Stoch_K'].iloc[-2] <= df['Stoch_D'].iloc[-2]:
            signals['Stochastic'] = 'Bullish Crossover'
        elif latest['Stoch_K'] < latest['Stoch_D'] and df['Stoch_K'].iloc[-2] >= df['Stoch_D'].iloc[-2]:
            signals['Stochastic'] = 'Bearish Crossover'
        else:
            signals['Stochastic'] = 'Neutral'
    
    # Volume signals
    if 'Volume_Ratio' in df.columns:
        if latest['Volume_Ratio'] > 2:
            signals['Volume'] = 'Unusually High'
        elif latest['Volume_Ratio'] < 0.5:
            signals['Volume'] = 'Unusually Low'
        else:
            signals['Volume'] = 'Normal'
    
    return signals

def calculate_support_resistance(df, window=20):
    """
    Calculate simple support and resistance levels.
    
    Args:
        df: DataFrame with stock price data
        window: Window to look for local maxima/minima
        
    Returns:
        Dictionary with support and resistance levels
    """
    result = {}
    
    # Find local maxima and minima
    highs = []
    lows = []
    
    for i in range(window, len(df) - window):
        # Check if it's a local maximum
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            highs.append(df['High'].iloc[i])
        
        # Check if it's a local minimum
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            lows.append(df['Low'].iloc[i])
    
    # Get the most recent price
    latest_price = df['Close'].iloc[-1]
    
    # Find resistance levels (above current price)
    resistance_levels = [h for h in highs if h > latest_price]
    resistance_levels.sort()
    
    # Find support levels (below current price)
    support_levels = [l for l in lows if l < latest_price]
    support_levels.sort(reverse=True)
    
    # Get the closest levels
    result['resistance'] = resistance_levels[:3] if resistance_levels else []
    result['support'] = support_levels[:3] if support_levels else []
    
    return result 