import yfinance as yf
import pandas as pd
import os
import datetime
import json

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def get_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Period to fetch data for (default: '1y')
        interval (str): Data interval (default: '1d')
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        stock = yf.Ticker(ticker)
        # Add more debugging
        print(f"Fetching history for {ticker} with period={period}, interval={interval}")
        history = stock.history(period=period, interval=interval)
        
        # Check if data is empty
        if history.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        history = history.reset_index()
        
        # Convert timezone-aware dates to timezone-naive
        if 'Date' in history.columns and not history.empty:
            history['Date'] = history['Date'].dt.tz_localize(None)
            
        return history
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_info(ticker):
    """
    Fetch stock information and company details
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Stock information
    """
    try:
        stock = yf.Ticker(ticker)
        # For newer yfinance versions, explicitly fetch fast info
        info = stock.fast_info
        if not info or len(info) == 0:
            # Fallback to regular info
            info = stock.info
        
        if not info:
            print(f"No info returned for {ticker}")
            return {}
        
        # Extract key information with proper type checking
        result = {}
        for key in [
            'shortName', 'longName', 'sector', 'industry', 'website',
            'marketCap', 'currentPrice', 'regularMarketPrice', 'lastPrice',
            'targetHighPrice', 'targetLowPrice', 'targetMeanPrice',
            'recommendationKey', 'forwardPE', 'trailingPE',
            'dividendYield', 'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'
        ]:
            if key in info:
                result[key] = info[key]
            else:
                result[key] = 'N/A'
        
        # Special handling for current price (fallback to regularMarketPrice or lastPrice)
        if result.get('currentPrice', 'N/A') == 'N/A':
            if result.get('regularMarketPrice', 'N/A') != 'N/A':
                result['currentPrice'] = result['regularMarketPrice']
            elif result.get('lastPrice', 'N/A') != 'N/A':
                result['currentPrice'] = result['lastPrice']
            elif hasattr(info, 'last_price') and info.last_price is not None:
                result['currentPrice'] = info.last_price
        
        # Special handling for dividend yield (convert to percentage)
        if result.get('dividendYield', 'N/A') != 'N/A' and isinstance(result['dividendYield'], (int, float)):
            result['dividendYield'] = result['dividendYield'] * 100
        
        return result
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {}

def save_stock_data(ticker, data):
    """
    Save stock data to local storage
    
    Args:
        ticker (str): Stock ticker symbol
        data (pandas.DataFrame): Stock data to save
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    data.to_csv(file_path, index=False)
    
def load_stock_data(ticker):
    """
    Load stock data from local storage
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        pandas.DataFrame: Stock data or empty DataFrame if file doesn't exist
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=['Date'])
    return pd.DataFrame()

def save_tracked_stocks(ticker_list):
    """
    Save list of stocks being tracked
    
    Args:
        ticker_list (list): List of stock ticker symbols
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "tracked_stocks.json")
    with open(file_path, 'w') as f:
        json.dump(ticker_list, f)
        
def load_tracked_stocks():
    """
    Load list of tracked stocks (fallback function)
    
    Returns:
        list: List of stock ticker symbols
    """
    file_path = os.path.join(DATA_DIR, "tracked_stocks.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Default stocks

def get_latest_prices(tickers):
    """
    Get latest prices for multiple tickers
    
    Args:
        tickers (list): List of stock ticker symbols
        
    Returns:
        dict: Dictionary with ticker symbols as keys and latest prices as values
    """
    result = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            print(f"Fetching latest price for {ticker}")
            hist = stock.history(period="1d")
            
            if not hist.empty:
                # If we have valid data, use iloc instead of positional indexing
                result[ticker] = {
                    'price': hist['Close'].iloc[-1],
                    'change': hist['Close'].iloc[-1] - hist['Open'].iloc[-1],
                    'change_percent': ((hist['Close'].iloc[-1] - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                }
            else:
                # Try to get data from fast_info
                try:
                    fast_info = stock.fast_info
                    if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
                        # We have the price but not the daily change
                        result[ticker] = {
                            'price': fast_info.last_price,
                            'change': 0,  # Default to 0 since we don't have Open price
                            'change_percent': 0  # Default to 0
                        }
                    else:
                        result[ticker] = {'price': None, 'change': None, 'change_percent': None}
                except:
                    result[ticker] = {'price': None, 'change': None, 'change_percent': None}
        except Exception as e:
            print(f"Error fetching latest price for {ticker}: {e}")
            result[ticker] = {'price': None, 'change': None, 'change_percent': None}
    return result 