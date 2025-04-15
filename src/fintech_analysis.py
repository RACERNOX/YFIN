import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import yfinance as yf

def calculate_fintech_metrics(df, benchmark_ticker='^GSPC', risk_free_rate=0.03, confidence_level=0.95):
    """
    Calculate advanced fintech metrics for stock analysis
    
    Args:
        df (pandas.DataFrame): Stock price data with OHLC columns
        benchmark_ticker (str): Ticker symbol for benchmark (default: S&P 500)
        risk_free_rate (float): Annual risk-free rate as decimal (default: 3%)
        confidence_level (float): Confidence level for VaR calculation (default: 95%)
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    # Initialize default result with N/A values
    default_result = {
        'returns': {},
        'risk_metrics': {},
        'performance_metrics': {},
        'financial_ratios': {},
        'trading_signals': {},
        'benchmark_comparison': {}
    }
    
    # Check if dataframe is empty or too small
    if df is None or df.empty or len(df) < 30:
        print("DataFrame is None, empty, or has too few rows")
        return default_result
    
    # Check if required column exists
    if 'Close' not in df.columns:
        print("'Close' column not found in dataframe")
        return default_result
    
    try:
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Sort by date to ensure chronological order
        if 'Date' in result_df.columns:
            result_df = result_df.sort_values('Date')
        
        # Calculate returns
        result_df['Daily_Return'] = result_df['Close'].pct_change()
        result_df['Log_Return'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
        result_df = result_df.dropna()
        
        # Handle case where there's insufficient data after cleaning
        if len(result_df) < 5:
            print("Insufficient data after cleaning NaN values")
            return default_result
        
        # Get benchmark data for the same period
        benchmark = None
        try:
            # Get the date range from our stock data
            if 'Date' in result_df.columns:
                start_date = result_df['Date'].iloc[0]
                end_date = result_df['Date'].iloc[-1]
                print(f"Downloading benchmark data from {start_date} to {end_date}")
                benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
            else:
                # If no Date column, just get a matching length of data
                print(f"Downloading benchmark data for 5 years")
                benchmark = yf.download(benchmark_ticker, period="5y", progress=False)
                benchmark = benchmark.tail(len(result_df))
            
            if benchmark is not None and not benchmark.empty:
                benchmark['Daily_Return'] = benchmark['Close'].pct_change()
                benchmark = benchmark.dropna()
                
                # Ensure benchmark data isn't too small
                if len(benchmark) < 5:
                    print("Insufficient benchmark data after cleaning")
                    benchmark = None
            else:
                print("Benchmark data is empty or None")
                benchmark = None
        except Exception as e:
            print(f"Error getting benchmark data: {e}")
            benchmark = None
        
        # ---- 1. Returns Calculation ----
        daily_returns = result_df['Daily_Return'].values
        mean_daily_return = np.mean(daily_returns)
        
        # Safety checks for return calculations
        current_price = result_df['Close'].iloc[-1] if not result_df.empty else None
        daily_return = result_df['Daily_Return'].iloc[-1] * 100 if not result_df.empty else None
        
        # Annualized return calculation
        # If we have enough data, use a longer period for annualization to avoid short-term biases
        if len(daily_returns) >= 252:
            # Use the longest period available, up to 3 years, for annualization
            lookback_period = min(len(daily_returns), 252 * 3)
            long_term_returns = daily_returns[-lookback_period:]
            long_term_mean_daily_return = np.mean(long_term_returns)
            annualized_return = (((1 + long_term_mean_daily_return) ** 252) - 1) * 100
        else:
            # If we have less than a year of data, use what we have but note that it's less reliable
            annualized_return = (((1 + mean_daily_return) ** 252) - 1) * 100
        
        returns_data = {
            'last_price': current_price,
            'daily_return': daily_return,
            'weekly_return': calculate_period_return(result_df, 5) * 100 if len(result_df) > 5 else None,
            'monthly_return': calculate_period_return(result_df, 21) * 100 if len(result_df) > 21 else None,
            'quarterly_return': calculate_period_return(result_df, 63) * 100 if len(result_df) > 63 else None,
            'yearly_return': calculate_period_return(result_df, 252) * 100 if len(result_df) > 252 else None,
            'ytd_return': calculate_ytd_return(result_df) * 100,
            'annualized_return': annualized_return
        }
        
        # ---- 2. Risk Metrics ----
        # Standard deviation (volatility)
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Value at Risk (VaR)
        var_historical = calculate_var_historical(daily_returns, confidence_level)
        var_parametric = calculate_var_parametric(daily_returns, confidence_level)
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall
        cvar = calculate_cvar(daily_returns, confidence_level)
        
        # Downside deviation (only negative returns)
        downside_returns = np.array([r for r in daily_returns if r < 0])
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Maximum Drawdown - handle errors
        try:
            cumulative_returns = (1 + result_df['Daily_Return']).cumprod()
            max_return = cumulative_returns.expanding().max()
            drawdown = ((cumulative_returns / max_return) - 1)
            max_drawdown = drawdown.min() * 100
        except Exception as e:
            print(f"Error calculating drawdown: {e}")
            max_drawdown = None
        
        risk_data = {
            'daily_volatility': daily_volatility * 100,
            'annual_volatility': annual_volatility * 100,
            'var_95': var_parametric * 100,  # Daily VaR at 95% confidence
            'cvar_95': cvar * 100,  # Daily CVaR at 95% confidence
            'downside_deviation': downside_deviation * 100,
            'max_drawdown': max_drawdown
        }
        
        # ---- 3. Performance Metrics ----
        # Sharpe Ratio (with actual risk-free rate)
        daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1  # Convert annual to daily
        
        # Handle case where volatility is zero
        sharpe_ratio = None
        if daily_volatility > 0:
            sharpe_ratio = (mean_daily_return - daily_rf) / daily_volatility * np.sqrt(252)
        
        # Sortino Ratio (using downside deviation)
        sortino_ratio = None 
        if downside_deviation > 0:
            sortino_ratio = (mean_daily_return - daily_rf) / downside_deviation * np.sqrt(252)
        
        # Calmar Ratio (return / max drawdown)
        calmar_ratio = None
        if max_drawdown is not None and max_drawdown != 0:
            calmar_ratio = returns_data['annualized_return'] / abs(max_drawdown)
        
        # Information Ratio and Jensen's Alpha (if benchmark available)
        information_ratio = None
        jensens_alpha = None
        treynor_ratio = None
        beta = None
        
        if benchmark is not None and not benchmark.empty:
            # Match dates between stock and benchmark
            if 'Date' in result_df.columns:
                # Convert benchmark index to datetime if needed
                if not isinstance(benchmark.index, pd.DatetimeIndex):
                    benchmark.index = pd.to_datetime(benchmark.index)
                
                try:
                    # Convert stock dates to same format as benchmark dates
                    stock_dates = pd.to_datetime(result_df['Date'])
                    
                    # Get returns for matching dates
                    matched_returns = []
                    matched_benchmark_returns = []
                    
                    for date in stock_dates:
                        # Try to find the closest benchmark date
                        closest_date = find_closest_date(benchmark.index, date)
                        if closest_date is not None:
                            stock_idx = result_df[result_df['Date'] == date].index[0]
                            matched_returns.append(result_df.loc[stock_idx, 'Daily_Return'])
                            matched_benchmark_returns.append(benchmark.loc[closest_date, 'Daily_Return'])
                    
                    if matched_returns and matched_benchmark_returns:
                        # Calculate metrics using matched returns
                        matched_returns = np.array(matched_returns)
                        matched_benchmark_returns = np.array(matched_benchmark_returns)
                        
                        beta = calculate_beta(matched_returns, matched_benchmark_returns)
                        jensens_alpha = calculate_alpha(matched_returns, matched_benchmark_returns, risk_free_rate, beta)
                        information_ratio = calculate_information_ratio(matched_returns, matched_benchmark_returns)
                        treynor_ratio = calculate_treynor_ratio(matched_returns, matched_benchmark_returns, risk_free_rate)
                except Exception as e:
                    print(f"Error matching dates: {e}")
            else:
                # Simplified approach using available data
                try:
                    # Make sure lengths match
                    min_len = min(len(result_df), len(benchmark))
                    stock_returns = result_df['Daily_Return'].iloc[-min_len:].values
                    benchmark_returns = benchmark['Daily_Return'].iloc[-min_len:].values
                    
                    # Calculate metrics
                    beta = calculate_beta(stock_returns, benchmark_returns)
                    jensens_alpha = calculate_alpha(stock_returns, benchmark_returns, risk_free_rate, beta)
                    information_ratio = calculate_information_ratio(stock_returns, benchmark_returns)
                    treynor_ratio = calculate_treynor_ratio(stock_returns, benchmark_returns, risk_free_rate)
                except Exception as e:
                    print(f"Error calculating benchmark metrics: {e}")
        
        performance_data = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'jensens_alpha': jensens_alpha * 100 if jensens_alpha is not None else None,  # Convert to percentage
            'treynor_ratio': treynor_ratio
        }
        
        # ---- 4. Financial Ratios ----
        # Try to get financial data from Yahoo Finance
        stock_ticker = None
        if hasattr(df, 'attrs') and 'ticker' in df.attrs:
            stock_ticker = df.attrs['ticker']
        
        financial_ratios = {
            'price_to_book': None,
            'price_to_earnings': None,
            'price_to_sales': None,
            'peg_ratio': None,
            'dividend_yield': None,
            'debt_to_equity': None,
            'roe': None,
            'roa': None
        }
        
        if stock_ticker:
            try:
                stock = yf.Ticker(stock_ticker)
                info = {}
                
                # Handle both versions of yfinance API
                if hasattr(stock, 'info'):
                    info = stock.info
                elif hasattr(stock, 'fast_info'):
                    info = stock.fast_info
                
                # Update financial ratios with actual data when available
                for ratio, yf_key in [
                    ('price_to_book', 'priceToBook'),
                    ('price_to_earnings', 'trailingPE'),
                    ('price_to_sales', 'priceToSalesTrailing12Months'),
                    ('peg_ratio', 'pegRatio'),
                    ('dividend_yield', 'dividendYield'),
                    ('debt_to_equity', 'debtToEquity'),
                    ('roe', 'returnOnEquity'),
                    ('roa', 'returnOnAssets')
                ]:
                    if yf_key in info and info[yf_key] not in (None, 'N/A'):
                        financial_ratios[ratio] = info[yf_key]
                        
                        # Data validation and correction for financial ratios
                        # Dividend yield - Yahoo Finance sometimes returns as decimal, sometimes as percentage
                        if ratio == 'dividend_yield' and isinstance(financial_ratios[ratio], (int, float)):
                            # If dividend yield is unreasonably high (>20%), assume it's already in percentage
                            if financial_ratios[ratio] < 20:
                                financial_ratios[ratio] *= 100
                            
                        # ROE validation - Yahoo Finance sometimes returns ROE as decimal, sometimes as percentage
                        if ratio == 'roe' and isinstance(financial_ratios[ratio], (int, float)):
                            # If ROE is unreasonably high (>500%), cap it
                            if financial_ratios[ratio] > 5:
                                if financial_ratios[ratio] > 500:
                                    financial_ratios[ratio] = 100  # Cap at 100%
                                else:
                                    financial_ratios[ratio] *= 100  # Convert to percentage
                        
                        # Price to book validation - should rarely be above 100
                        if ratio == 'price_to_book' and isinstance(financial_ratios[ratio], (int, float)):
                            if financial_ratios[ratio] > 100:
                                financial_ratios[ratio] = None  # Likely an error, discard value
            except Exception as e:
                print(f"Error fetching financial ratios: {e}")
        
        # ---- 5. Trading Signals ----
        # Simple moving averages
        try:
            result_df['SMA_50'] = result_df['Close'].rolling(window=50).mean()
            result_df['SMA_200'] = result_df['Close'].rolling(window=200).mean()
            
            # MACD
            result_df['EMA_12'] = result_df['Close'].ewm(span=12, adjust=False).mean()
            result_df['EMA_26'] = result_df['Close'].ewm(span=26, adjust=False).mean()
            result_df['MACD_line'] = result_df['EMA_12'] - result_df['EMA_26']
            result_df['MACD_signal'] = result_df['MACD_line'].ewm(span=9, adjust=False).mean()
            
            # RSI
            delta = result_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Get signals
            latest = result_df.iloc[-1]
            signals = {}
            
            # MA signals
            if 'SMA_50' in latest and 'SMA_200' in latest and not pd.isna(latest['SMA_50']) and not pd.isna(latest['SMA_200']):
                if latest['SMA_50'] > latest['SMA_200']:
                    if result_df['SMA_50'].iloc[-2] <= result_df['SMA_200'].iloc[-2]:
                        signals['MA_Crossover'] = 'Golden Cross (Bullish)'
                    else:
                        signals['MA_Crossover'] = 'Bullish Trend'
                else:
                    if result_df['SMA_50'].iloc[-2] >= result_df['SMA_200'].iloc[-2]:
                        signals['MA_Crossover'] = 'Death Cross (Bearish)'
                    else:
                        signals['MA_Crossover'] = 'Bearish Trend'
            
            # MACD signals
            if 'MACD_line' in latest and 'MACD_signal' in latest and not pd.isna(latest['MACD_line']) and not pd.isna(latest['MACD_signal']):
                if latest['MACD_line'] > latest['MACD_signal']:
                    if result_df['MACD_line'].iloc[-2] <= result_df['MACD_signal'].iloc[-2]:
                        signals['MACD'] = 'Bullish Crossover (Buy)'
                    else:
                        signals['MACD'] = 'Bullish'
                else:
                    if result_df['MACD_line'].iloc[-2] >= result_df['MACD_signal'].iloc[-2]:
                        signals['MACD'] = 'Bearish Crossover (Sell)'
                    else:
                        signals['MACD'] = 'Bearish'
            
            # RSI signals
            if 'RSI' in latest and not pd.isna(latest['RSI']):
                if latest['RSI'] < 30:
                    signals['RSI'] = 'Oversold (Buy)'
                elif latest['RSI'] > 70:
                    signals['RSI'] = 'Overbought (Sell)'
                else:
                    signals['RSI'] = 'Neutral'
        except Exception as e:
            print(f"Error calculating trading signals: {e}")
            signals = {}
        
        # ---- 6. Benchmark Comparison ----
        benchmark_data = {}
        
        if benchmark is not None and not benchmark.empty:
            try:
                # Calculate benchmark metrics
                benchmark_return = benchmark['Daily_Return'].mean()
                benchmark_annual_return = (((1 + benchmark_return) ** 252) - 1) * 100
                benchmark_volatility = benchmark['Daily_Return'].std() * np.sqrt(252) * 100
                
                # Compare performance
                return_difference = returns_data['annualized_return'] - benchmark_annual_return
                
                # Beta and correlation
                if beta is not None:
                    benchmark_data = {
                        'beta': beta,
                        'correlation': calculate_correlation(result_df['Daily_Return'].values, benchmark['Daily_Return'].values),
                        'benchmark_annual_return': benchmark_annual_return,
                        'benchmark_volatility': benchmark_volatility,
                        'outperformance': return_difference
                    }
            except Exception as e:
                print(f"Error calculating benchmark comparison: {e}")
        
        # Compile and return all metrics
        return {
            'returns': returns_data,
            'risk_metrics': risk_data,
            'performance_metrics': performance_data,
            'financial_ratios': financial_ratios,
            'trading_signals': signals,
            'benchmark_comparison': benchmark_data
        }
    
    except Exception as e:
        print(f"Error in calculate_fintech_metrics: {e}")
        import traceback
        traceback.print_exc()
        return default_result

# ===== Helper Functions =====

def calculate_period_return(df, periods):
    """Calculate return over specified number of periods"""
    if len(df) <= periods:
        return None
    
    current_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-periods-1]
    return (current_price / previous_price) - 1

def calculate_ytd_return(df):
    """Calculate year-to-date return"""
    if 'Date' not in df.columns:
        return calculate_period_return(df, min(252, len(df)-1))
    
    # Get current year's first trading day
    current_year = pd.to_datetime(df['Date'].iloc[-1]).year
    start_of_year = df[pd.to_datetime(df['Date']).dt.year == current_year].iloc[0]
    
    current_price = df['Close'].iloc[-1]
    start_price = start_of_year['Close']
    
    return (current_price / start_price) - 1

def calculate_var_historical(returns, confidence_level=0.95):
    """Calculate Value at Risk using historical method"""
    if len(returns) == 0:
        return 0
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_var_parametric(returns, confidence_level=0.95):
    """Calculate Value at Risk using parametric method"""
    if len(returns) == 0:
        return 0
    mean = np.mean(returns)
    std = np.std(returns)
    return mean + stats.norm.ppf(1 - confidence_level) * std

def calculate_cvar(returns, confidence_level=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    if len(returns) == 0:
        return 0
    var = calculate_var_historical(returns, confidence_level)
    # Filter returns <= VaR and handle case where there are no returns meeting the criterion
    cvar_returns = returns[returns <= var]
    if len(cvar_returns) == 0:
        return var  # If no returns below VaR, return VaR as a fallback
    return np.mean(cvar_returns)

def find_closest_date(date_index, target_date):
    """Find the closest date in a DatetimeIndex to a target date"""
    try:
        # Convert to datetime if needed
        target_date = pd.to_datetime(target_date)
        
        # First try exact match
        if target_date in date_index:
            return target_date
        
        # Find closest date
        closest_dates = date_index[date_index.get_indexer([target_date], method='nearest')]
        if not closest_dates.empty:
            return closest_dates[0]
    except Exception as e:
        print(f"Error finding closest date: {e}")
    
    return None

def calculate_beta(stock_returns, benchmark_returns):
    """Calculate beta (systematic risk)"""
    covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    return covariance / variance if variance != 0 else None

def calculate_alpha(stock_returns, benchmark_returns, risk_free_rate, beta):
    """Calculate Jensen's Alpha"""
    # Convert annual risk free rate to daily
    daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1
    
    # Calculate average returns
    avg_stock_return = np.mean(stock_returns)
    avg_benchmark_return = np.mean(benchmark_returns)
    
    # Calculate alpha
    if beta is not None:
        return avg_stock_return - (daily_rf + beta * (avg_benchmark_return - daily_rf))
    return None

def calculate_information_ratio(stock_returns, benchmark_returns):
    """Calculate Information Ratio"""
    # Calculate excess returns
    excess_returns = np.array(stock_returns) - np.array(benchmark_returns)
    
    # Calculate tracking error
    tracking_error = np.std(excess_returns)
    
    # Calculate IR
    if tracking_error != 0:
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)
    return None

def calculate_treynor_ratio(stock_returns, benchmark_returns, risk_free_rate):
    """Calculate Treynor Ratio"""
    # Convert annual risk free rate to daily
    daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1
    
    # Calculate average return
    avg_return = np.mean(stock_returns)
    
    # Calculate beta
    beta = calculate_beta(stock_returns, benchmark_returns)
    
    # Calculate Treynor ratio
    if beta is not None and beta != 0:
        return (avg_return - daily_rf) / beta * np.sqrt(252)
    return None

def calculate_correlation(returns1, returns2):
    """Calculate correlation coefficient between two return series"""
    # Ensure both arrays have the same length
    min_len = min(len(returns1), len(returns2))
    returns1 = returns1[-min_len:]
    returns2 = returns2[-min_len:]
    
    return np.corrcoef(returns1, returns2)[0, 1]

def prepare_visualization_data(metrics):
    """
    Prepare data for visualization based on calculated metrics
    
    Args:
        metrics (dict): Dictionary with calculated fintech metrics
        
    Returns:
        dict: Dictionary with data formatted for visualization
    """
    # Initialize with default values
    viz_data = {
        'return_comparison': {
            'labels': ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly', 'YTD', 'Annualized'],
            'values': [0, 0, 0, 0, 0, 0, 0]
        },
        'risk_assessment': {
            'labels': ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Annual Volatility'],
            'values': [0, 0, 0, 0]
        },
        'performance_ratios': {
            'labels': ['Sharpe', 'Sortino', 'Calmar', 'Information', 'Treynor'],
            'values': [0, 0, 0, 0, 0]
        },
        'benchmark_data': {
            'stock_return': 0,
            'benchmark_return': 0,
            'stock_volatility': 0,
            'benchmark_volatility': 0,
            'beta': 0,
            'correlation': 0,
            'alpha': 0
        }
    }
    
    # Check if all required sections exist
    required_sections = ['returns', 'risk_metrics', 'performance_metrics', 'benchmark_comparison']
    if not all(section in metrics for section in required_sections):
        print("Warning: Missing required metric sections")
        return viz_data
    
    try:
        # Update return comparison data
        returns = metrics.get('returns', {})
        viz_data['return_comparison']['values'] = [
            returns.get('daily_return', 0) if returns.get('daily_return') is not None else 0,
            returns.get('weekly_return', 0) if returns.get('weekly_return') is not None else 0,
            returns.get('monthly_return', 0) if returns.get('monthly_return') is not None else 0,
            returns.get('quarterly_return', 0) if returns.get('quarterly_return') is not None else 0,
            returns.get('yearly_return', 0) if returns.get('yearly_return') is not None else 0,
            returns.get('ytd_return', 0) if returns.get('ytd_return') is not None else 0,
            returns.get('annualized_return', 0) if returns.get('annualized_return') is not None else 0
        ]
        
        # Update risk assessment data
        risk = metrics.get('risk_metrics', {})
        viz_data['risk_assessment']['values'] = [
            risk.get('var_95', 0) if risk.get('var_95') is not None else 0,
            risk.get('cvar_95', 0) if risk.get('cvar_95') is not None else 0,
            risk.get('max_drawdown', 0) if risk.get('max_drawdown') is not None else 0,
            risk.get('annual_volatility', 0) if risk.get('annual_volatility') is not None else 0
        ]
        
        # Update performance ratios data
        performance = metrics.get('performance_metrics', {})
        viz_data['performance_ratios']['values'] = [
            performance.get('sharpe_ratio', 0) if performance.get('sharpe_ratio') is not None else 0,
            performance.get('sortino_ratio', 0) if performance.get('sortino_ratio') is not None else 0,
            performance.get('calmar_ratio', 0) if performance.get('calmar_ratio') is not None else 0,
            performance.get('information_ratio', 0) if performance.get('information_ratio') is not None else 0,
            performance.get('treynor_ratio', 0) if performance.get('treynor_ratio') is not None else 0
        ]
        
        # Update benchmark data
        benchmark = metrics.get('benchmark_comparison', {})
        viz_data['benchmark_data'] = {
            'stock_return': returns.get('annualized_return', 0) if returns.get('annualized_return') is not None else 0,
            'benchmark_return': benchmark.get('benchmark_annual_return', 0) if benchmark.get('benchmark_annual_return') is not None else 0,
            'stock_volatility': risk.get('annual_volatility', 0) if risk.get('annual_volatility') is not None else 0,
            'benchmark_volatility': benchmark.get('benchmark_volatility', 0) if benchmark.get('benchmark_volatility') is not None else 0,
            'beta': benchmark.get('beta', 0) if benchmark.get('beta') is not None else 0,
            'correlation': benchmark.get('correlation', 0) if benchmark.get('correlation') is not None else 0,
            'alpha': performance.get('jensens_alpha', 0) if performance.get('jensens_alpha') is not None else 0
        }
    except Exception as e:
        print(f"Error preparing visualization data: {e}")
    
    return viz_data 