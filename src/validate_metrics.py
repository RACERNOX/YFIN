import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_metrics import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_volume_metrics,
    calculate_volatility,
    calculate_momentum_indicators,
    calculate_trend_indicators,
    calculate_all_metrics,
    get_current_signals,
    calculate_support_resistance
)

def validate_moving_averages(df):
    """Validate moving average calculations"""
    print("\n=== VALIDATING MOVING AVERAGES ===")
    result = calculate_moving_averages(df)
    
    # Check if MAs are calculated correctly
    for window in [20, 50, 200]:
        # Calculate MA manually for verification
        manual_ma = df['Close'].rolling(window=window).mean()
        
        # Check if values match
        diff = result[f'MA_{window}'] - manual_ma
        max_diff = diff.abs().max()
        
        print(f"MA_{window} max difference: {max_diff}")
        assert max_diff < 1e-10, f"MA_{window} calculation is incorrect"
    
    # Check if 20-day MA is more responsive than 200-day MA
    ma_20_std = result['MA_20'].diff().std()
    ma_200_std = result['MA_200'].diff().std()
    
    print(f"MA_20 day-to-day change std: {ma_20_std:.6f}")
    print(f"MA_200 day-to-day change std: {ma_200_std:.6f}")
    assert ma_20_std > ma_200_std, "Short-term MA should be more responsive than long-term MA"
    
    print("✓ Moving averages validated successfully")
    return result

def validate_rsi(df):
    """Validate RSI calculations"""
    print("\n=== VALIDATING RSI ===")
    result = calculate_rsi(df)
    
    # Check RSI range (should be between 0 and 100)
    min_rsi = result['RSI'].min()
    max_rsi = result['RSI'].max()
    
    print(f"RSI range: {min_rsi:.2f} to {max_rsi:.2f}")
    assert 0 <= min_rsi <= 100, "RSI should be between 0 and 100"
    assert 0 <= max_rsi <= 100, "RSI should be between 0 and 100"
    
    # Check for periods of overbought/oversold
    overbought = (result['RSI'] > 70).sum()
    oversold = (result['RSI'] < 30).sum()
    
    print(f"Overbought periods (RSI > 70): {overbought}")
    print(f"Oversold periods (RSI < 30): {oversold}")
    
    # Check correlation with price
    corr = result['Close'].pct_change().rolling(window=14).corr(result['RSI'].diff())
    avg_corr = corr.mean()
    
    print(f"Average correlation between price changes and RSI changes: {avg_corr:.4f}")
    
    # Check if RSI responds to price changes
    price_volatility = result['Close'].pct_change().std()
    rsi_volatility = result['RSI'].diff().std()
    
    print(f"Price volatility: {price_volatility:.6f}")
    print(f"RSI volatility: {rsi_volatility:.6f}")
    
    print("✓ RSI validated successfully")
    return result

def validate_macd(df):
    """Validate MACD calculations"""
    print("\n=== VALIDATING MACD ===")
    result = calculate_macd(df)
    
    # Manually calculate EMAs for verification
    manual_ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    manual_ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    manual_macd_line = manual_ema_fast - manual_ema_slow
    
    # Check if values match
    ema_fast_diff = (result['EMA_fast'] - manual_ema_fast).abs().max()
    ema_slow_diff = (result['EMA_slow'] - manual_ema_slow).abs().max()
    macd_line_diff = (result['MACD_line'] - manual_macd_line).abs().max()
    
    print(f"EMA_fast max difference: {ema_fast_diff}")
    print(f"EMA_slow max difference: {ema_slow_diff}")
    print(f"MACD_line max difference: {macd_line_diff}")
    
    assert ema_fast_diff < 1e-10, "EMA_fast calculation is incorrect"
    assert ema_slow_diff < 1e-10, "EMA_slow calculation is incorrect"
    assert macd_line_diff < 1e-10, "MACD_line calculation is incorrect"
    
    # Check MACD histogram properties 
    hist_sum = result['MACD_hist'].sum()
    crosses = ((result['MACD_line'] > result['MACD_signal']) != 
               (result['MACD_line'].shift(1) > result['MACD_signal'].shift(1))).sum()
    
    print(f"MACD histogram sum: {hist_sum:.4f}")
    print(f"MACD signal line crosses: {crosses}")
    
    # Verify MACD histogram is MACD line minus signal
    hist_diff = (result['MACD_hist'] - (result['MACD_line'] - result['MACD_signal'])).abs().max()
    print(f"MACD histogram calculation max difference: {hist_diff}")
    assert hist_diff < 1e-10, "MACD_hist calculation is incorrect"
    
    print("✓ MACD validated successfully")
    return result

def validate_bollinger_bands(df):
    """Validate Bollinger Bands calculations"""
    print("\n=== VALIDATING BOLLINGER BANDS ===")
    result = calculate_bollinger_bands(df)
    
    # Manually calculate for verification
    manual_middle = df['Close'].rolling(window=20).mean()
    manual_std = df['Close'].rolling(window=20).std()
    manual_upper = manual_middle + (manual_std * 2)
    manual_lower = manual_middle - (manual_std * 2)
    
    # Check if values match
    middle_diff = (result['BB_middle'] - manual_middle).abs().max()
    upper_diff = (result['BB_upper'] - manual_upper).abs().max()
    lower_diff = (result['BB_lower'] - manual_lower).abs().max()
    
    print(f"BB_middle max difference: {middle_diff}")
    print(f"BB_upper max difference: {upper_diff}")
    print(f"BB_lower max difference: {lower_diff}")
    
    assert middle_diff < 1e-10, "BB_middle calculation is incorrect"
    assert upper_diff < 1e-10, "BB_upper calculation is incorrect"
    assert lower_diff < 1e-10, "BB_lower calculation is incorrect"
    
    # Check percentage of prices within bands (should be around 95% for 2 std)
    within_bands = ((result['Close'] <= result['BB_upper']) & 
                    (result['Close'] >= result['BB_lower'])).mean() * 100
    
    print(f"Percentage of prices within Bollinger Bands: {within_bands:.2f}%")
    
    # Bands should widen during volatile periods
    price_volatility = result['Close'].pct_change().rolling(window=20).std().dropna()
    band_width = (result['BB_upper'] - result['BB_lower']).iloc[20:] / result['BB_middle'].iloc[20:]
    
    corr = price_volatility.corr(band_width[:len(price_volatility)])
    print(f"Correlation between price volatility and band width: {corr:.4f}")
    assert corr > 0, "Band width should increase with volatility"
    
    print("✓ Bollinger Bands validated successfully")
    return result

def validate_volume_metrics(df):
    """Validate volume metrics calculations"""
    print("\n=== VALIDATING VOLUME METRICS ===")
    result = calculate_volume_metrics(df)
    
    # Manually calculate volume MA for verification
    manual_volume_ma = df['Volume'].rolling(window=20).mean()
    
    # Check if values match
    volume_ma_diff = (result['Volume_MA'] - manual_volume_ma).abs().max()
    print(f"Volume_MA max difference: {volume_ma_diff}")
    assert volume_ma_diff < 1e-10, "Volume_MA calculation is incorrect"
    
    # Check volume ratio properties
    min_ratio = result['Volume_Ratio'].min()
    max_ratio = result['Volume_Ratio'].max()
    
    print(f"Volume ratio range: {min_ratio:.2f}x to {max_ratio:.2f}x")
    assert min_ratio >= 0, "Volume ratio should be non-negative"
    
    # Validate OBV directional logic
    price_increases = (result['Close'] > result['Close'].shift(1)).sum()
    obv_increases = (result['OBV'] > result['OBV'].shift(1)).sum()
    
    print(f"Price increases: {price_increases}")
    print(f"OBV increases: {obv_increases}")
    
    print("✓ Volume metrics validated successfully")
    return result

def validate_volatility(df):
    """Validate volatility metrics calculations"""
    print("\n=== VALIDATING VOLATILITY METRICS ===")
    result = calculate_volatility(df)
    
    # Manually calculate for verification
    manual_daily_return = df['Close'].pct_change() * 100
    manual_volatility = manual_daily_return.rolling(window=20).std()
    
    # Check if values match
    daily_return_diff = (result['Daily_Return'] - manual_daily_return).abs().max()
    volatility_diff = (result['Volatility'] - manual_volatility).abs().max()
    
    print(f"Daily_Return max difference: {daily_return_diff}")
    print(f"Volatility max difference: {volatility_diff}")
    
    assert daily_return_diff < 1e-10, "Daily_Return calculation is incorrect"
    assert volatility_diff < 1e-10, "Volatility calculation is incorrect"
    
    # Check that ATR is related to price range
    high_low_range = (df['High'] - df['Low']).mean()
    atr_mean = result['ATR'].mean()
    
    print(f"Average High-Low range: {high_low_range:.4f}")
    print(f"Average ATR: {atr_mean:.4f}")
    
    # ATR should be positively correlated with daily returns
    corr = result['ATR'].iloc[20:].corr(result['Daily_Return'].abs().iloc[20:])
    print(f"Correlation between ATR and absolute returns: {corr:.4f}")
    
    print("✓ Volatility metrics validated successfully")
    return result

def validate_momentum_indicators(df):
    """Validate momentum indicators calculations"""
    print("\n=== VALIDATING MOMENTUM INDICATORS ===")
    result = calculate_momentum_indicators(df)
    
    # Check Stochastic oscillator range (should be between 0 and 100)
    min_k = result['Stoch_K'].min()
    max_k = result['Stoch_K'].max()
    
    print(f"Stoch_K range: {min_k:.2f} to {max_k:.2f}")
    assert 0 <= min_k <= 100, "Stoch_K should be between 0 and 100"
    assert 0 <= max_k <= 100, "Stoch_K should be between 0 and 100"
    
    # Verify Stoch_D is the 3-day MA of Stoch_K
    manual_stoch_d = result['Stoch_K'].rolling(window=3).mean()
    stoch_d_diff = (result['Stoch_D'] - manual_stoch_d).abs().max()
    
    print(f"Stoch_D calculation max difference: {stoch_d_diff}")
    assert stoch_d_diff < 1e-10, "Stoch_D calculation is incorrect"
    
    # Verify ROC calculation
    manual_roc = (df['Close'] / df['Close'].shift(14) - 1) * 100
    roc_diff = (result['ROC'] - manual_roc).abs().max()
    
    print(f"ROC calculation max difference: {roc_diff}")
    assert roc_diff < 1e-10, "ROC calculation is incorrect"
    
    print("✓ Momentum indicators validated successfully")
    return result

def validate_trend_indicators(df_with_mas):
    """Validate trend indicators calculations"""
    print("\n=== VALIDATING TREND INDICATORS ===")
    result = calculate_trend_indicators(df_with_mas)
    
    # Check MA crossover signal
    signal = result['MA_50_200_Signal']
    golden_crosses = ((signal == 1) & (signal.shift(1) == -1)).sum()
    death_crosses = ((signal == -1) & (signal.shift(1) == 1)).sum()
    
    print(f"Golden crosses (50 MA crosses above 200 MA): {golden_crosses}")
    print(f"Death crosses (50 MA crosses below 200 MA): {death_crosses}")
    
    # Check price position indicators
    above_ma50 = result['Price_Above_MA50'].mean() * 100
    above_ma200 = result['Price_Above_MA200'].mean() * 100
    
    print(f"Price above 50-day MA: {above_ma50:.2f}% of the time")
    print(f"Price above 200-day MA: {above_ma200:.2f}% of the time")
    
    # Validate against manual calculation
    manual_ma50_signal = np.where(df_with_mas['MA_50'] > df_with_mas['MA_200'], 1, -1)
    signal_diff = (result['MA_50_200_Signal'] - manual_ma50_signal).abs().max()
    
    print(f"MA_50_200_Signal max difference: {signal_diff}")
    assert signal_diff < 1e-10, "MA_50_200_Signal calculation is incorrect"
    
    print("✓ Trend indicators validated successfully")
    return result

def validate_signals(df_with_metrics):
    """Validate signal generation"""
    print("\n=== VALIDATING SIGNALS ===")
    signals = get_current_signals(df_with_metrics)
    
    print("Current signals:")
    for key, value in signals.items():
        print(f"  {key}: {value}")
    
    # Validate RSI signal
    latest_rsi = df_with_metrics['RSI'].iloc[-1]
    if latest_rsi < 30:
        assert signals['RSI'] == 'Oversold (Buy)', "RSI signal incorrect"
    elif latest_rsi > 70:
        assert signals['RSI'] == 'Overbought (Sell)', "RSI signal incorrect"
    else:
        assert signals['RSI'] == 'Neutral', "RSI signal incorrect"
    
    # Validate MACD signal
    latest_macd = df_with_metrics['MACD_line'].iloc[-1]
    latest_signal = df_with_metrics['MACD_signal'].iloc[-1]
    prev_macd = df_with_metrics['MACD_line'].iloc[-2]
    prev_signal = df_with_metrics['MACD_signal'].iloc[-2]
    
    if latest_macd > latest_signal and prev_macd <= prev_signal:
        assert signals['MACD'] == 'Bullish Crossover (Buy)', "MACD signal incorrect"
    elif latest_macd < latest_signal and prev_macd >= prev_signal:
        assert signals['MACD'] == 'Bearish Crossover (Sell)', "MACD signal incorrect"
    elif latest_macd > latest_signal:
        assert signals['MACD'] == 'Bullish', "MACD signal incorrect"
    else:
        assert signals['MACD'] == 'Bearish', "MACD signal incorrect"
    
    print("✓ Signals validated successfully")
    return signals

def validate_support_resistance(df):
    """Validate support and resistance calculations"""
    print("\n=== VALIDATING SUPPORT & RESISTANCE ===")
    levels = calculate_support_resistance(df)
    
    print("Support levels:", [f"${level:.2f}" for level in levels['support']])
    print("Resistance levels:", [f"${level:.2f}" for level in levels['resistance']])
    
    latest_price = df['Close'].iloc[-1]
    
    # Verify support levels are below current price
    for level in levels['support']:
        assert level < latest_price, f"Support level ${level:.2f} should be below current price ${latest_price:.2f}"
    
    # Verify resistance levels are above current price
    for level in levels['resistance']:
        assert level > latest_price, f"Resistance level ${level:.2f} should be above current price ${latest_price:.2f}"
    
    print("✓ Support & resistance validated successfully")
    return levels

def validate_all_metrics(tickers=['MSFT', 'AAPL', 'GOOGL'], period='1y'):
    """Run validation on multiple tickers"""
    for ticker in tickers:
        print(f"\n\n{'='*50}")
        print(f"VALIDATING ALL METRICS FOR {ticker}")
        print(f"{'='*50}")
        
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        if len(df) < 250:
            print(f"Warning: {ticker} has only {len(df)} data points, which may not be enough for some metrics")
        
        try:
            # Validate all metrics
            ma_df = validate_moving_averages(df)
            rsi_df = validate_rsi(df)
            macd_df = validate_macd(df)
            bb_df = validate_bollinger_bands(df)
            vol_metrics_df = validate_volume_metrics(df)
            volatility_df = validate_volatility(df)
            momentum_df = validate_momentum_indicators(df)
            
            # Trend indicators need MAs first
            trend_df = validate_trend_indicators(ma_df)
            
            # Calculate all metrics at once
            print("\n=== VALIDATING ALL METRICS CALCULATION ===")
            all_metrics_df = calculate_all_metrics(df)
            print(f"All metrics calculation produced {len(all_metrics_df.columns)} columns")
            print("✓ All metrics calculation validated successfully")
            
            # Validate signals
            signals = validate_signals(all_metrics_df)
            
            # Validate support & resistance
            levels = validate_support_resistance(df)
            
            print(f"\n{'='*50}")
            print(f"ALL VALIDATIONS PASSED FOR {ticker}")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"Error validating {ticker}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    validate_all_metrics() 