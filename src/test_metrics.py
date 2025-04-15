import yfinance as yf
import pandas as pd
from stock_metrics import calculate_all_metrics, get_current_signals, calculate_support_resistance

def test_stock_metrics(ticker="MSFT", period="6mo"):
    """Test stock metrics calculations for a given ticker"""
    print(f"Testing metrics calculations for {ticker}")
    
    # Get stock data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Calculate all metrics
    metrics_df = calculate_all_metrics(df)
    
    # Get latest signals
    signals = get_current_signals(metrics_df)
    
    # Calculate support and resistance
    levels = calculate_support_resistance(df)
    
    # Print results
    print("\nLatest Values:")
    latest = metrics_df.iloc[-1]
    print(f"Close: ${latest['Close']:.2f}")
    print(f"RSI (14): {latest['RSI']:.2f}")
    print(f"MACD Line: {latest['MACD_line']:.4f}")
    print(f"MACD Signal: {latest['MACD_signal']:.4f}")
    print(f"Bollinger Upper: ${latest['BB_upper']:.2f}")
    print(f"Bollinger Middle: ${latest['BB_middle']:.2f}")
    print(f"Bollinger Lower: ${latest['BB_lower']:.2f}")
    print(f"Volatility (20d): {latest['Volatility']:.2f}%")
    print(f"ATR (20d): ${latest['ATR']:.2f}")
    print(f"Volume Ratio: {latest['Volume_Ratio']:.2f}x")
    
    print("\nSignals:")
    for key, value in signals.items():
        print(f"{key}: {value}")
    
    print("\nSupport Levels:")
    for level in levels['support']:
        print(f"${level:.2f}")
    
    print("\nResistance Levels:")
    for level in levels['resistance']:
        print(f"${level:.2f}")
    
    print("\nMetrics DataFrame Shape:", metrics_df.shape)
    
    return metrics_df

if __name__ == "__main__":
    # Test with Microsoft stock
    metrics_df = test_stock_metrics("MSFT", "1y")
    
    # Show some of the columns
    print("\nAvailable Metrics:")
    print(metrics_df.columns.tolist()) 