import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_stock_price(data, ticker, indicators=None):
    """
    Plot stock price with selected indicators
    
    Args:
        data (DataFrame): Stock price data with indicators
        ticker (str): Stock ticker symbol
        indicators (list): List of indicators to include in the plot
        
    Returns:
        Plotly figure
    """
    if indicators is None:
        indicators = []
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart for OHLC data
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        )
    )
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            yaxis='y2'
        )
    )
    
    # Add selected indicators
    for indicator in indicators:
        if indicator in data.columns:
            # Special case for RSI (needs its own y-axis)
            if indicator == 'RSI':
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[indicator],
                        name=indicator,
                        yaxis='y3'
                    )
                )
                
                # Add horizontal lines for overbought/oversold levels
                fig.add_shape(
                    type="line",
                    y0=70, y1=70, x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],
                    yref='y3'
                )
                fig.add_shape(
                    type="line",
                    y0=30, y1=30, x0=data['Date'].iloc[0], x1=data['Date'].iloc[-1],
                    yref='y3'
                )
                
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[indicator],
                        name=indicator
                    )
                )
    
    # Create layout with multiple y-axes
    fig.update_layout(
        title=f'{ticker} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            domain=[0.3, 1.0],
            showticklabels=True
        ),
        yaxis2=dict(
            domain=[0, 0.2],
            showticklabels=True,
            title='Volume',
            anchor='x',
            overlaying='y',
            side='right'
        ),
        height=600,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1)
    )
    
    # Add RSI y-axis if RSI is in the indicators
    if 'RSI' in indicators and 'RSI' in data.columns:
        fig.update_layout(
            yaxis3=dict(
                domain=[0.75, 0.95],
                range=[0, 100],
                showticklabels=True,
                title='RSI',
                anchor='free',
                overlaying='y',
                side='right',
                position=0.95
            )
        )
    
    return fig

def plot_technical_analysis(df, ticker):
    """
    Create a comprehensive technical analysis chart with multiple indicators
    
    Args:
        df (pandas.DataFrame): Stock price data with technical indicators
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Interactive technical analysis chart
    """
    if df.empty:
        return go.Figure()
    
    # Create figure with subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=(f'{ticker} Price & MAs', 'MACD', 'RSI', 'Volume'))
    
    # Add candlestick chart with Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Add Bollinger Bands
    if 'Bollinger_High' in df.columns and 'Bollinger_Low' in df.columns and 'Bollinger_Mid' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Bollinger_High'],
            name='Bollinger High'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Bollinger_Low'],
            name='Bollinger Low'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Bollinger_Mid'],
            name='Bollinger Mid'
        ), row=1, col=1)
    
    # Add Moving Averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_20'],
            name='SMA 20'
        ), row=1, col=1)
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_50'],
            name='SMA 50'
        ), row=1, col=1)
    
    # Add MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'],
            name='MACD'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD_Signal'],
            name='MACD Signal'
        ), row=2, col=1)
        
        # Add MACD histogram as bar chart
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['MACD_Histogram'],
            name='MACD Histogram'
        ), row=2, col=1)
    
    # Add RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            name='RSI'
        ), row=3, col=1)
        
        # Add RSI levels
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[70, 70],
            line=dict(dash='dash'),
            name='Overbought'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[30, 30],
            line=dict(dash='dash'),
            name='Oversold'
        ), row=3, col=1)
    
    # Add Volume
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            name='Volume'
        ), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis4_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def plot_correlation_matrix(correlation_matrix):
    """
    Plot a correlation matrix as a heatmap
    
    Args:
        correlation_matrix (DataFrame): Correlation matrix
        
    Returns:
        Plotly figure
    """
    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=correlation_matrix.round(2).values,
        texttemplate='%{text}',
        hovertemplate='%{x} and %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Correlation Matrix',
        xaxis_title='Stock',
        yaxis_title='Stock',
        height=600,
        width=800
    )
    
    return fig

def plot_performance_comparison(stocks_data):
    """
    Plot normalized price performance comparison for multiple stocks
    
    Args:
        stocks_data (dict): Dictionary of dataframes with stock price data
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Process each stock
    for ticker, data in stocks_data.items():
        if not data.empty and 'Close' in data.columns:
            # Normalize to the first day (100%)
            normalized = data['Close'] / data['Close'].iloc[0] * 100
            
            # Add line for this stock
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=normalized,
                    name=ticker,
                    mode='lines',
                    hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
                )
            )
    
    # Update layout
    fig.update_layout(
        title='Normalized Price Performance',
        xaxis_title='Date',
        yaxis_title='Normalized Price (%)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_trading_signals(df, ticker):
    """
    Create a chart showing trading signals based on technical indicators
    
    Args:
        df (pandas.DataFrame): Stock price data with trading signals
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Interactive chart with buy/sell signals
    """
    if df.empty or 'Signal_Combined' not in df.columns:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price'
    ))
    
    # Add buy signals
    buy_signals = df[df['Signal_Combined'] > 0.5]
    fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(
            symbol='triangle-up'
        )
    ))
    
    # Add sell signals
    sell_signals = df[df['Signal_Combined'] < -0.5]
    fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(
            symbol='triangle-down'
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600
    )
    
    return fig

def apply_default_styling(fig, theme="dark"):
    """
    Apply default styling to Plotly figures based on the selected theme
    
    Args:
        fig (plotly.graph_objects.Figure): The figure to style
        theme (str): The theme to apply ('dark' or 'light')
        
    Returns:
        plotly.graph_objects.Figure: The styled figure
    """
    if theme == "dark":
        template = "plotly_dark"
        bg_color = "rgba(0,0,0,0)"
        text_color = "#e0e0e0"
        title_color = "#ffffff"
        grid_color = "rgba(255,255,255,0.1)"
        zero_line_color = "rgba(255,255,255,0.2)"
        legend_bg = "rgba(0,0,0,0.2)"
        legend_border = "rgba(255,255,255,0.2)"
    else:  # light theme
        template = "plotly_white"
        bg_color = "rgba(255,255,255,0)"
        text_color = "#333333"
        title_color = "#111111"
        grid_color = "rgba(0,0,0,0.1)"
        zero_line_color = "rgba(0,0,0,0.2)"
        legend_bg = "rgba(255,255,255,0.7)"
        legend_border = "rgba(0,0,0,0.1)"

    # Apply theme styling
    fig.update_layout(
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        title_font=dict(color=title_color),
        legend=dict(
            font=dict(color=text_color),
            bgcolor=legend_bg,
            bordercolor=legend_border
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=zero_line_color,
            title_font=dict(color=text_color)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=zero_line_color,
            title_font=dict(color=text_color)
        )
    )
    
    # If the figure has multiple axes, style them too
    for axis in fig.layout:
        if (axis.startswith('xaxis') or axis.startswith('yaxis')) and axis not in ['xaxis', 'yaxis']:
            fig.layout[axis].update(
                gridcolor=grid_color,
                zerolinecolor=zero_line_color,
                title_font=dict(color=text_color)
            )
    
    return fig
    
# ... existing code ... 