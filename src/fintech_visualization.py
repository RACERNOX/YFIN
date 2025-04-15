import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_returns_comparison(viz_data):
    """
    Create a bar chart comparing different return metrics
    
    Args:
        viz_data (dict): Visualization data prepared by prepare_visualization_data
        
    Returns:
        plotly.graph_objects.Figure: Returns comparison chart
    """
    # Create default figure if data is missing
    if not viz_data or 'return_comparison' not in viz_data:
        fig = go.Figure()
        fig.update_layout(
            title="Return Metrics Comparison (No Data Available)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Extract data
    labels = viz_data['return_comparison']['labels']
    values = viz_data['return_comparison']['values']
    
    # Ensure values are numeric
    values = [0 if v is None or pd.isna(v) else v for v in values]
    
    # Create color array based on values
    colors = ['green' if val >= 0 else 'red' for val in values]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='auto',
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Return Metrics Comparison",
        xaxis_title="Time Period",
        yaxis_title="Return (%)",
        height=500,
        xaxis={'categoryorder':'array', 'categoryarray':labels},
        yaxis=dict(
            gridcolor='rgba(230, 230, 230, 0.3)',
            zerolinecolor='rgba(230, 230, 230, 0.3)',
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(labels)-0.5,
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    return fig

def plot_risk_metrics(viz_data):
    """
    Create a visualization of risk metrics
    
    Args:
        viz_data (dict): Visualization data prepared by prepare_visualization_data
        
    Returns:
        plotly.graph_objects.Figure: Risk metrics visualization
    """
    # Extract data
    labels = viz_data['risk_assessment']['labels']
    values = viz_data['risk_assessment']['values']
    
    # Define color scheme based on metric type
    colors = ['#FF6B6B', '#FF9E40', '#FF6B6B', '#FF9E40']
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition='auto',
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Risk Metrics",
        xaxis_title="Metric",
        yaxis_title="Value (%)",
        height=500,
        xaxis={'categoryorder':'array', 'categoryarray':labels},
        yaxis=dict(
            gridcolor='rgba(230, 230, 230, 0.3)',
            zerolinecolor='rgba(230, 230, 230, 0.3)',
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    return fig

def plot_performance_ratios(viz_data):
    """
    Create a radar chart of performance ratios
    
    Args:
        viz_data (dict): Visualization data prepared by prepare_visualization_data
        
    Returns:
        plotly.graph_objects.Figure: Performance ratio radar chart
    """
    # Create default figure if data is missing
    if not viz_data or 'performance_ratios' not in viz_data:
        fig = go.Figure()
        fig.update_layout(
            title="Performance Ratios (No Data Available)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Extract data
    labels = viz_data['performance_ratios']['labels']
    values = viz_data['performance_ratios']['values']
    
    # Replace None with 0
    values = [0 if v is None or pd.isna(v) else v for v in values]
    
    # For negative values, convert to absolute values but remember which ones were negative
    is_negative = [v < 0 for v in values]
    abs_values = [abs(v) for v in values]
    
    # Ensure we have some non-zero values
    if max(abs_values) == 0:
        # If all values are zero, set a small default
        abs_values = [0.01] * len(abs_values)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add radar chart
    fig.add_trace(
        go.Scatterpolar(
            r=abs_values,
            theta=labels,
            fill='toself',
            fillcolor='rgba(67, 147, 195, 0.2)',
            line=dict(color='rgb(67, 147, 195)'),
            marker=dict(
                color=['red' if neg else 'green' for neg in is_negative],
                size=10,
                symbol='circle'
            ),
            text=[f"{'+' if not neg else '-'}{v:.2f}" for neg, v in zip(is_negative, abs_values)],
            hoverinfo='text+theta'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Performance Ratios",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(abs_values) * 1.2 or 0.1]  # Ensure non-zero range
            )
        ),
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    return fig

def plot_benchmark_comparison(viz_data):
    """
    Create visualizations comparing stock performance to benchmark
    
    Args:
        viz_data (dict): Visualization data prepared by prepare_visualization_data
        
    Returns:
        plotly.graph_objects.Figure: Benchmark comparison charts
    """
    # Create default figure if data is missing
    if not viz_data or 'benchmark_data' not in viz_data:
        fig = make_subplots(rows=1, cols=2)
        fig.update_layout(
            title="Benchmark Comparison (No Data Available)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Extract data
    bench_data = viz_data['benchmark_data']
    
    # Check if we have valid benchmark data
    has_valid_data = all(
        key in bench_data and bench_data[key] is not None and not pd.isna(bench_data[key])
        for key in ['stock_return', 'benchmark_return', 'stock_volatility', 'benchmark_volatility']
    )
    
    if not has_valid_data:
        fig = make_subplots(rows=1, cols=2)
        fig.update_layout(
            title="Benchmark Comparison (Insufficient Data)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=("Return vs. Benchmark", "Risk-Return Profile")
    )
    
    # Ensure values are numeric
    stock_return = 0 if bench_data['stock_return'] is None or pd.isna(bench_data['stock_return']) else bench_data['stock_return']
    bench_return = 0 if bench_data['benchmark_return'] is None or pd.isna(bench_data['benchmark_return']) else bench_data['benchmark_return']
    stock_volatility = 0 if bench_data['stock_volatility'] is None or pd.isna(bench_data['stock_volatility']) else bench_data['stock_volatility']
    bench_volatility = 0 if bench_data['benchmark_volatility'] is None or pd.isna(bench_data['benchmark_volatility']) else bench_data['benchmark_volatility']
    
    # 1. Return comparison (bar chart)
    fig.add_trace(
        go.Bar(
            x=['Stock', 'Benchmark'],
            y=[stock_return, bench_return],
            marker_color=['#4393C3', '#5AAE61'],
            text=[f"{stock_return:.2f}%", f"{bench_return:.2f}%"],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # 2. Risk-Return scatter plot
    fig.add_trace(
        go.Scatter(
            x=[bench_volatility, stock_volatility],
            y=[bench_return, stock_return],
            mode='markers+text',
            marker=dict(
                size=15,
                color=['#5AAE61', '#4393C3'],
            ),
            text=['Benchmark', 'Stock'],
            textposition="top center"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Risk (Volatility %)",
        row=1, col=2,
        gridcolor='rgba(230, 230, 230, 0.3)',
        zerolinecolor='rgba(230, 230, 230, 0.3)',
    )
    
    fig.update_yaxes(
        title_text="Return (%)",
        row=1, col=2,
        gridcolor='rgba(230, 230, 230, 0.3)',
        zerolinecolor='rgba(230, 230, 230, 0.3)',
    )
    
    fig.update_yaxes(
        title_text="Annualized Return (%)",
        row=1, col=1
    )
    
    return fig

def plot_efficient_frontier(stock_vol, stock_ret, benchmark_vol, benchmark_ret, risk_free_rate=0.03):
    """
    Create an efficient frontier plot with the stock and benchmark
    
    Args:
        stock_vol (float): Stock volatility
        stock_ret (float): Stock return
        benchmark_vol (float): Benchmark volatility
        benchmark_ret (float): Benchmark return
        risk_free_rate (float): Risk-free rate
        
    Returns:
        plotly.graph_objects.Figure: Efficient frontier plot
    """
    # Check for valid inputs
    if (stock_vol is None or pd.isna(stock_vol) or stock_ret is None or pd.isna(stock_ret) or
        benchmark_vol is None or pd.isna(benchmark_vol) or benchmark_ret is None or pd.isna(benchmark_ret) or
        stock_vol == 0 or benchmark_vol == 0):
        
        fig = go.Figure()
        fig.update_layout(
            title="Efficient Frontier (Insufficient Data)",
            height=600,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Create points for the efficient frontier
    vol_range = np.linspace(0, max(stock_vol, benchmark_vol) * 1.5, 100)
    
    # Calculate the Capital Market Line (CML)
    try:
        sharpe_ratio = (benchmark_ret - risk_free_rate * 100) / benchmark_vol
        cml_returns = risk_free_rate * 100 + sharpe_ratio * vol_range
    except Exception as e:
        print(f"Error calculating CML: {e}")
        # Fallback to simple line
        cml_returns = np.linspace(risk_free_rate * 100, max(stock_ret, benchmark_ret) * 1.2, 100)
    
    # Create figure
    fig = go.Figure()
    
    # Add Capital Market Line
    fig.add_trace(
        go.Scatter(
            x=vol_range,
            y=cml_returns,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='gray', dash='dash')
        )
    )
    
    # Add risk-free rate marker
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[risk_free_rate * 100],  # Convert to percentage
            mode='markers+text',
            marker=dict(color='gold', size=10),
            text=['Risk-Free Rate'],
            textposition="top right",
            name='Risk-Free Rate'
        )
    )
    
    # Add benchmark marker
    fig.add_trace(
        go.Scatter(
            x=[benchmark_vol],
            y=[benchmark_ret],
            mode='markers+text',
            marker=dict(color='green', size=12),
            text=['Benchmark'],
            textposition="top center",
            name='Benchmark'
        )
    )
    
    # Add stock marker
    fig.add_trace(
        go.Scatter(
            x=[stock_vol],
            y=[stock_ret],
            mode='markers+text',
            marker=dict(color='blue', size=12),
            text=['Stock'],
            textposition="top center",
            name='Stock'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Risk-Return Analysis (Efficient Frontier)",
        xaxis_title="Risk (Annual Volatility %)",
        yaxis_title="Return (Annual %)",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(230, 230, 230, 0.3)',
            zerolinecolor='rgba(230, 230, 230, 0.3)',
        ),
        yaxis=dict(
            gridcolor='rgba(230, 230, 230, 0.3)',
            zerolinecolor='rgba(230, 230, 230, 0.3)',
        ),
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    return fig 