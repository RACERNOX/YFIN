import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from datetime import datetime, timedelta

# Import custom modules
from stock_data import (
    get_stock_data, get_stock_info, save_stock_data, 
    load_stock_data, save_tracked_stocks, load_tracked_stocks, 
    get_latest_prices
)
from analysis import (
    add_technical_indicators, get_support_resistance_levels,
    calculate_performance_metrics, get_trading_signals,
    calculate_correlation_matrix
)
from visualization import (
    plot_stock_price, plot_technical_analysis, 
    plot_correlation_matrix, plot_performance_comparison,
    plot_trading_signals, apply_default_styling
)
from prediction import (
    predict_with_prophet, train_linear_regression_model, 
    predict_next_day
)
from sentiment import (
    get_news_sentiment, analyze_stock_with_sentiment
)
from auth import (
    setup_auth, authenticate_user, save_user, logout_user,
    get_user_stocks, set_user_stocks, get_user_data
)

# Page configuration
st.set_page_config(
    page_title="YFin - Advanced Stock Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up authentication
setup_auth()

# Add theme to session state
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'

# Function to toggle theme
def toggle_theme():
    st.session_state['theme'] = 'light' if st.session_state['theme'] == 'dark' else 'dark'

# Get current theme
current_theme = st.session_state['theme']

# Custom CSS for better styling with theme support
st.markdown(f"""
<style>
    /* Modern color scheme based on theme */
    :root {{
        --primary: {("#2E5EAA" if current_theme == 'dark' else "#1E4E8A")};
        --secondary: {("#5886E3" if current_theme == 'dark' else "#3E76D3")};
        --accent: {("#47B5FF" if current_theme == 'dark' else "#2795DF")};
        --success: {("#28a745" if current_theme == 'dark' else "#198754")};
        --danger: {("#dc3545" if current_theme == 'dark' else "#bb2d3b")};
        --warning: {("#ffc107" if current_theme == 'dark' else "#ffca2c")};
        --light: {("#f8f9fa" if current_theme == 'dark' else "#ffffff")};
        --dark: {("#343a40" if current_theme == 'dark' else "#212529")};
        --text: {("#e0e0e0" if current_theme == 'dark' else "#212529")};
        --border: {("#4e5862" if current_theme == 'dark' else "#dee2e6")};
        --background: {("#121212" if current_theme == 'dark' else "#f9fafb")};
        --card-bg: {("#1e1e1e" if current_theme == 'dark' else "#ffffff")};
    }}
    
    /* Base styling */
    .stApp {{
        background-color: var(--background);
    }}
    
    /* Text colors */
    p, h1, h2, h3, h4, h5, h6, li, span, div {{
        color: var(--text);
    }}
    
    /* Headers */
    .main-header {{
        font-size: 2.5rem;
        color: var(--primary);
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        padding-bottom: 10px;
        border-bottom: 2px solid var(--border);
    }}
    
    .sub-header {{
        font-size: 1.75rem;
        color: var(--text);
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 5px;
        border-bottom: 1px solid var(--border);
    }}
    
    /* Containers */
    .metric-container {{
        background-color: var(--card-bg);
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
        text-align: center;
    }}
    
    .metric-container h4 {{
        font-size: 1.1rem;
        margin-bottom: 12px;
        color: var(--text);
        font-weight: 600;
    }}
    
    .metric-container p {{
        font-size: 1.8rem !important;
        margin: 0;
        padding: 0;
        font-weight: 700;
    }}
    
    .metric-container:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }}
    
    /* Value styling */
    .positive {{
        color: var(--success);
        font-weight: 600;
    }}
    
    .negative {{
        color: var(--danger);
        font-weight: 600;
    }}
    
    .neutral {{
        color: var(--text);
        font-weight: 600;
    }}
    
    /* Authentication forms */
    .auth-form {{
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        border: 1px solid var(--border);
    }}
    
    /* Button styling */
    .stButton > button {{
        background-color: var(--primary);
        color: var(--light);
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    
    /* Input fields */
    .stTextInput > div > div > input {{
        border-radius: 6px;
        border: 1px solid var(--border);
        padding: 10px;
        transition: all 0.2s ease;
        color: var(--text);
        background-color: var(--card-bg);
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(46,94,170,0.2);
    }}
    
    /* Dataframe styling */
    .dataframe {{
        border: 1px solid var(--border);
        border-collapse: collapse;
        width: 100%;
    }}
    
    .dataframe th {{
        background-color: {("rgba(0,0,0,0.2)" if current_theme == 'dark' else "var(--light)")};
        color: var(--text);
        font-weight: bold;
        padding: 12px;
        border: 1px solid var(--border);
        text-align: left;
    }}
    
    .dataframe td {{
        padding: 12px;
        border: 1px solid var(--border);
        color: var(--text);
        background-color: {("rgba(30,30,30,0.3)" if current_theme == 'dark' else "var(--card-bg)")};
    }}
    
    /* Remove extra white boxes - important fix */
    .element-container, .stDataFrame, .stPlotlyChart {{
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        box-shadow: none;
        margin-bottom: 1rem;
        border: none;
    }}
    
    /* Only apply background to data tables */
    [data-testid="stTable"] {{
        background-color: var(--card-bg);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
    }}
    
    /* Fix sidebar appearance */
    [data-testid="stSidebar"] {{
        background-color: {("rgba(18,18,18,0.8)" if current_theme == 'dark' else "#f8f9fa")};
        border-right: 1px solid var(--border);
    }}
    
    /* Multi-select dropdown */
    .stMultiSelect > div > div {{
        background-color: var(--card-bg);
        border-radius: 6px;
        border: 1px solid var(--border);
    }}
    
    /* Sliders */
    .stSlider > div > div {{
        color: var(--primary);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: var(--text);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--card-bg);
        color: var(--primary);
        font-weight: bold;
        border-top: 2px solid var(--primary);
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: var(--primary);
    }}
    
    /* Stock ticker symbols */
    .stock-ticker {{
        font-weight: bold;
        font-family: monospace;
        color: var(--primary);
    }}
    
    /* Metrics */
    .stMetric {{
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid var(--border);
    }}
    
    .stMetric label {{
        color: var(--text);
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: var(--text);
    }}
    
    /* Card class for charts and other content */
    .card {{
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }}
    
    /* Additional professional touches */
    .info-box {{
        background-color: {("rgba(71, 181, 255, 0.05)" if current_theme == 'dark' else "rgba(71, 181, 255, 0.1)")};
        border-left: 4px solid var(--accent);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }}
    
    .disclaimer-box {{
        background-color: {("rgba(220, 53, 69, 0.03)" if current_theme == 'dark' else "rgba(220, 53, 69, 0.05)")};
        border: 1px solid {("rgba(220, 53, 69, 0.15)" if current_theme == 'dark' else "rgba(220, 53, 69, 0.2)")};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }}
    
    .insight-container {{
        background-color: var(--card-bg);
        border-left: 4px solid var(--success);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Fix for numeric formatter in dataframes */
    .dataframe td {{
        white-space: nowrap;
    }}
    
    /* Chart containers - theme aware */
    .stPlotlyChart {{
        background-color: transparent;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 1.5rem;
        box-shadow: {("0 8px 16px rgba(0, 0, 0, 0.3)" if current_theme == 'dark' else "0 8px 16px rgba(0, 0, 0, 0.1)")};
        border: 1px solid var(--border);
        overflow: visible;
    }}
    
    /* Fix selectbox colors */
    .stSelectbox label {{
        color: var(--text) !important;
    }}
    
    .stSelectbox > div > div {{
        background-color: var(--card-bg);
        color: var(--text);
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        position: fixed;
        top: 10px;
        right: 20px;
        z-index: 1000;
        background-color: {("#1e1e1e" if current_theme == 'dark' else "#f0f0f0")};
        color: {("#f0f0f0" if current_theme == 'dark' else "#1e1e1e")};
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 5px 15px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .theme-toggle:hover {{
        background-color: var(--primary);
        color: white;
    }}
</style>

<!-- Theme toggle button -->
<button onclick="window.requestRerun()" class="theme-toggle">
    {("☀️ Light Mode" if current_theme == 'dark' else "🌙 Dark Mode")}
</button>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>YFin - Advanced Stock Tracker</h1>", unsafe_allow_html=True)

# Add theme toggle in sidebar
with st.sidebar:
    st.write("Theme Settings")
    if st.button("🔄 Toggle Theme" if current_theme == 'dark' else "🔄 Toggle Theme"):
        toggle_theme()
        st.rerun()

def login_page():
    """
    Display login page
    """
    st.markdown("<h2 class='sub-header' style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=False):
        st.markdown("<div class='auth-form'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            success, message = authenticate_user(username, password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state['show_signup'] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    """
    Display signup page
    """
    st.markdown("<h2 class='sub-header' style='text-align: center;'>Create an Account</h2>", unsafe_allow_html=True)
    
    with st.form("signup_form", clear_on_submit=True):
        st.markdown("<div class='auth-form'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if submit:
        if not username or not password or not confirm_password:
            st.error("Please fill all required fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        else:
            success, message = save_user(username, password, email)
            if success:
                st.success(message)
                st.session_state['show_signup'] = False
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    st.markdown("Already have an account?")
    if st.button("Log In"):
        st.session_state['show_signup'] = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def dashboard_page(tracked_stocks):
    """
    Display the main dashboard with an overview of all tracked stocks
    """
    st.markdown("<h2 class='sub-header'>Market Overview</h2>", unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state["refresh_time"] = time.time()
        st.rerun()
    
    # Welcome message
    st.markdown(f"<h3>Welcome back, {st.session_state['username']}!</h3>", unsafe_allow_html=True)
    
    # Market indices
    indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
    index_names = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ", "^VIX": "VIX"}
    
    indices_data = {}
    for idx in indices:
        data = get_stock_data(idx, period="5d")
        if not data.empty:
            indices_data[idx] = data
    
    # Display indices in columns
    if indices_data:
        cols = st.columns(len(indices))
        for i, (idx, data) in enumerate(indices_data.items()):
            with cols[i]:
                if not data.empty:
                    last_price = data["Close"].iloc[-1]
                    prev_price = data["Close"].iloc[-2]
                    change = last_price - prev_price
                    change_pct = (change / prev_price) * 100
                    
                    color = "positive" if change >= 0 else "negative"
                    change_text = f"{change:.2f} ({change_pct:.2f}%)"
                    
                    st.metric(
                        label=index_names.get(idx, idx),
                        value=f"${last_price:.2f}",
                        delta=change_text
                    )
    
    # Tracked stocks overview
    st.markdown("<h2 class='sub-header'>Tracked Stocks</h2>", unsafe_allow_html=True)
    
    # Manage tracked stocks (moved from Settings page)
    with st.expander("Manage Tracked Stocks"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_stock = st.text_input("Add new stock (ticker symbol)").upper()
        with col2:
            add_button = st.button("Add")
        
        if add_button and new_stock:
            if new_stock in tracked_stocks:
                st.warning(f"{new_stock} is already in your tracked stocks.")
            else:
                # Verify stock exists
                try:
                    stock = yf.Ticker(new_stock)
                    info = stock.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        # Stock exists, add to tracked stocks
                        new_tracked_stocks = tracked_stocks + [new_stock]
                        set_user_stocks(st.session_state['username'], new_tracked_stocks)
                        st.success(f"{new_stock} added to tracked stocks!")
                        st.rerun()
                    else:
                        st.error(f"Could not find stock with ticker symbol {new_stock}.")
                except:
                    st.error(f"Could not find stock with ticker symbol {new_stock}.")
        
        # Remove stock
        st.markdown("<h4>Remove Stock</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_to_remove = st.selectbox("Select stock to remove", tracked_stocks)
        with col2:
            remove_button = st.button("Remove")
        
        if remove_button and stock_to_remove:
            if len(tracked_stocks) <= 1:
                st.error("You must have at least one stock in your tracked stocks.")
            else:
                new_tracked_stocks = [s for s in tracked_stocks if s != stock_to_remove]
                set_user_stocks(st.session_state['username'], new_tracked_stocks)
                st.success(f"{stock_to_remove} removed from tracked stocks!")
                st.rerun()
    
    # Get latest prices for all tracked stocks
    latest_prices = get_latest_prices(tracked_stocks)
    
    # Display stocks in a table
    if latest_prices:
        # Create DataFrame from latest prices
        data = []
        for ticker, info in latest_prices.items():
            if info["price"] is not None:
                data.append({
                    "Ticker": ticker,
                    "Price": info["price"],
                    "Change": info["change"],
                    "Change %": info["change_percent"]
                })
        
        if data:
            df = pd.DataFrame(data)
            df = df.set_index("Ticker")
            
            # Apply styling based on change direction
            def color_change(val):
                color = "green" if val > 0 else "red" if val < 0 else "gray"
                return f"color: {color}"
            
            styled_df = df.style.format({
                "Price": "${:.2f}",
                "Change": "{:.2f}",
                "Change %": "{:.2f}%"
            }).applymap(color_change, subset=["Change", "Change %"])
            
            st.dataframe(styled_df, use_container_width=True)
    
    # Recent price charts
    st.markdown("<h2 class='sub-header'>Recent Performance</h2>", unsafe_allow_html=True)
    
    # Get data for tracked stocks
    stocks_data = {}
    for ticker in tracked_stocks:
        data = get_stock_data(ticker, period="1mo")
        if not data.empty:
            # Add technical indicators
            data = add_technical_indicators(data)
            stocks_data[ticker] = data
    
    # Create performance comparison chart
    if stocks_data:
        comparison_fig = plot_performance_comparison(stocks_data)
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Stock price cards (3 columns)
    if stocks_data:
        num_columns = 3
        stocks_per_row = (len(tracked_stocks) + num_columns - 1) // num_columns
        
        for i in range(stocks_per_row):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                idx = i * num_columns + j
                if idx < len(tracked_stocks):
                    ticker = tracked_stocks[idx]
                    with cols[j]:
                        if ticker in stocks_data and not stocks_data[ticker].empty:
                            st.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)
                            fig = plot_stock_price(stocks_data[ticker], ticker, indicators=["SMA_20"])
                            st.plotly_chart(fig, use_container_width=True)

def stock_analysis_page(tracked_stocks):
    """
    Display detailed analysis for a selected stock
    """
    st.markdown("<h2 class='sub-header'>Stock Analysis</h2>", unsafe_allow_html=True)
    
    # Stock selector
    ticker = st.selectbox("Select a stock", tracked_stocks)
    
    # Date range selector
    periods = {
        "1 Week": "1wk",
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Maximum": "max"
    }
    selected_period = st.select_slider(
        "Select time period",
        options=list(periods.keys())
    )
    period = periods[selected_period]
    
    # Get stock data
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            data = get_stock_data(ticker, period=period)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}. The stock symbol may be invalid or the data is currently unavailable.")
                return
                
            # Apply technical indicators
            data_with_indicators = add_technical_indicators(data)
            
            # Get stock info
            info = get_stock_info(ticker)
            
            # Performance metrics
            metrics = calculate_performance_metrics(data)
            
            # Support/resistance levels
            levels = get_support_resistance_levels(data)
            
            # Create trading signals
            data_with_signals = get_trading_signals(data_with_indicators)
            
            # Save data for future use
            save_stock_data(ticker, data)
            
        except Exception as e:
            st.error(f"An error occurred while analyzing {ticker}: {str(e)}")
            return
    
    # Display stock info
    if 'info' in locals() and info:
        st.markdown("<h3>Company Information</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Basic Info
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'longName' in info and info['longName'] != 'N/A':
                st.markdown(f"<h4>{info['longName']}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4>{ticker}</h4>", unsafe_allow_html=True)
                
            if 'sector' in info and info['sector'] != 'N/A':
                st.markdown(f"**Sector:** {info['sector']}", unsafe_allow_html=True)
            
            if 'industry' in info and info['industry'] != 'N/A':
                st.markdown(f"**Industry:** {info['industry']}", unsafe_allow_html=True)
                
            if 'website' in info and info['website'] != 'N/A':
                st.markdown(f"**Website:** {info['website']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Column 2: Price Info
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'currentPrice' in info and info['currentPrice'] != 'N/A':
                try:
                    current_price = float(info['currentPrice'])
                    st.markdown(f"**Current Price:** ${current_price:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**Current Price:** {info['currentPrice']}", unsafe_allow_html=True)
            
            if 'fiftyTwoWeekHigh' in info and info['fiftyTwoWeekHigh'] != 'N/A':
                try:
                    high = float(info['fiftyTwoWeekHigh'])
                    st.markdown(f"**52W High:** ${high:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**52W High:** {info['fiftyTwoWeekHigh']}", unsafe_allow_html=True)
                
            if 'fiftyTwoWeekLow' in info and info['fiftyTwoWeekLow'] != 'N/A':
                try:
                    low = float(info['fiftyTwoWeekLow'])
                    st.markdown(f"**52W Low:** ${low:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**52W Low:** {info['fiftyTwoWeekLow']}", unsafe_allow_html=True)
                
            if 'marketCap' in info and info['marketCap'] != 'N/A':
                market_cap_str = format_market_cap(info['marketCap'])
                st.markdown(f"**Market Cap:** {market_cap_str}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Column 3: Financials
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if 'trailingPE' in info and info['trailingPE'] != 'N/A':
                try:
                    pe = float(info['trailingPE'])
                    st.markdown(f"**P/E Ratio:** {pe:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**P/E Ratio:** {info['trailingPE']}", unsafe_allow_html=True)
                
            if 'forwardPE' in info and info['forwardPE'] != 'N/A':
                try:
                    fpe = float(info['forwardPE'])
                    st.markdown(f"**Forward P/E:** {fpe:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**Forward P/E:** {info['forwardPE']}", unsafe_allow_html=True)
                
            if 'dividendYield' in info and info['dividendYield'] != 'N/A':
                if isinstance(info['dividendYield'], (int, float)):
                    st.markdown(f"**Dividend Yield:** {info['dividendYield']:.2f}%", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Dividend Yield:** {info['dividendYield']}", unsafe_allow_html=True)
                
            if 'beta' in info and info['beta'] != 'N/A':
                try:
                    beta = float(info['beta'])
                    st.markdown(f"**Beta:** {beta:.2f}", unsafe_allow_html=True)
                except (ValueError, TypeError):
                    st.markdown(f"**Beta:** {info['beta']}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Could not retrieve company information for {ticker}")
    
    # Performance metrics
    if 'metrics' in locals() and metrics:
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        
        # Use st.columns for a more reliable layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'day_change' in metrics and metrics['day_change'] is not None:
                color = "green" if metrics['day_change'] >= 0 else "red"
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Day Change</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {color};">
                            {metrics['day_change']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col2:
            if 'week_change' in metrics and metrics['week_change'] is not None:
                color = "green" if metrics['week_change'] >= 0 else "red"
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Week Change</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {color};">
                            {metrics['week_change']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col3:
            if 'month_change' in metrics and metrics['month_change'] is not None:
                color = "green" if metrics['month_change'] >= 0 else "red"
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Month Change</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {color};">
                            {metrics['month_change']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col4:
            if 'ytd_change' in metrics and metrics['ytd_change'] is not None:
                color = "green" if metrics['ytd_change'] >= 0 else "red"
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">YTD Change</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {color};">
                            {metrics['ytd_change']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'volatility' in metrics and metrics['volatility'] is not None:
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Volatility (Annual)</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">
                            {metrics['volatility']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col2:
            if 'max_drawdown' in metrics and metrics['max_drawdown'] is not None:
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Maximum Drawdown</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: red;">
                            {metrics['max_drawdown']:.2f}%
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col3:
            if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] is not None:
                st.markdown(
                    f"""
                    <div class="metric-container" style="text-align: center;">
                        <h4 style="margin-bottom: 10px;">Sharpe Ratio</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">
                            {metrics['sharpe_ratio']:.2f}
                        </p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    # Support and resistance levels
    if 'levels' in locals() and levels:
        if levels['support'] or levels['resistance']:
            st.markdown("<h3>Support and Resistance Levels</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("<h4>Support Levels</h4>", unsafe_allow_html=True)
                if levels['support']:
                    for level in levels['support']:
                        st.markdown(f"<p>${level:.2f}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>No support levels detected</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown("<h4>Resistance Levels</h4>", unsafe_allow_html=True)
                if levels['resistance']:
                    for level in levels['resistance']:
                        st.markdown(f"<p>${level:.2f}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>No resistance levels detected</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Stock price chart
    if 'data' in locals() and not data.empty:
        st.markdown("<h3>Price Chart</h3>", unsafe_allow_html=True)
        
        # Indicator selection
        available_indicators = [
            "SMA_20", "SMA_50", "SMA_200", 
            "EMA_12", "EMA_26",
            "Bollinger_High", "Bollinger_Low", "Bollinger_Mid",
            "RSI"
        ]
        
        selected_indicators = st.multiselect(
            "Select indicators to display",
            available_indicators,
            default=["SMA_20", "SMA_50"]
        )
        
        # Show chart
        try:
            fig = plot_stock_price(data_with_indicators, ticker, indicators=selected_indicators)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trading signals if requested
            if st.checkbox("Show trading signals"):
                signals_fig = plot_trading_signals(data_with_signals, ticker)
                st.plotly_chart(signals_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    else:
        st.error(f"No price data available for {ticker}")

def format_market_cap(value):
    """Format market cap value to readable string"""
    if value is None or value == 'N/A':
        return 'N/A'
    
    try:
        # Convert to float if it's not already
        value = float(value)
        
        if value >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        elif value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        else:
            return f"${value:.2f}"
    except (ValueError, TypeError):
        # Return as is if conversion fails
        return str(value)

def technical_indicators_page(tracked_stocks):
    """Display technical indicators page"""
    st.markdown("<h2 class='sub-header'>Technical Indicators</h2>", unsafe_allow_html=True)
    
    # Under development message
    st.markdown("<div style='text-align:center; padding:50px;'>", unsafe_allow_html=True)
    st.markdown("🚧 **Under Development** 🚧", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;'>This feature is coming soon!</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<h3>What to expect</h3>", unsafe_allow_html=True)
        st.markdown("""
        - Detailed technical analysis
        - Multiple indicator visualizations
        - Customizable chart periods
        - Signal generation
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<h3>Available Indicators</h3>", unsafe_allow_html=True)
        st.markdown("""
        - Moving Averages (SMA, EMA)
        - Bollinger Bands
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        """)
        st.markdown("</div>", unsafe_allow_html=True)

def prediction_page(tracked_stocks):
    """
    Display price prediction page with multiple forecasting models
    """
    st.markdown("<h2 class='sub-header'>Price Prediction</h2>", unsafe_allow_html=True)
    
    # Stock selector
    ticker = st.selectbox("Select a stock", tracked_stocks)
    
    # Date range selector for historical data
    periods = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.select_slider(
        "Select historical data period",
        options=list(periods.keys()),
        value="1 Year"
    )
    period = periods[selected_period]
    
    # Prediction model selection
    prediction_model = st.radio(
        "Select prediction model",
        ["Facebook Prophet", "Linear Regression"],
        horizontal=True
    )
    
    # For Prophet model, allow user to select forecast period
    if prediction_model == "Facebook Prophet":
        forecast_days = st.slider("Forecast days", min_value=7, max_value=90, value=30, step=1)
    
    # Get stock data
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            data = get_stock_data(ticker, period=period)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}. The stock symbol may be invalid or the data is currently unavailable.")
                return
                
            # Apply technical indicators
            data_with_indicators = add_technical_indicators(data)
            
            # Get stock info
            info = get_stock_info(ticker)
            
        except Exception as e:
            st.error(f"An error occurred while analyzing {ticker}: {str(e)}")
            return
    
    # Display current price
    if 'Close' in data.columns and not data.empty:
        current_price = data['Close'].iloc[-1]
        st.markdown(f"<h3>Current Price: ${current_price:.2f}</h3>", unsafe_allow_html=True)
    
    # Display prediction based on selected model
    if prediction_model == "Facebook Prophet":
        st.markdown("<h3>Prophet Model Prediction</h3>", unsafe_allow_html=True)
        
        with st.spinner("Generating prediction..."):
            # Run Prophet prediction
            prophet_result = predict_with_prophet(data, ticker, periods=forecast_days)
            
            if prophet_result and 'forecast' in prophet_result and not prophet_result['forecast'].empty:
                # Display the prediction chart
                prophet_fig = prophet_result['fig']
                st.plotly_chart(prophet_fig, use_container_width=True)
                
                # Display forecast table
                last_actual_date = data['Date'].iloc[-1]
                forecast_df = prophet_result['forecast']
                
                # Filter to only show future dates
                future_forecast = forecast_df[forecast_df['ds'] > last_actual_date].copy()
                
                if not future_forecast.empty:
                    # Format the forecast table
                    future_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    future_forecast.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                    
                    # Only show a subset of days to avoid overwhelming the user
                    display_days = min(10, len(future_forecast))
                    display_indices = list(range(0, len(future_forecast), max(1, len(future_forecast) // display_days)))
                    display_forecast = future_forecast.iloc[display_indices].copy()
                    
                    # Format the dates and values
                    display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
                    display_forecast['Predicted Price'] = display_forecast['Predicted Price'].map('${:.2f}'.format)
                    display_forecast['Lower Bound'] = display_forecast['Lower Bound'].map('${:.2f}'.format)
                    display_forecast['Upper Bound'] = display_forecast['Upper Bound'].map('${:.2f}'.format)
                    
                    st.markdown("<h4>Forecast Values</h4>", unsafe_allow_html=True)
                    st.dataframe(display_forecast, use_container_width=True)
                    
                    # Display next day prediction prominently
                    next_day_pred = future_forecast.iloc[0]
                    st.markdown(f"""
                    <div class='metric-container'>
                    <h4>Next Trading Day Prediction</h4>
                    <p><strong>Date:</strong> {next_day_pred['Date'].strftime('%Y-%m-%d')}</p>
                    <p><strong>Predicted Price:</strong> ${next_day_pred['Predicted Price']:.2f}</p>
                    <p><strong>Prediction Interval:</strong> ${next_day_pred['Lower Bound']:.2f} to ${next_day_pred['Upper Bound']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Could not generate future predictions.")
            else:
                st.error("Failed to generate Prophet forecast. Try a different stock or time period.")
    
    elif prediction_model == "Linear Regression":
        st.markdown("<h3>Linear Regression Model Prediction</h3>", unsafe_allow_html=True)
        
        with st.spinner("Training linear regression model..."):
            # Train linear regression model
            lr_result = train_linear_regression_model(data)
            
            if lr_result and 'model' in lr_result:
                # Display the prediction chart
                lr_fig = lr_result['fig'] 
                st.plotly_chart(lr_fig, use_container_width=True)
                
                # Calculate and display model metrics
                st.markdown("<h4>Model Performance</h4>", unsafe_allow_html=True)
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Training RMSE", f"${lr_result['train_rmse']:.2f}")
                with metrics_col2:
                    st.metric("Testing RMSE", f"${lr_result['test_rmse']:.2f}")
                
                # Display next day prediction
                next_day = predict_next_day(data, ticker, model_type='linear')
                
                if next_day and next_day['prediction'] is not None:
                    prediction_value = next_day['prediction']
                    lower_bound = next_day['lower_bound']
                    upper_bound = next_day['upper_bound']
                    
                    # Calculate potential change
                    change = prediction_value - current_price
                    change_percent = (change / current_price) * 100
                    
                    # Determine if prediction is bullish or bearish
                    prediction_type = "Bullish 📈" if change > 0 else "Bearish 📉"
                    
                    st.markdown(f"""
                    <div class='metric-container'>
                    <h4>Next Day Prediction ({prediction_type})</h4>
                    <p><strong>Predicted Price:</strong> ${prediction_value:.2f}</p>
                    <p><strong>Expected Change:</strong> <span style="color: {'green' if change > 0 else 'red'};">${change:.2f} ({change_percent:.2f}%)</span></p>
                    <p><strong>Prediction Interval:</strong> ${lower_bound:.2f} to ${upper_bound:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display feature importance
                    st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
                    
                    feature_importance = lr_result['feature_importance']
                    importance_df = pd.DataFrame({
                        'Feature': list(feature_importance.keys()),
                        'Importance': list(feature_importance.values())
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Only show top 10 features
                    top_features = importance_df.head(10)
                    
                    # Create a bar chart of feature importance
                    fig = px.bar(
                        top_features, 
                        x='Feature', 
                        y='Importance',
                        title='Top 10 Feature Importance',
                        labels={'Importance': 'Coefficient Value (Impact on Price)'},
                        color='Importance',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not generate next day prediction.")
            else:
                st.error("Failed to train linear regression model. Try a different stock or time period.")
    
    # Display disclaimer
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
    <h4>⚠️ Disclaimer</h4>
    <p style="font-size: 0.8em;">
    Stock price predictions are based on historical data and statistical models. 
    They should not be considered as financial advice. Past performance is not 
    indicative of future results. Always do your own research before making 
    investment decisions.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional resources
    with st.expander("About the Prediction Models"):
        st.markdown("""
        ### Facebook Prophet
        Prophet is an open-source forecasting tool developed by Facebook. It's designed for forecasting time series data with strong seasonal patterns and can handle missing values and outliers effectively.
        
        **Strengths:**
        - Handles seasonality automatically
        - Works well with missing data
        - Provides prediction intervals for uncertainty estimation
        
        ### Linear Regression
        Linear regression is a statistical approach that models the relationship between a dependent variable and one or more independent variables.
        
        **Strengths:**
        - Simple to understand and interpret
        - Feature importance is transparent
        - Works well with limited data
        
        Remember that all prediction models have limitations and may not accurately predict market movements during unusual events.
        """)

def sentiment_page(tracked_stocks):
    """Display news sentiment analysis page"""
    st.markdown("<h2 class='sub-header'>News Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    # Stock selection
    if not tracked_stocks:
        st.warning("You don't have any tracked stocks. Add some stocks in the Settings page.")
        return
    
    selected_stock = st.selectbox(
        "Select a stock for analysis",
        options=tracked_stocks
    )
    
    # Create tabs for two different modes
    basic_tab, advanced_tab = st.tabs(["Basic Stock Analysis", "Advanced Ollama Search"])
    
    # Basic sentiment analysis tab
    with basic_tab:
        # Model selection dropdown
        ollama_model = st.selectbox(
            "Select Ollama model",
            options=["llama3.1", "llama2", "mistral", "gemma"],
            index=0,
            help="Choose which Ollama model to use for analysis"
        )
        
        # Add simple parameters for search context
        col1, col2 = st.columns(2)
        with col1:
            include_price_data = st.checkbox("Include recent price data", value=True,
                                           help="Include recent stock price movements in the analysis")
        with col2:
            include_company_updates = st.checkbox("Include recent company updates", value=True,
                                                help="Include recent company news and updates in the analysis")
        
        # Show processing details toggle
        show_debug = st.checkbox("Show processing details", value=False)
        
        # Simple analysis button
        if st.button("Analyze Stock"):
            with st.spinner("Loading..."):
                try:
                    # Get stock data first to display current price information
                    stock_data = get_stock_data(selected_stock, period="30d")
                    
                    if not stock_data.empty:
                        # Show current price information
                        current_price = stock_data['Close'].iloc[-1]
                        prev_price = stock_data['Close'].iloc[-2]
                        price_change = current_price - prev_price
                        price_change_pct = (price_change / prev_price) * 100
                        
                        # Display current price info
                        price_col1, price_col2, price_col3 = st.columns([1,1,1])
                        price_col1.metric("Current Price", f"${current_price:.2f}")
                        price_col2.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                        
                        # Display a simple chart
                        st.subheader(f"{selected_stock} Recent Price Movement")
                        st.line_chart(stock_data['Close'])
                    
                    # Build custom query based on parameters
                    custom_query = f"Give me exactly 4-5 brief key insights about {selected_stock} stock. Focus only on the most important recent information."
                    if include_price_data:
                        custom_query += " Include one insight about recent price movements or technical analysis."
                    if include_company_updates:
                        custom_query += " Include one insight about recent company news, earnings, or events."
                    custom_query += " Keep each insight to one sentence. Format as a bulleted list."
                    
                    # Use the web search for analysis
                    from web_search import OllamaWebSearcher
                    
                    searcher = OllamaWebSearcher(model=ollama_model, debug_to_streamlit=show_debug)
                    result = searcher.search_and_answer(custom_query)
                    
                    # Display the insights
                    st.subheader(f"Key Insights for {selected_stock}")
                    st.markdown(result["answer"])
                    
                    # Display source if available
                    if result.get("search_performed", False) and "source" in result and "source_title" in result:
                        st.caption(f"Source: [{result['source_title']}]({result['source']})")
                    
                except Exception as e:
                    st.error(f"Error analyzing stock: {e}")
                    st.error("Make sure Ollama is installed and running locally.")
    
    # Advanced Ollama search tab
    with advanced_tab:
        st.info("This mode lets you ask Ollama any financial question without restricting to stock analysis. With web search enabled, it can provide up-to-date information.")
        
        # Add model selection dropdown
        ollama_model = st.selectbox(
            "Select Ollama model",
            options=["llama3.1", "llama2", "mistral", "gemma"],
            index=0,
            help="Choose which Ollama model to use for analysis",
            key="advanced_model"
        )
        
        # Web search option
        use_web_search = st.checkbox("Enable web search", value=True, 
                                   help="When enabled, Ollama will search the web for the most up-to-date information")
        
        # Show processing details toggle
        show_debug = st.checkbox("Show processing details", value=False, key="advanced_debug")
        
        # Check if we have previous results in session state
        if 'ollama_results' not in st.session_state:
            st.session_state.ollama_results = None
            st.session_state.last_query = None
        
        # Handle new search or follow-up question
        if st.session_state.ollama_results is not None:
            # Display previous results
            st.subheader("Ollama Insights")
            st.markdown(f"**Previous Query:** {st.session_state.last_query}")
            st.markdown(st.session_state.ollama_results)
            
            # Option for follow-up question
            st.markdown("### Ask a follow-up question")
            follow_up = st.text_input("Enter your follow-up question:")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Ask Follow-up"):
                    if follow_up:
                        with st.spinner("Loading..."):
                            try:
                                if use_web_search:
                                    # Use web search for follow-up
                                    from web_search import OllamaWebSearcher
                                    searcher = OllamaWebSearcher(model=ollama_model, debug_to_streamlit=show_debug)
                                    
                                    # Create a context-aware prompt
                                    context_prompt = f"""
                                    Previous question: {st.session_state.last_query}
                                    Follow-up question: {follow_up}
                                    """
                                    
                                    result = searcher.search_and_answer(context_prompt)
                                    
                                    # Update session state
                                    st.session_state.ollama_results = result["answer"]
                                    st.session_state.last_query = follow_up
                                else:
                                    # Use regular Ollama for follow-up
                                    from search_agent import OllamaSearchAgent
                                    agent = OllamaSearchAgent(model=ollama_model, debug_to_streamlit=show_debug)
                                    
                                    # Create a context-aware prompt
                                    context_prompt = f"""
                                    Previous question: {st.session_state.last_query}
                                    Follow-up question: {follow_up}
                                    
                                    Please answer the follow-up question considering the context 
                                    of the previous question.
                                    """
                                    
                                    insights = agent.get_custom_insights("GENERAL", context_prompt)
                                    
                                    # Update session state
                                    st.session_state.ollama_results = insights['response']
                                    st.session_state.last_query = follow_up
                                
                                # Force a rerun to show new results
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error using Ollama: {e}")
                    else:
                        st.warning("Please enter a follow-up question.")
            
            with col2:
                if st.button("New Search"):
                    # Clear previous results
                    st.session_state.ollama_results = None
                    st.session_state.last_query = None
                    st.experimental_rerun()
        
        else:
            # Custom query input for new search
            st.markdown("### Ask Ollama anything about finance")
            custom_query = st.text_area("Enter your query:", height=100, 
                                      placeholder="Examples:\n- What are the key factors driving tech stocks this quarter?\n- Explain the recent trends in cryptocurrency markets\n- How might rising interest rates affect the banking sector?")
            
            if st.button("Search with Ollama"):
                if not custom_query:
                    st.warning("Please enter a query to search with Ollama.")
                else:
                    try:
                        if use_web_search:
                            # Use web search
                            from web_search import OllamaWebSearcher
                            
                            with st.spinner("Loading..."):
                                searcher = OllamaWebSearcher(model=ollama_model, debug_to_streamlit=show_debug)
                                result = searcher.search_and_answer(custom_query)
                                
                                # Store results in session state for follow-up questions
                                st.session_state.ollama_results = result["answer"]
                                st.session_state.last_query = custom_query
                                
                                # Display insights
                                st.subheader("Ollama Insights")
                                st.markdown(f"**Query:** {custom_query}")
                                
                                if result["search_performed"]:
                                    st.success("Information retrieved from the web")
                                else:
                                    st.info("Answered using model's knowledge (no web search results)")
                                    
                                st.markdown(result["answer"])
                        else:
                            # Use regular Ollama
                            from search_agent import OllamaSearchAgent
                            
                            with st.spinner("Loading..."):
                                agent = OllamaSearchAgent(model=ollama_model, debug_to_streamlit=show_debug)
                                insights = agent.get_custom_insights("GENERAL", custom_query)
                                
                                # Store results in session state for follow-up questions
                                st.session_state.ollama_results = insights['response']
                                st.session_state.last_query = custom_query
                                
                                # Display insights
                                st.subheader("Ollama Insights")
                                st.markdown(f"**Query:** {custom_query}")
                                st.markdown(insights['response'])
                    except Exception as e:
                        st.error(f"Error using Ollama: {e}")
                        st.error("Make sure Ollama is installed and running locally.")

def display_sentiment_results(stock_ticker, sentiment_data, analysis_results):
    """Display sentiment analysis results in a clean format"""
    if sentiment_data and sentiment_data['news_data']:
        col1, col2, col3 = st.columns(3)
        
        # Calculate sentiment score (normalized to 0-100)
        sentiment_score = ((sentiment_data['average_sentiment'] + 1) / 2) * 100
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<h4>Sentiment Score</h4>", unsafe_allow_html=True)
            
            # Create a colored score display
            if sentiment_data['average_sentiment'] > 0.1:
                color = "green"
                rating = "Positive"
            elif sentiment_data['average_sentiment'] < -0.1:
                color = "red"
                rating = "Negative"
            else:
                color = "gray"
                rating = "Neutral"
                
            st.markdown(f"""
                <div style="text-align: center;">
                    <p style="font-size: 2.5rem; font-weight: bold; color: {color};">
                        {sentiment_score:.1f}<span style="font-size: 1rem;">/100</span>
                    </p>
                    <p style="color: {color}; font-weight: bold; margin-top: -15px;">
                        {rating}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<h4>Sentiment Distribution</h4>", unsafe_allow_html=True)
            
            # Create a pie chart for sentiment distribution
            dist = sentiment_data['sentiment_distribution']
            
            fig = px.pie(
                values=[dist['positive'], dist['neutral'], dist['negative']],
                names=['Positive', 'Neutral', 'Negative'],
                color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                hole=0.4
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150)
            fig.update_traces(textinfo='percent+label')
            
            # Apply theme styling
            fig = apply_dark_theme_to_px(fig)
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<h4>News Volume</h4>", unsafe_allow_html=True)
            
            num_articles = len(sentiment_data['news_data'])
            st.markdown(f"""
                <div style="text-align: center;">
                    <p style="font-size: 2.5rem; font-weight: bold;">
                        {num_articles}
                    </p>
                    <p style="margin-top: -15px;">
                        Articles analyzed
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display sentiment over time chart
        st.markdown("<h3>Sentiment Trend Over Time</h3>", unsafe_allow_html=True)
        sentiment_fig = sentiment_data['fig']
        
        # Apply theme styling
        sentiment_fig = apply_dark_theme_to_px(sentiment_fig)
        
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Display price vs sentiment correlation
        if analysis_results and 'fig' in analysis_results:
            st.markdown("<h3>Price vs. Sentiment Correlation</h3>", unsafe_allow_html=True)
            
            # Show correlation plot
            correlation_fig = analysis_results['fig']
            
            # Apply theme styling
            correlation_fig = apply_dark_theme_to_px(correlation_fig)
            
            st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Display news items
        st.markdown("<h3>Recent News Articles</h3>", unsafe_allow_html=True)
        
        news_df = pd.DataFrame(sentiment_data['news_data'])
        if not news_df.empty:
            news_df = news_df.sort_values('date', ascending=False)
            
            for i, row in news_df.iterrows():
                # Determine sentiment color
                if row['sentiment_category'] == 'positive':
                    sentiment_color = 'green'
                    sentiment_icon = '📈'
                elif row['sentiment_category'] == 'negative':
                    sentiment_color = 'red'
                    sentiment_icon = '📉'
                else:
                    sentiment_color = 'gray'
                    sentiment_icon = '⏹️'
                
                # Format date
                date_obj = pd.to_datetime(row['date'])
                date_str = date_obj.strftime('%b %d, %Y')
                
                # Create card for each news item
                st.markdown(f"""
                <div style="border: 1px solid var(--border); border-left: 5px solid {sentiment_color}; 
                        border-radius: 5px; padding: 15px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text); font-size: 0.8rem;">{date_str} • {row['source']}</span>
                        <span style="font-size: 0.9rem;">
                            {sentiment_icon} <span style="color: {sentiment_color};">
                                {row['sentiment_category'].title()} ({row['sentiment']:.2f})
                            </span>
                        </span>
                    </div>
                    <div style="margin: 10px 0;">
                        <strong style="font-size: 1.1rem;">{row['title']}</strong>
                    </div>
                    <a href="{row['url']}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: var(--primary); color: white; 
                                border: none; padding: 5px 10px; border-radius: 4px; 
                                cursor: pointer; font-size: 0.8rem;">
                            Read More
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)

def profile_page():
    """
    Display user profile page
    """
    st.markdown("<h2 class='sub-header'>My Profile</h2>", unsafe_allow_html=True)
    
    # Get user data
    user_data = get_user_data(st.session_state['username'])
    
    if user_data:
        # Create two columns layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown(f"<h3>{user_data['username']}</h3>", unsafe_allow_html=True)
            
            # Display avatar (placeholder)
            st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=150)
            
            st.markdown(f"**User ID:** {user_data['user_id'][:8]}...")
            
            if user_data['email']:
                st.markdown(f"**Email:** {user_data['email']}")
            
            st.markdown(f"**Member since:** {user_data['created_at'][:10]}")
            
            if user_data['last_login']:
                st.markdown(f"**Last login:** {user_data['last_login'][:10]}")
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<h3>Activity Summary</h3>", unsafe_allow_html=True)
            
            # Stock count
            stock_count = len(user_data['preferences'].get('default_stocks', []))
            st.markdown(f"**Tracked stocks:** {stock_count}")
            
            # Display tracked stocks
            st.markdown("**Your stocks:**")
            stocks = user_data['preferences'].get('default_stocks', [])
            stocks_str = ", ".join(stocks)
            st.markdown(f"{stocks_str}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Account actions
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.markdown("<h3>Account Actions</h3>", unsafe_allow_html=True)
            
            if st.button("Edit Profile"):
                st.info("Profile editing is currently under development.")
            
            if st.button("Change Password"):
                st.info("Password changing is currently under development.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Could not load user profile. Please try logging in again.")

def apply_dark_theme_to_px(fig):
    """
    Apply theme styling to Plotly Express figures based on current theme
    
    Args:
        fig (plotly.graph_objects.Figure): The figure to style
        
    Returns:
        plotly.graph_objects.Figure: The styled figure
    """
    # Apply appropriate theme styling based on current theme
    return apply_default_styling(fig, theme=st.session_state.get('theme', 'dark'))

def compare_stocks_page(tracked_stocks):
    """Display comparison between multiple stocks"""
    st.markdown("<h2 class='sub-header'>Compare Stocks</h2>", unsafe_allow_html=True)
    
    # Stock selection
    selected_stocks = st.multiselect(
        "Select stocks to compare (2-5 recommended)",
        options=tracked_stocks,
        default=tracked_stocks[:min(3, len(tracked_stocks))]
    )
    
    if len(selected_stocks) < 2:
        st.warning("Please select at least 2 stocks to compare.")
        return
    
    # Time period selection
    periods = {
        "1 Week": "1wk",
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.select_slider(
        "Select time period",
        options=list(periods.keys())
    )
    period = periods[selected_period]
    
    # Comparison type
    comparison_type = st.radio(
        "Select comparison type",
        ["Price Performance", "Volume Analysis", "Technical Indicators", "Correlation Analysis"],
        horizontal=True
    )
    
    # Get data for selected stocks
    with st.spinner("Loading data for selected stocks..."):
        stocks_data = {}
        for ticker in selected_stocks:
            data = get_stock_data(ticker, period=period)
            if not data.empty:
                # Add technical indicators
                data = add_technical_indicators(data)
                stocks_data[ticker] = data
    
    if not stocks_data:
        st.error("Could not fetch data for any of the selected stocks.")
        return
    
    # Display comparison based on selected type
    if comparison_type == "Price Performance":
        st.markdown("<h3>Price Performance Comparison</h3>", unsafe_allow_html=True)
        
        # Normalize option
        normalize = st.checkbox("Normalize prices (start at 100%)", value=True)
        
        if normalize:
            # Create normalized comparison chart
            comparison_fig = plot_performance_comparison(stocks_data)
        else:
            # Create regular price chart for all selected stocks
            comparison_fig = go.Figure()
            
            for ticker, data in stocks_data.items():
                if not data.empty and 'Close' in data.columns:
                    comparison_fig.add_trace(
                        go.Scatter(
                            x=data['Date'],
                            y=data['Close'],
                            name=f"{ticker} Close",
                            mode='lines'
                        )
                    )
            
            comparison_fig.update_layout(
                title='Stock Price Comparison',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500
            )
        
        # Apply theme styling
        comparison_fig = apply_dark_theme_to_px(comparison_fig)
        
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Display percentage changes
        st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
        
        metrics_data = []
        for ticker, data in stocks_data.items():
            if not data.empty and len(data) > 1:
                # Calculate changes
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                price_change = (end_price - start_price) / start_price * 100
                
                # Calculate volatility
                daily_returns = data['Close'].pct_change().dropna()
                volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
                
                metrics_data.append({
                    'Ticker': ticker,
                    'Start Price': start_price,
                    'End Price': end_price,
                    'Change %': price_change,
                    'Volatility %': volatility
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Style the dataframe
            def color_change(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            styled_df = metrics_df.style.format({
                'Start Price': '${:.2f}',
                'End Price': '${:.2f}',
                'Change %': '{:.2f}%',
                'Volatility %': '{:.2f}%'
            }).applymap(color_change, subset=['Change %'])
            
            st.dataframe(styled_df, use_container_width=True)
    
    elif comparison_type == "Volume Analysis":
        st.markdown("<h3>Trading Volume Comparison</h3>", unsafe_allow_html=True)
        
        # Create volume comparison chart
        volume_fig = go.Figure()
        
        for ticker, data in stocks_data.items():
            if not data.empty and 'Volume' in data.columns:
                volume_fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data['Volume'],
                        name=f"{ticker} Volume",
                        mode='lines'
                    )
                )
        
        volume_fig.update_layout(
            title='Trading Volume Comparison',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=500
        )
        
        # Apply theme styling
        volume_fig = apply_dark_theme_to_px(volume_fig)
        
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Display volume statistics
        st.markdown("<h3>Volume Statistics</h3>", unsafe_allow_html=True)
        
        volume_stats = []
        for ticker, data in stocks_data.items():
            if not data.empty and 'Volume' in data.columns:
                avg_volume = data['Volume'].mean()
                max_volume = data['Volume'].max()
                max_volume_date = data.loc[data['Volume'].idxmax(), 'Date']
                
                volume_stats.append({
                    'Ticker': ticker,
                    'Avg. Daily Volume': avg_volume,
                    'Max Volume': max_volume,
                    'Max Volume Date': max_volume_date.strftime('%Y-%m-%d')
                })
        
        if volume_stats:
            volume_df = pd.DataFrame(volume_stats)
            
            # Format the dataframe
            volume_df['Avg. Daily Volume'] = volume_df['Avg. Daily Volume'].map(lambda x: f"{x:,.0f}")
            volume_df['Max Volume'] = volume_df['Max Volume'].map(lambda x: f"{x:,.0f}")
            
            st.dataframe(volume_df, use_container_width=True)
    
    elif comparison_type == "Technical Indicators":
        st.markdown("<h3>Technical Indicators Comparison</h3>", unsafe_allow_html=True)
        
        # Select indicator to compare
        indicators = [
            "SMA_20", "SMA_50", "SMA_200", 
            "EMA_12", "EMA_26",
            "RSI", "MACD"
        ]
        
        selected_indicator = st.selectbox(
            "Select technical indicator to compare",
            options=indicators
        )
        
        # Create indicator comparison chart
        indicator_fig = go.Figure()
        
        for ticker, data in stocks_data.items():
            if not data.empty and selected_indicator in data.columns:
                indicator_fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[selected_indicator],
                        name=f"{ticker} {selected_indicator}",
                        mode='lines'
                    )
                )
        
        # Add reference lines for RSI
        if selected_indicator == "RSI":
            indicator_fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", annotation_position="right")
            indicator_fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", annotation_position="right")
        
        indicator_fig.update_layout(
            title=f'{selected_indicator} Comparison',
            xaxis_title='Date',
            yaxis_title=selected_indicator,
            height=500
        )
        
        # Apply theme styling
        indicator_fig = apply_dark_theme_to_px(indicator_fig)
        
        st.plotly_chart(indicator_fig, use_container_width=True)
    
    elif comparison_type == "Correlation Analysis":
        st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
        
        # Calculate correlation matrix
        closing_prices = {}
        
        for ticker, data in stocks_data.items():
            if not data.empty and 'Close' in data.columns:
                closing_prices[ticker] = data['Close'].reset_index(drop=True)
        
        if closing_prices:
            # Create DataFrame from closing prices
            prices_df = pd.DataFrame(closing_prices)
            
            # Calculate correlation matrix
            correlation_matrix = prices_df.corr()
            
            # Display correlation matrix as a heatmap
            fig = plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display interpretation
            st.markdown("<h4>Interpretation</h4>", unsafe_allow_html=True)
            st.markdown("""
            - **1.0**: Perfect positive correlation (stocks move in same direction)
            - **0.0**: No correlation (stocks move independently)
            - **-1.0**: Perfect negative correlation (stocks move in opposite directions)
            
            Higher positive correlations may indicate stocks affected by similar factors, while 
            negative correlations can be useful for diversification.
            """)

def main():
    # Check if user is authenticated
    if not st.session_state['authenticated']:
        # Show signup page if requested
        if 'show_signup' not in st.session_state:
            st.session_state['show_signup'] = False
        
        if st.session_state['show_signup']:
            signup_page()
        else:
            login_page()
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = [
        "Dashboard", 
        "Stock Analysis",
        "Price Prediction",
        "News Sentiment",
        "Compare Stocks",
        "Portfolio Tracker",
        "My Profile"
    ]
    selection = st.sidebar.radio("Go to", pages)
    
    # Display username and logout button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()
    
    # Load tracked stocks for the logged-in user
    tracked_stocks = get_user_stocks(st.session_state['username'])
    
    # Handle different pages
    if selection == "Dashboard":
        dashboard_page(tracked_stocks)
    elif selection == "Stock Analysis":
        stock_analysis_page(tracked_stocks)
    elif selection == "Price Prediction":
        prediction_page(tracked_stocks)
    elif selection == "News Sentiment":
        sentiment_page(tracked_stocks)
    elif selection == "Compare Stocks":
        compare_stocks_page(tracked_stocks)
    elif selection == "Portfolio Tracker":
        portfolio_page(tracked_stocks)
    elif selection == "My Profile":
        profile_page()

# Run the application
if __name__ == "__main__":
    main() 