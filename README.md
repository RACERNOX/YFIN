# YFIN - AI-Driven Financial Analytics Forecasting Platform

https://yfinan.streamlit.app/

A comprehensive financial analysis platform built with Streamlit that provides powerful tools for stock tracking, technical analysis, news sentiment analysis, and price prediction.

## 🚀 Features

- **Real-time Market Data**: Live updates of stock prices and market indices
- **Interactive Dashboard**: Visual overview of your tracked stocks and market performance
- **Advanced Stock Analysis**: Deep dive into individual stock performance with key metrics
- **News Sentiment Analysis**: Analyze how news sentiment affects stock prices
- **Price Prediction Models**: Forecast future stock prices using AI models
- **Stock Comparison**: Side-by-side comparison of multiple stocks
- **Stock Metrics Validation**: Validate and test financial metrics accuracy
- **Web Search Integration**: Search for financial information directly within the app
- **FinTech Analytics Tools**: Specialized financial technology analysis features
- **Theme Support**: Toggle between dark and light modes
- **User Authentication**: Secure account management with personalized stock lists
- **Responsive Design**: Optimized for both desktop and mobile viewing

## 📊 Key Components

### Dashboard
- Market overview with major indices (S&P 500, Dow Jones, NASDAQ, VIX)  
- Stock management directly from the dashboard
- Interactive charts with key technical indicators
- Performance metrics and visual trends

### Stock Analysis
- Comprehensive company information
- Performance metrics (day/week/month/YTD changes)  
- Risk analysis (volatility, maximum drawdown, Sharpe ratio)
- Support/resistance level detection
- Customizable technical charts

### News Sentiment
- Real-time news sentiment scoring
- Sentiment distribution visualization
- Historical sentiment trends
- Price-sentiment correlation analysis
- News aggregation with sentiment indicators

### Price Prediction
- Multiple forecasting models:
  - Facebook Prophet (time-series forecasting)
  - Linear Regression (with feature importance)
- Configurable prediction parameters
- Next-day price predictions with confidence intervals
- Model performance evaluation

### Stock Comparison
- Normalized price performance comparison
- Volatility and risk metrics comparison
- Correlation analysis
- Key financial metrics side-by-side

### FinTech Analysis
- Specialized financial technology metrics
- Advanced visualization for fintech-specific data
- Validation tools for financial data accuracy

## 💻 Installation

### Prerequisites
- Python 3.8+ (Python 3.13 is supported)
- Git

### Setup Steps

1. Clone this repository:
   ```
   git clone https://github.com/RACERNOX/yfin.git
   cd yfin
   ```

2. Create and activate a virtual environment:
   ```
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install NLTK resources for sentiment analysis:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## 🤖 Setting Up Ollama

YFin uses Ollama for enhanced sentiment analysis and news search capabilities. This step is optional but recommended for accessing all features.

### Installing Ollama

1. Visit [Ollama's official website](https://ollama.ai/) and download the installer for your operating system.

2. Follow the installation instructions for your platform:
   - **macOS**: Run the downloaded .dmg file and follow the installer prompts
   - **Windows**: Run the installer .exe file
   - **Linux**: Follow the command-line instructions from the Ollama website

### Configuring Ollama for YFin

1. Pull the recommended model (llama3.1) used by YFin:
   ```
   ollama pull llama3.1
   ```

2. List available models to verify installation:
   ```
   ollama list
   ```
   You should see "llama3.1" in the list of available models.

3. The YFin application will automatically use Ollama if it's running. To manually start the Ollama service:
   ```
   # On macOS/Linux, you can start Ollama with:
   ollama serve
   
   # Keep this terminal window open while using YFin
   ```

4. To test that Ollama is working correctly:
   ```
   ollama run llama3.1
   ```
   Type a test prompt and press Ctrl+C to exit when finished.

When running YFin, the application will automatically attempt to use Ollama for enhanced sentiment analysis if it's available.

## 🚀 Running the Application

Option 1: Using the run script (recommended):
```
python run.py
```

Option 2: Directly with Streamlit:
```
streamlit run src/app.py
```

The application will be available at http://localhost:8501 in your web browser.

### Troubleshooting

If you encounter dependency issues during installation:
- For Python 3.13 users: Some packages may require the latest versions. Try removing version constraints from requirements.txt.
- For Apple Silicon (M1/M2/M3) Mac users: Make sure you're using Python built for ARM architecture.

## 📂 Project Structure

- `src/` - Source code
  - `app.py` - Main Streamlit application
  - `auth.py` - Authentication system
  - `stock_data.py` - Data fetching and processing
  - `analysis.py` - Financial analysis functions
  - `visualization.py` - Charts and data visualization
  - `fintech_visualization.py` - Specialized fintech visualizations
  - `prediction.py` - Price prediction models
  - `sentiment.py` - News sentiment analysis
  - `stock_metrics.py` - Stock performance metrics calculation
  - `validate_metrics.py` - Validation tools for metrics
  - `fintech_analysis.py` - Specialized fintech analysis tools
  - `search_agent.py` - Web search functionality
  - `web_search.py` - Web data retrieval functions
- `data/` - Local data storage
  - `users/` - User profiles and preferences

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **yfinance**: Yahoo Finance data API
- **pandas/numpy**: Data manipulation and analysis
- **Prophet/sklearn**: Predictive modeling
- **TextBlob/NLTK**: Natural language processing for sentiment analysis
- **Requests/BeautifulSoup**: Web scraping and data retrieval
- **Scipy**: Advanced scientific computing
- **Ta**: Technical analysis indicators

## 🙏 Acknowledgments

- Yahoo Finance for providing financial data
- Streamlit for the amazing web app framework
- The open-source community for all the incredible libraries

---

Created with ❤️ by Shubham Solanki
