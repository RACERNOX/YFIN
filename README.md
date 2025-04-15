# YFin - Advanced Stock Tracker

A Streamlit-based web application for tracking and analyzing stocks using data from Yahoo Finance.

## Features

- User authentication system with signup and login
- Real-time stock data fetching from Yahoo Finance
- Technical analysis with multiple indicators (RSI, MACD, Bollinger Bands)
- Interactive charts and visualizations
- Sentiment analysis of related news
- Price prediction using Prophet
- Local data storage for tracking performance over time
- User profiles with personalized stock tracking

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Install NLTK resources (first time only):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Running the Application

You can run the application in two ways:

### Option 1: Using the run.py script

```
python run.py
```

### Option 2: Directly with Streamlit

```
streamlit run src/app.py
```

The application will open in your default web browser at http://localhost:8501

## Usage

1. **Create an account** or log in if you already have one
2. **Dashboard**: View an overview of all tracked stocks and market indices
3. **Stock Analysis**: Analyze individual stocks with technical indicators and metrics
4. **Technical Indicators**: View detailed technical analysis for a selected stock
5. **Price Prediction**: Get AI-based price predictions using Prophet and linear regression
6. **News Sentiment**: Analyze news sentiment and its correlation with stock price
7. **Compare Stocks**: Compare performance of multiple stocks side by side
8. **Portfolio Tracker**: Track a portfolio of stocks and analyze performance
9. **Settings**: Configure application settings and manage tracked stocks
10. **My Profile**: View your user profile and account information

## Authentication System

The application includes a user authentication system with the following features:

- **User Registration**: Create an account with username, email, and password
- **Login**: Secure login with password hashing
- **User Profiles**: Each user has their own profile with account information
- **Personalized Stock Lists**: Each user can manage their own list of tracked stocks
- **Session Management**: Stay logged in between sessions

## Project Structure

- `src/` - Source code
  - `app.py` - Main Streamlit application
  - `auth.py` - Authentication system
  - `stock_data.py` - Functions for fetching and processing stock data
  - `analysis.py` - Technical analysis functions
  - `visualization.py` - Chart and visualization components
  - `prediction.py` - AI-based prediction models
  - `sentiment.py` - News sentiment analysis
- `data/` - Local data storage
  - `users/` - User data and preferences

## Dependencies

- Python 3.7+
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- scikit-learn
- prophet
- nltk
- textblob
- ta (Technical Analysis Library) 