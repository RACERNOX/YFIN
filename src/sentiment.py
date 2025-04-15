import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Download NLTK resources first time (uncomment for first run)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def get_news_sentiment(ticker, period=30):
    """
    Get sentiment analysis of recent news for a stock
    
    Args:
        ticker (str): Stock ticker symbol
        period (int): Number of days to look back for news
        
    Returns:
        dict: Dictionary with sentiment scores and data
    """
    try:
        # Fetch stock news
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return {
                'ticker': ticker,
                'average_sentiment': None,
                'sentiment_scores': [],
                'news_data': [],
                'sentiment_distribution': {},
                'fig': go.Figure()
            }
        
        # Calculate sentiment for each news item
        news_data = []
        sentiment_scores = []
        
        for item in news:
            # Format date from timestamp
            date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
            
            # Extract title and combine with summary if available
            text = item['title']
            if 'summary' in item and item['summary']:
                text += " " + item['summary']
            
            # Calculate sentiment
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # Store data
            sentiment_scores.append(sentiment)
            news_data.append({
                'date': date,
                'title': item['title'],
                'source': item.get('publisher', 'Unknown'),
                'url': item.get('link', ''),
                'sentiment': sentiment,
                'sentiment_category': categorize_sentiment(sentiment)
            })
        
        # Calculate average sentiment
        average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Create sentiment distribution
        sentiment_distribution = {
            'positive': len([s for s in sentiment_scores if s > 0.1]),
            'negative': len([s for s in sentiment_scores if s < -0.1]),
            'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
        }
        
        # Create visualization
        fig = create_sentiment_visualization(news_data, ticker)
        
        return {
            'ticker': ticker,
            'average_sentiment': average_sentiment,
            'sentiment_scores': sentiment_scores,
            'news_data': news_data,
            'sentiment_distribution': sentiment_distribution,
            'fig': fig
        }
    
    except Exception as e:
        print(f"Error analyzing sentiment for {ticker}: {e}")
        return {
            'ticker': ticker,
            'average_sentiment': None,
            'sentiment_scores': [],
            'news_data': [],
            'sentiment_distribution': {},
            'fig': go.Figure()
        }

def categorize_sentiment(score):
    """
    Categorize sentiment score
    
    Args:
        score (float): Sentiment score
        
    Returns:
        str: Sentiment category
    """
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

def create_sentiment_visualization(news_data, ticker):
    """
    Create visualization of sentiment analysis
    
    Args:
        news_data (list): List of news items with sentiment scores
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    if not news_data:
        return go.Figure()
    
    # Sort by date
    df = pd.DataFrame(news_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create plot
    fig = make_sentiment_plot(df, ticker)
    
    return fig

def make_sentiment_plot(df, ticker):
    """
    Create detailed sentiment visualization
    
    Args:
        df (pandas.DataFrame): News data with sentiment scores
        ticker (str): Stock ticker symbol
        
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    # Create two subplots
    fig = px.scatter(
        df, 
        x='date', 
        y='sentiment',
        color='sentiment_category',
        color_discrete_map={
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        },
        size=[1] * len(df),  # Uniform size
        hover_name='title',
        hover_data={
            'date': True,
            'source': True,
            'sentiment': ':.2f',
            'sentiment_category': True,
            'url': True
        }
    )
    
    # Add a trend line
    if len(df) > 1:
        df['day_num'] = (df['date'] - df['date'].min()).dt.days
        x = df['day_num'].values.reshape(-1, 1)
        y = df['sentiment'].values
        
        if len(x) > 1:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            # Generate trend line points
            x_trend = np.array([df['day_num'].min(), df['day_num'].max()]).reshape(-1, 1)
            y_trend = model.predict(x_trend)
            
            # Map back to dates
            min_date = df['date'].min()
            x_dates = [min_date + timedelta(days=int(x_trend[0])), min_date + timedelta(days=int(x_trend[1]))]
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=x_dates,
                y=y_trend,
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Sentiment Trend'
            ))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} News Sentiment Analysis',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        height=500,
        template='plotly_white'
    )
    
    # Add horizontal lines for sentiment zones
    fig.add_shape(
        type="line",
        x0=df['date'].min(),
        x1=df['date'].max(),
        y0=0.1,
        y1=0.1,
        line=dict(color="green", width=1, dash="dash"),
        name="Positive Threshold"
    )
    
    fig.add_shape(
        type="line",
        x0=df['date'].min(),
        x1=df['date'].max(),
        y0=-0.1,
        y1=-0.1,
        line=dict(color="red", width=1, dash="dash"),
        name="Negative Threshold"
    )
    
    return fig

def analyze_stock_with_sentiment(df, sentiment_data):
    """
    Analyze correlation between stock prices and sentiment
    
    Args:
        df (pandas.DataFrame): Stock price data
        sentiment_data (dict): Sentiment analysis data
        
    Returns:
        dict: Analysis results
    """
    if df.empty or not sentiment_data or not sentiment_data['news_data']:
        return {}
    
    # Create DataFrame with news sentiment
    sentiment_df = pd.DataFrame(sentiment_data['news_data'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Group by date and calculate average sentiment
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
    
    # Prepare stock data
    stock_df = df.copy()
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Merge data
    merged_df = pd.merge(stock_df, daily_sentiment, left_on='Date', right_on='date', how='left')
    
    # Forward fill missing sentiment values
    merged_df['sentiment'] = merged_df['sentiment'].ffill()
    
    # Calculate correlation with returns
    merged_df['Daily_Return'] = merged_df['Close'].pct_change() * 100
    
    # Shift sentiment to analyze predictive power
    merged_df['Prev_Day_Sentiment'] = merged_df['sentiment'].shift(1)
    
    # Calculate correlations
    correlations = {
        'sentiment_price': merged_df[['Close', 'sentiment']].corr().iloc[0, 1],
        'sentiment_return': merged_df[['Daily_Return', 'sentiment']].corr().iloc[0, 1],
        'sentiment_next_day_return': merged_df[['Daily_Return', 'Prev_Day_Sentiment']].dropna().corr().iloc[0, 1]
    }
    
    # Create visualization
    fig = go.Figure()
    
    # Plot stock price
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['Close'],
        mode='lines',
        name='Stock Price',
        line=dict(color='blue'),
        yaxis='y1'
    ))
    
    # Plot sentiment on secondary axis
    fig.add_trace(go.Scatter(
        x=merged_df['Date'],
        y=merged_df['sentiment'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='red'),
        marker=dict(size=5),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Stock Price vs. News Sentiment',
        xaxis_title='Date',
        yaxis=dict(
            title='Price',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Sentiment',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[-1, 1]
        ),
        height=600,
        template='plotly_white'
    )
    
    return {
        'correlations': correlations,
        'merged_data': merged_df,
        'fig': fig
    } 