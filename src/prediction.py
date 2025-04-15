import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta

def predict_with_prophet(df, ticker, periods=30):
    """
    Predict future stock prices using Facebook Prophet
    
    Args:
        df (pandas.DataFrame): Stock price data with Date and Close columns
        ticker (str): Stock ticker symbol
        periods (int): Number of days to forecast
        
    Returns:
        dict: Dictionary containing forecast DataFrame and plotly figure
    """
    if df.empty or len(df) < 30:
        return {'forecast': pd.DataFrame(), 'fig': go.Figure()}
    
    # Prepare data for Prophet
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Fit Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Create forecast plot
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tail(periods),
        y=forecast['yhat'].tail(periods),
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add prediction intervals
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'].tail(periods), forecast['ds'].tail(periods).iloc[::-1]]),
        y=pd.concat([forecast['yhat_lower'].tail(periods), forecast['yhat_upper'].tail(periods).iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Prediction Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Price Forecast (Next {periods} Days)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        template='plotly_white',
        showlegend=True
    )
    
    return {
        'forecast': forecast,
        'fig': fig
    }

def create_features(df, target_col='Close'):
    """
    Create features for ML prediction models
    
    Args:
        df (pandas.DataFrame): Stock price data
        target_col (str): Target column for prediction
        
    Returns:
        pandas.DataFrame: DataFrame with added features
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Create lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Create rolling window features
    for window in [5, 10, 20, 50]:
        df_features[f'rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    
    # Calculate price momentum
    for period in [5, 10, 20]:
        df_features[f'momentum_{period}'] = df_features[target_col] - df_features[target_col].shift(period)
    
    # Add day of week and month features
    df_features['day_of_week'] = pd.to_datetime(df_features['Date']).dt.dayofweek
    df_features['month'] = pd.to_datetime(df_features['Date']).dt.month
    
    # Create target (next day's price)
    df_features['target'] = df_features[target_col].shift(-1)
    
    # Drop NaN values
    df_features = df_features.dropna()
    
    return df_features

def train_linear_regression_model(df, target_col='Close', test_size=0.2):
    """
    Train a linear regression model to predict stock prices
    
    Args:
        df (pandas.DataFrame): Stock price data with features
        target_col (str): Target column for prediction
        test_size (float): Proportion of data to use for testing
        
    Returns:
        dict: Dictionary containing model, train/test data, predictions and evaluation metrics
    """
    if df.empty or len(df) < 30:
        return {}
    
    # Create features
    df_features = create_features(df, target_col)
    
    if df_features.empty:
        return {}
    
    # Prepare features and target
    feature_cols = [col for col in df_features.columns if col not in ['Date', target_col, 'target', 'Open', 'High', 'Low', 'Volume']]
    X = df_features[feature_cols]
    y = df_features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Evaluate model
    train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))
    
    # Create plot
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=df_features['Date'].iloc[-len(y_test):],
        y=y_test,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=df_features['Date'].iloc[-len(y_test):],
        y=test_preds,
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    ))
    
    # Update layout
    fig.update_layout(
        title='Linear Regression Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        template='plotly_white'
    )
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_preds': train_preds,
        'test_preds': test_preds,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'feature_importance': dict(zip(feature_cols, model.coef_)),
        'fig': fig
    }

def predict_next_day(df, ticker, model_type='prophet'):
    """
    Predict the next day's stock price
    
    Args:
        df (pandas.DataFrame): Stock price data
        ticker (str): Stock ticker symbol
        model_type (str): Type of model to use ('prophet' or 'linear')
        
    Returns:
        dict: Dictionary containing prediction and confidence interval
    """
    if df.empty or len(df) < 30:
        return {
            'ticker': ticker,
            'prediction': None,
            'lower_bound': None,
            'upper_bound': None
        }
    
    if model_type == 'prophet':
        # Use Prophet for prediction
        prophet_result = predict_with_prophet(df, ticker, periods=1)
        forecast = prophet_result['forecast']
        
        if not forecast.empty:
            last_prediction = forecast.iloc[-1]
            return {
                'ticker': ticker,
                'prediction': last_prediction['yhat'],
                'lower_bound': last_prediction['yhat_lower'],
                'upper_bound': last_prediction['yhat_upper']
            }
    else:
        # Use linear regression for prediction
        lr_result = train_linear_regression_model(df)
        
        if lr_result and 'model' in lr_result:
            # Create features for the most recent day
            latest_features = create_features(df)
            
            if not latest_features.empty:
                feature_cols = [col for col in latest_features.columns if col not in ['Date', 'Close', 'target', 'Open', 'High', 'Low', 'Volume']]
                latest_X = latest_features[feature_cols].iloc[-1].values.reshape(1, -1)
                
                # Make prediction
                prediction = lr_result['model'].predict(latest_X)[0]
                
                # Calculate confidence interval (using RMSE as a simple approximation)
                rmse = lr_result['test_rmse']
                
                return {
                    'ticker': ticker,
                    'prediction': prediction,
                    'lower_bound': prediction - 1.96 * rmse,
                    'upper_bound': prediction + 1.96 * rmse
                }
    
    return {
        'ticker': ticker,
        'prediction': None,
        'lower_bound': None,
        'upper_bound': None
    } 