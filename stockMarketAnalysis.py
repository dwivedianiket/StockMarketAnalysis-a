import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np

API_KEY = 'ISFQL7OJCVVEX6GC'  # Alpha Vantage API Key

def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (1min)']).transpose()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

def preprocess_data(df):
    df['Pct_Change'] = df['Close'].pct_change()
    for lag in [1, 5, 10, 15, 30]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    df['Price_Change'] = df['Close'].diff()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # Added SMA_20
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / df['Close'].diff().apply(lambda x: -min(x, 0)).rolling(window=14).mean())))
    df['Bollinger_High'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Low'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    df = df.dropna()
    scaler = StandardScaler()
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'Volatility', 'Pct_Change', 'RSI']
    df.loc[:, features] = scaler.fit_transform(df[features])
    return df

def train_model(df):
    X = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'Volatility', 'Pct_Change', 'RSI']].values
    y = df['Close'].values
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')
    predictions = best_model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error on full dataset: {mse}')
    return best_model

def predict_real_time(df, model):
    X_real_time = df[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'Volatility', 'Pct_Change', 'RSI']].values[-1:]
    prediction = model.predict(X_real_time)
    return prediction

# Streamlit UI
st.title('Real-Time Stock Market Analysis')

symbol = st.text_input('Enter Stock Symbol', 'AAPL')

if symbol:
    try:
        latest_data = fetch_stock_data(symbol)
        processed_data = preprocess_data(latest_data)
        model = train_model(processed_data)
        prediction = predict_real_time(processed_data, model)
        
        st.write(f'Real-Time Prediction for {symbol}: {prediction[0]}')
        
        # Plot Close Price
        fig_price = px.line(processed_data, x=processed_data.index, y='Close', title=f'Price of {symbol}')
        st.plotly_chart(fig_price)

        # Plot SMA
        fig_sma_10 = px.line(processed_data, x=processed_data.index, y='SMA_10', title=f'SMA 10 for {symbol}')
        st.plotly_chart(fig_sma_10)

        fig_sma_50 = px.line(processed_data, x=processed_data.index, y='SMA_50', title=f'SMA 50 for {symbol}')
        st.plotly_chart(fig_sma_50)

        # Plot EMA
        fig_ema_10 = px.line(processed_data, x=processed_data.index, y='EMA_10', title=f'EMA 10 for {symbol}')
        st.plotly_chart(fig_ema_10)

        fig_ema_50 = px.line(processed_data, x=processed_data.index, y='EMA_50', title=f'EMA 50 for {symbol}')
        st.plotly_chart(fig_ema_50)

        # Plot Bollinger Bands
        fig_bollinger = go.Figure()
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], mode='lines', name='Close'))
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(dash='dash')))
        fig_bollinger.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(dash='dash')))
        fig_bollinger.update_layout(title=f'Bollinger Bands for {symbol}')
        st.plotly_chart(fig_bollinger)

        # Plot RSI
        fig_rsi = px.line(processed_data, x=processed_data.index, y='RSI', title=f'RSI for {symbol}')
        st.plotly_chart(fig_rsi)

        # Plot Volume
        fig_volume = px.bar(processed_data, x=processed_data.index, y='Volume', title=f'Trading Volume for {symbol}')
        st.plotly_chart(fig_volume)

        # Plot feature correlations
        corr_matrix = processed_data[['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'Volatility', 'Pct_Change', 'RSI']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title=f'Feature Correlation Matrix for {symbol}')
        st.plotly_chart(fig_corr)
        
    except Exception as e:
        st.error(f'Error fetching or processing data for {symbol}: {e}')
