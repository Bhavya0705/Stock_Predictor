import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
import yfinance as yf
from datetime import datetime, timedelta


def load_data(stock_symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3650)).strftime('%Y-%m-%d')  
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['ATR'] = compute_atr(df)
    df.dropna(inplace=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal


def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()


def train_lstm(df):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i - 60:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=1)
    return model, scaler


def train_random_forest(df):
    features = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'ATR']
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    df.dropna(inplace=True)
    X, y = df[features], df['Target']
    model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    return model, accuracy


def predict_next_day(df, lstm_model, lstm_scaler, rf_model):
    last_60_days = lstm_scaler.transform(df[['Close']].tail(60))
    X_input = np.array([last_60_days])
    predicted_price = lstm_scaler.inverse_transform(lstm_model.predict(X_input))[0][0]
    
    latest_features = df[['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'ATR']].iloc[-1:].dropna().values
    if latest_features.shape[0] == 0:
        return predicted_price, "No Decision (Missing Data)"
    
    predicted_signal = rf_model.predict(latest_features)[0]
    signal_text = 'Buy' if predicted_signal == 1 else 'Sell' if predicted_signal == -1 else 'Hold'
    
    return predicted_price, signal_text


def main():
    st.title("📈 Stock Price Prediction & Trading Signals")
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    df = load_data(stock_symbol)
    lstm_model, lstm_scaler = train_lstm(df)
    rf_model, rf_accuracy = train_random_forest(df)
    predicted_price, trading_decision = predict_next_day(df, lstm_model, lstm_scaler, rf_model)
    
    
    st.subheader("📊 Next Day Prediction")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Predicted Price", value=f"${predicted_price:.2f}")
        with col2:
            st.metric(label="Trading Decision", value=f"{trading_decision}")
        with col3:
            st.metric(label="Accuracy", value=f"{rf_accuracy * 100:.2f}%")
    
    
    st.subheader("📊 Stock Data")
    df_display = df.copy()
    df_display.index = df_display.index.strftime('%Y-%m-%d')
    df_display = df_display.round(2)
    st.dataframe(df_display)
    
   
    st.subheader("📈 Actual vs Predicted Prices")
    actual_prices = df['Close'].values[60:]
    predicted_prices = [lstm_scaler.inverse_transform(
        lstm_model.predict(np.array([lstm_scaler.transform(df[['Close']].values[i-60:i])]))
    )[0][0] for i in range(60, len(df))]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[60:], actual_prices, label="Actual Prices", color='blue')
    ax.plot(df.index[60:], predicted_prices, label="Predicted Prices", color='red', linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title("Actual vs Predicted Stock Prices")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
