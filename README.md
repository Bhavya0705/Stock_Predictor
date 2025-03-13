# Stock_Predictor
A machine learning-based stock price predictor that forecasts the next day's price and generates a Buy/Sell/Hold decision using LSTM  networks and a Random Forest classifier.

Overview
This project is a machine learning-based stock prediction system that forecasts stock prices and generates trading signals using LSTM (Long Short-Term Memory) networks and Random Forest classifiers. It fetches real-time stock data from Yahoo Finance, processes multiple technical indicators, and provides insights through a Streamlit-powered web app.

Features
✔ Stock Data Retrieval – Fetches historical stock data (up to 10 years) from Yahoo Finance.
✔ Technical Indicators – Uses key indicators like RSI, MACD, SMA, EMA, ATR, Bollinger Bands, Momentum, and Volume Analysis.
✔ LSTM for Stock Price Prediction – Predicts the next day's stock price based on past trends.
✔ Random Forest for Trading Signals – Classifies stocks into Buy, Sell, or Hold decisions.
✔ Streamlit Web Dashboard – Interactive interface for real-time stock predictions and analysis.
✔ Graphical Insights – Visualizes actual vs. predicted prices using Matplotlib.

Tech Stack
Python (NumPy, Pandas, Matplotlib, yFinance, Sklearn, TensorFlow/Keras)
Machine Learning
LSTM (for price prediction)
Random Forest (for trading signals)
Data Preprocessing & Feature Engineering
Scaling & normalization using MinMaxScaler
Feature extraction from technical indicators
Streamlit (for the interactive web dashboard)


How It Works
1️⃣ Fetch Data: The app downloads stock data from Yahoo Finance.
2️⃣ Feature Engineering: Calculates RSI, MACD, Moving Averages, ATR, and more.
3️⃣ Train LSTM Model: Predicts the next day's closing price.
4️⃣ Train Random Forest Model: Classifies the stock movement into Buy/Sell/Hold.
5️⃣ Make Predictions: The app displays predicted prices, trading signals, and accuracy metrics.
6️⃣ Visualize Results: The system plots historical vs. predicted prices for analysis.



