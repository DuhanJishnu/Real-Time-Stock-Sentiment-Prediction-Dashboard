import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def calculate_indicators(df):
    """Adds technical indicators like SMA, EMA, and RSI to the dataframe."""
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()  # Handle missing values
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.fillna(method="bfill", inplace=True)  # Backfill missing values
    return df

def train_stock_model(ticker):
    """Fetches stock data, calculates indicators, and trains a RandomForest model."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")

        if df.empty:
            print(f"Error: No data found for ticker '{ticker}'.")
            return

        df = calculate_indicators(df)
        df["Target"] = df["Close"].shift(-1)  # Predict next day's close price
        df.dropna(inplace=True)  # Drop rows with NaN values

        # Features: Open, High, Low, Volume, SMA, EMA, RSI
        X = df[["Open", "High", "Low", "Volume", "SMA_10", "EMA_10", "RSI"]]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)  # Ensure model directory exists
        joblib.dump(model, "models/trained_model.pkl")
        print("✅ Model trained successfully and saved.")

    except Exception as e:
        print(f"⚠️ Error during training: {e}")


def predict_stock_price(open_price, high, low, volume, sma, ema, rsi):
    """Loads the trained model and predicts the next closing price."""
    try:
        model_path = "models/trained_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")

        model = joblib.load(model_path)
        return model.predict([[open_price, high, low, volume, sma, ema, rsi]])[0]
    
    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")
        return None
