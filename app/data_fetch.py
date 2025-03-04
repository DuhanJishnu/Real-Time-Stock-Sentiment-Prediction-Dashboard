import yfinance as yf
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return price

def get_stock_news():
    url = f"https://newsapi.org/v2/everything?q=stocks&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    return [article["title"] for article in response["articles"][:5]]

def get_historical_data(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist[["Close"]].reset_index()
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def calculate_sma(data, window=14):
    return data["Close"].rolling(window=window).mean().iloc[-1]

def calculate_ema(data, window=14):
    return data["Close"].ewm(span=window, adjust=False).mean().iloc[-1]

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]
