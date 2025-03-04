import streamlit as st
import sys
import os
import matplotlib.pyplot as plt

# Ensure proper module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetch import get_stock_price, get_stock_news, get_historical_data, calculate_sma, calculate_ema, calculate_rsi
from sentiment_analysis import analyze_sentiment
from ml_model import predict_stock_price, train_stock_model

# Set page configuration
st.set_page_config(page_title="Stock Sentiment & Prediction Dashboard", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("Customize your dashboard experience.")
    st.markdown("---")
    st.write("### Select Time Period")
    time_period = st.selectbox("Choose the time period for historical data", ["1mo", "3mo", "6mo", "1y", "2y"])
    st.markdown("---")
    st.write("### About")
    st.write("This dashboard provides real-time stock sentiment analysis and price predictions using advanced machine learning models.")

# UI Title
st.title("📊 Real-Time Stock Sentiment & Prediction Dashboard")

# Stock Symbol Input
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG    )", "AAPL")

if st.button("Analyze"):
    # Fetch stock data
    stock_price = get_stock_price(ticker)
    news = get_stock_news()  # Fetch news related to the entered stock
    historical_data = get_historical_data(ticker, period=time_period)  # Fetch historical price data

    # Ensure data is available
    if historical_data is not None:
        # Calculate SMA, EMA, RSI
        historical_data["SMA_50"] = historical_data["Close"].rolling(window=50).mean()
        historical_data["EMA_50"] = historical_data["Close"].ewm(span=50, adjust=False).mean()
        
        delta = historical_data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        RS = gain / loss
        historical_data["RSI"] = 100 - (100 / (1 + RS))

        # Display Stock Price
        st.write(f"## {ticker} Stock Price: **${stock_price:.2f}**")

        # News & Sentiment Section
        st.write("### 📰 Latest News & Sentiment Analysis")
        sentiments = [analyze_sentiment(article) for article in news]
        for i, article in enumerate(news):
            sentiment_score = sentiments[i]
            sentiment = (
                "🟢 Positive" if sentiment_score > 0 else 
                "🔴 Negative" if sentiment_score < 0 else 
                "⚪ Neutral"
            )
            st.markdown(f"- **{article}**  \n ➝ Sentiment: {sentiment}")

        # Technical Indicators Visualization
        st.write("### 📉 Technical Analysis: SMA, EMA & RSI")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(historical_data["Close"], label="Close Price", color="blue")
        ax.plot(historical_data["SMA_50"], label="SMA (50-day)", linestyle="dashed", color="orange")
        ax.plot(historical_data["EMA_50"], label="EMA (50-day)", linestyle="dashed", color="red")
        ax.legend()
        st.pyplot(fig)

        # RSI Plot
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
        ax_rsi.plot(historical_data["RSI"], label="RSI (14-day)", color="purple")
        ax_rsi.axhline(70, linestyle="dashed", color="red", label="Overbought (70)")
        ax_rsi.axhline(30, linestyle="dashed", color="green", label="Oversold (30)")
        ax_rsi.legend()
        st.pyplot(fig_rsi)

        # Fetch historical data
    historical_data = get_historical_data(ticker, period=time_period)
    
    if historical_data is not None:
        sma = calculate_sma(historical_data)
        ema = calculate_ema(historical_data)
        rsi = calculate_rsi(historical_data)

        train_stock_model(ticker)
        
        st.write("### 📈 Stock Price Prediction")
        prediction = predict_stock_price(stock_price, stock_price * 1.02, stock_price * 0.98, 1000000, sma, ema, rsi)
        st.write(f"#### Predicted Next-Day Close: ${prediction:.2f}")
    else:
        st.error("Could not fetch historical data for calculations.")