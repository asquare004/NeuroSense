import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

st.title("📈 NeuroSense - Sense the Market, Predict the Future")

# User-defined stock selection
selected_stock = st.text_input('Enter Stock Ticker:', 'AAPL').upper()

def fetch_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="5y")
        if data.empty:
            st.error("Invalid stock symbol or no data available. Please try again.")
            st.stop()
        return data[["Close", "Volume"]]
    except:
        st.error("Error fetching stock data. Check the symbol and try again.")
        st.stop()

def preprocess_data(df):
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_50"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # RSI
    delta = df["Close"].diff(1)
    gain = (delta.where(delta>0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["Middle Band"] = df["Close"].rolling(window=20).mean()
    df["Upper Band"] = df["Middle Band"] + (df["Close"].rolling(window=20).std() * 2)
    df["Lower Band"] = df["Middle Band"] - (df["Close"].rolling(window=20).std() * 2)

    # MACD
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    # OBV
    df["OBV"] = (df["Close"].diff() > 0).astype(int) - (df["Close"].diff() < 0).astype(int)
    df["OBV"] = df["OBV"].cumsum()

    # Stochastic Oscillator
    low_14 = df["Close"].rolling(14).min()
    high_14 = df["Close"].rolling(14).max()
    df["Stochastic %K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["Close"]])
    return scaled_data, scaler

def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True, input_shape=input_shape)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_prices(model, scaler, scaled_data, days=10):
    predictions = []
    input_sequence = scaled_data[-60:]
    for _ in range(days):
        input_sequence = input_sequence.reshape(1, -1, 1)
        predicted_price = model.predict(input_sequence)[0][0]
        predictions.append(predicted_price)
        input_sequence = np.append(input_sequence[0][1:], [[predicted_price]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[:, 0]

data = fetch_data(selected_stock)
st.write(f'Latest Price: ${data["Close"].iloc[-1]:.2f}')

if st.button("Train and Predict"):
    st.write("Training Model, Please Wait...")
    
    data = preprocess_data(data)
    scaled_data, scaler = normalize_data(data)
    X, y = prepare_data(scaled_data)
    
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    if selected_stock in st.session_state:
        model = st.session_state[selected_stock]  # Retrieve cached model
    else:
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, batch_size=32, epochs=10)
        st.session_state[selected_stock] = model  # Cache model in session
    
    predictions = predict_prices(model, scaler, scaled_data, days=10)
    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime("%Y-%m-%d").tolist()
    prediction_df = pd.DataFrame({"Date": days, "Predicted Price": predictions})
    
    st.write("Predicted Prices:")
    st.table(prediction_df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prediction_df["Date"], y=prediction_df["Predicted Price"], mode="lines+markers", name="Predicted Prices"))
    fig.update_layout(title=f"10-Day Prediction for {selected_stock}", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark")
    st.plotly_chart(fig)
    
if st.checkbox("Show historical data"):
    st.line_chart(data["Close"])
