import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# --- Load the trained model ---
with open("champion_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Initialize session state for trades ---
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame({
        "price": [100, 102, 101, 103, 105],
        "size": [1.5, 2.0, 1.0, 2.5, 1.2],
        "side": ["buy","buy","buy","buy","buy"]
    })

st.title("Crypto Market Sentiment Analyser")

st.subheader("Editable Trades Table")

# --- Editable trades table using input widgets ---
edited_trades = []
for i, row in st.session_state.trades.iterrows():
    col1, col2, col3 = st.columns(3)
    price = col1.number_input(f"Price {i}", value=float(row["price"]), key=f"price_{i}")
    size = col2.number_input(f"Size {i}", value=float(row["size"]), key=f"size_{i}")
    side = col3.selectbox(f"Side {i}", ["buy", "sell"], index=0 if row["side"]=="buy" else 1, key=f"side_{i}")
    edited_trades.append({"price": price, "size": size, "side": side})

st.session_state.trades = pd.DataFrame(edited_trades)

# --- Feature calculation function ---
def compute_features(df):
    df = df.copy()
    # Example features - match your model's expected feature names
    df["total_usd"] = df["price"] * df["size"]
    df["buy_ratio"] = df["side"].apply(lambda x: 1 if x=="buy" else 0)
    df["crossed_ratio"] = df["buy_ratio"].rolling(3, min_periods=1).mean()
    df["avg_fee"] = df["size"] * 0.001  # Example fee calculation
    df["total_pnl"] = df["price"].diff().fillna(0) * df["size"]
    df["price_efficiency"] = df["price"].pct_change().fillna(0)
    df["fee_efficiency"] = df["avg_fee"] / (df["total_usd"] + 1e-6)
    # Add placeholder for remaining features expected by your model
    # Fill missing features with zeros
    expected_features = model.get_booster().feature_names
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = 0.0
    return df[expected_features]

# --- Compute features ---
features_df = compute_features(st.session_state.trades)

# --- Predict using model ---
predictions = model.predict(features_df)
st.session_state.trades["prediction"] = predictions

# --- Display results ---
st.subheader("Trades with Predictions")
st.dataframe(st.session_state.trades)

st.subheader("Top 3 Trades by Prediction")
st.dataframe(st.session_state.trades.nlargest(3, "prediction"))

st.subheader("Summary Statistics")
st.write(st.session_state.trades.describe())

st.subheader("Highlight Extremes")
st.write("High prediction trades:")
st.dataframe(st.session_state.trades[st.session_state.trades["prediction"] > st.session_state.trades["prediction"].mean()])

st.write("Low prediction trades:")
st.dataframe(st.session_state.trades[st.session_state.trades["prediction"] < st.session_state.trades["prediction"].mean()])
