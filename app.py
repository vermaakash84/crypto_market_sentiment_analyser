import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier  # or XGBRegressor if your model is regression

st.set_page_config(page_title="Crypto Market Sentiment Analyzer", layout="wide")

# --- Load model ---
with open("champion_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Initialize session state for trades ---
if "trades" not in st.session_state:
    st.session_state.trades = pd.DataFrame({
        "price": [100, 101, 102, 99, 98],
        "size": [5, 10, 7, 8, 6],
        "side": ["buy", "buy", "buy", "buy", "buy"]
    })

# --- Editable trades table ---
st.subheader("Editable Trades Table")
edited_df = st.dataframe(st.session_state.trades, height=300)
st.session_state.trades = edited_df.copy()

# --- Feature calculation ---
def calculate_features(df):
    # Example feature calculations: adjust these to match your training features
    features = pd.DataFrame()
    features["price_change"] = df["price"].pct_change().fillna(0)
    features["size_change"] = df["size"].pct_change().fillna(0)
    features["buy_flag"] = (df["side"] == "buy").astype(int)
    features["sell_flag"] = (df["side"] == "sell").astype(int)
    features["rolling_mean_price"] = df["price"].rolling(3, min_periods=1).mean()
    features["rolling_std_price"] = df["price"].rolling(3, min_periods=1).std().fillna(0)
    features["buy_sell_ratio"] = features["buy_flag"].cumsum() / (features["sell_flag"].cumsum() + 1)
    # Add other features exactly matching your model's trained features
    # Example: features["top_features_interaction"] = features["price_change"] * features["size_change"]
    return features.fillna(0)

features_df = calculate_features(st.session_state.trades)

# --- Make predictions ---
pred_scores = model.predict_proba(features_df) if hasattr(model, "predict_proba") else model.predict(features_df)

# --- Map predictions to sentiment labels ---
# Adjust mapping according to your model output
def map_sentiment(pred):
    if isinstance(pred, np.ndarray) and pred.ndim > 1:  # multi-class
        # Assume pred[:,1] is probability of "Greed" for demo
        score = pred[:,1]
    else:
        score = pred
    if score <= 0.25:
        return "Extreme Fear"
    elif score <= 0.5:
        return "Fear"
    elif score <= 0.75:
        return "Greed"
    else:
        return "Extreme Greed"

sentiments = [map_sentiment(p) for p in pred_scores]

st.session_state.trades["prediction_score"] = pred_scores if pred_scores.ndim == 1 else pred_scores[:,1]
st.session_state.trades["sentiment"] = sentiments

# --- Display predictions with highlighting ---
st.subheader("Predictions with Sentiment")
def highlight_extremes(row):
    if row["sentiment"] in ["Extreme Fear", "Extreme Greed"]:
        return ["background-color: yellow"]*len(row)
    return [""]*len(row)

st.dataframe(st.session_state.trades.style.apply(highlight_extremes, axis=1), height=400)

# --- Top 3 trades by predicted score ---
st.subheader("Top 3 Trades")
top_trades = st.session_state.trades.nlargest(3, "prediction_score")
st.dataframe(top_trades)

# --- Summary statistics ---
st.subheader("Summary Statistics")
st.write(st.session_state.trades.describe())
