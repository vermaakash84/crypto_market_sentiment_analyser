import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load champion model
with open('champion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize editable trades data
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame({
        'price': [50000, 51000],
        'size': [0.1, 0.2],
        'side': ['buy', 'sell'],
        'fee': [10, 12],
        'timestamp': ['2025-09-09 14:00', '2025-09-09 14:05']
    })

st.title("Crypto Market Sentiment Analyzer")

# Editable table
st.subheader("Editable Trades Table")
edited_df = st.data_editor(st.session_state.trades, num_rows="dynamic")
st.session_state.trades = edited_df.copy()

# Feature calculation (ensure correct feature order)
def calculate_features(trades_df):
    # Example of computed features
    features = {
        'mean_exec_price': [trades_df['price'].mean()],
        'total_usd': [trades_df['price'].sum()],
        'total_tokens': [trades_df['size'].sum()],
        'avg_fee': [trades_df['fee'].mean()],
        'total_pnl': [0],  # Placeholder
        'buy_ratio': [(trades_df['side'] == 'buy').mean()],
        'crossed_ratio': [0],  # Placeholder
        'exec_price_volatility': [trades_df['price'].std()],
        'fee_volatility': [trades_df['fee'].std()],
        'price_efficiency': [0],  # Placeholder
        'fee_efficiency': [0],    # Placeholder
        'risk_reward': [0],       # Placeholder
        'aggressiveness': [0],    # Placeholder
        'smart_money': [0],       # Placeholder
        'pnl_per_usd': [0],       # Placeholder
        'enhanced_fee_efficiency': [0],  # Placeholder
        'smart_fee_ratio': [0],          # Placeholder
        'risk_adjusted_efficiency': [0],# Placeholder
        'aggressive_efficiency': [0],    # Placeholder
        'volatility_ratio': [0],         # Placeholder
        'stable_efficiency': [0],        # Placeholder
        'top_features_interaction': [0],
        'buy_ratio': [(trades_df['side'] == 'buy').mean()]
    }
    return pd.DataFrame(features)

features_df = calculate_features(st.session_state.trades)

# Make prediction
if hasattr(model, "predict_proba"):
    pred_scores = model.predict_proba(features_df)
    pred_label = model.classes_[np.argmax(pred_scores)]
else:
    pred_label = model.predict(features_df)[0]

# Display prediction
st.subheader("Predicted Sentiment")
st.write(f"🧠 Predicted Sentiment: **{pred_label}**")

# Top 3 largest trades
st.subheader("Top 3 Largest Trades")
top_trades = st.session_state.trades.nlargest(3, 'size')
st.table(top_trades)

# Summary statistics
st.subheader("Summary Statistics")
st.write(st.session_state.trades.describe())

# Highlight extremes
st.subheader("Highlight Extremes")
max_price = st.session_state.trades['price'].max()
min_price = st.session_state.trades['price'].min()
st.write(f"💰 Maximum Price: {max_price}")
st.write(f"📉 Minimum Price: {min_price}")
