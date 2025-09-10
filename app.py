import streamlit as st
import pickle
import numpy as np

# Load model and label encoder
with open("production_model_final.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("Crypto Market Sentiment Analyzer")

# Example: user inputs or data you already have
mean_exec_price = st.number_input("Mean Execution Price")
total_usd = st.number_input("Total USD")
total_tokens = st.number_input("Total Tokens")
avg_fee = st.number_input("Average Fee")
total_pnl = st.number_input("Total PnL")
buy_ratio = st.number_input("Buy Ratio")
crossed_ratio = st.number_input("Crossed Ratio")
exec_price_volatility = st.number_input("Execution Price Volatility")
fee_volatility = st.number_input("Fee Volatility")
price_efficiency = st.number_input("Price Efficiency")
fee_efficiency = st.number_input("Fee Efficiency")
risk_reward = st.number_input("Risk Reward")
aggressiveness = st.number_input("Aggressiveness")
smart_money = st.number_input("Smart Money")
pnl_per_usd = st.number_input("PnL per USD")
enhanced_fee_efficiency = st.number_input("Enhanced Fee Efficiency")
smart_fee_ratio = st.number_input("Smart Fee Ratio")
risk_adjusted_efficiency = st.number_input("Risk Adjusted Efficiency")
aggressive_efficiency = st.number_input("Aggressive Efficiency")
volatility_ratio = st.number_input("Volatility Ratio")
stable_efficiency = st.number_input("Stable Efficiency")
top_features_interaction = st.number_input("Top Features Interaction")

# Prepare input features in correct order (22 features)
input_features = [[
    mean_exec_price,
    total_usd,
    total_tokens,
    avg_fee,
    total_pnl,
    buy_ratio,
    crossed_ratio,
    exec_price_volatility,
    fee_volatility,
    price_efficiency,
    fee_efficiency,
    risk_reward,
    aggressiveness,
    smart_money,
    pnl_per_usd,
    enhanced_fee_efficiency,
    smart_fee_ratio,
    risk_adjusted_efficiency,
    aggressive_efficiency,
    volatility_ratio,
    stable_efficiency,
    top_features_interaction
]]

# Make prediction
prediction_encoded = model.predict(input_features)
prediction = le.inverse_transform(prediction_encoded)[0]

# Display result
st.success(f"Predicted Market Sentiment: **{prediction}**")
