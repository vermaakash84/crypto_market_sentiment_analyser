import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('champion_xgb_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, label_encoder

model, label_encoder = load_model()

st.title("Crypto Market Sentiment Analyzer 📈")

st.markdown("""
You can input real-time trading data below to predict the market sentiment (Fear, Greed, Neutral, etc.). 
This helps analyze the relationship between trader behavior and market sentiment for smarter strategies.
""")

# Define input fields for the features
mean_exec_price = st.number_input("Mean Execution Price", min_value=0.0, value=1000.0)
total_usd = st.number_input("Total USD", min_value=0.0, value=50000.0)
total_tokens = st.number_input("Total Tokens", min_value=0.0, value=1500.0)
avg_fee = st.number_input("Average Fee", min_value=0.0, value=50.0)
total_pnl = st.number_input("Total PnL", value=1000.0)
buy_ratio = st.number_input("Buy Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
crossed_ratio = st.number_input("Crossed Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
exec_price_volatility = st.number_input("Execution Price Volatility", value=5.0)
fee_volatility = st.number_input("Fee Volatility", value=1.0)
price_efficiency = st.number_input("Price Efficiency", value=1.2)
fee_efficiency = st.number_input("Fee Efficiency", value=0.1)
risk_reward = st.number_input("Risk-Reward Ratio", value=0.05)
aggressiveness = st.number_input("Aggressiveness", value=0.3)
smart_money = st.number_input("Smart Money Indicator", value=0.7)

# Collect input into a DataFrame
input_features = pd.DataFrame({
    'mean_exec_price': [mean_exec_price],
    'total_usd': [total_usd],
    'total_tokens': [total_tokens],
    'avg_fee': [avg_fee],
    'total_pnl': [total_pnl],
    'buy_ratio': [buy_ratio],
    'crossed_ratio': [crossed_ratio],
    'exec_price_volatility': [exec_price_volatility],
    'fee_volatility': [fee_volatility],
    'price_efficiency': [price_efficiency],
    'fee_efficiency': [fee_efficiency],
    'risk_reward': [risk_reward],
    'aggressiveness': [aggressiveness],
    'smart_money': [smart_money]
})

if st.button("Predict Market Sentiment"):
    prediction_encoded = model.predict(input_features)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f"Predicted Market Sentiment: **{prediction}**")

    st.info("Use this result to understand how current trading behavior aligns or diverges from overall market sentiment.")

st.markdown("---")
st.write("🔍 Developed to analyze hidden trends and signals that could influence smarter trading strategies.")
