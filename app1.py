import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load saved models and encoders
model = joblib.load('production_model_final.pkl')
optimized_weights = joblib.load('optimal_weights_final.pkl')

# Load label encoder (you need to save and load this similarly if used)
le = LabelEncoder()
# Assuming classes: ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
le.classes_ = np.array(['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])

st.title('Crypto Market Sentiment & Trader Behavior Analyzer')

st.write("""
    Enter trading behavior metrics to predict market sentiment:
""")

# User input fields
mean_exec_price = st.number_input('Mean Execution Price', value=1000.0, step=1.0)
total_usd = st.number_input('Total USD Volume', value=100000.0, step=1000.0)
total_tokens = st.number_input('Total Tokens', value=100.0, step=1.0)
avg_fee = st.number_input('Average Fee', value=0.01, step=0.001)
total_pnl = st.number_input('Total PnL', value=500.0, step=1.0)
buy_ratio = st.number_input('Buy Ratio (0 to 1)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
crossed_ratio = st.number_input('Crossed Ratio (0 to 1)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
exec_price_volatility = st.number_input('Execution Price Volatility', value=10.0, step=0.1)
fee_volatility = st.number_input('Fee Volatility', value=0.01, step=0.001)
price_efficiency = st.number_input('Price Efficiency', value=1.0, step=0.01)
fee_efficiency = st.number_input('Fee Efficiency', value=0.01, step=0.001)
risk_reward = st.number_input('Risk Reward Ratio', value=1.0, step=0.01)
aggressiveness = st.number_input('Aggressiveness', value=0.5, step=0.01)
smart_money = st.number_input('Smart Money Indicator', value=0.5, step=0.01)

if st.button('Predict Market Sentiment'):
    input_features = np.array([[
        mean_exec_price, total_usd, total_tokens, avg_fee, total_pnl,
        buy_ratio, crossed_ratio, exec_price_volatility, fee_volatility,
        price_efficiency, fee_efficiency, risk_reward, aggressiveness, smart_money
    ]])

    prediction_encoded = model.predict(input_features)
    prediction = le.inverse_transform(prediction_encoded)[0]

    st.success(f'Predicted Market Sentiment: **{prediction}**')

    st.write('Optimized Weights Used for Training:')
    st.json(optimized_weights)
