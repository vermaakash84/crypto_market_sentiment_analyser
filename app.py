import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Pre-trained Model and Label Encoder ---
model = joblib.load('production_model_final.pkl')
le = joblib.load('production_label_encoder.pkl')

# --- Feature Engineering Function ---
def feature_engineering(df):
    df['exec_price_volatility'] = 0.01  # Placeholder constant (should ideally come from historical data)
    df['fee_volatility'] = 0.01        # Placeholder constant

    df['price_efficiency'] = df['mean_exec_price'] / (df['total_usd'] / (df['total_tokens'] + 1e-5) + 1e-5)
    df['fee_efficiency'] = df['avg_fee'] / (df['total_usd'] + 1e-5)
    df['risk_reward'] = (df['total_pnl'] / (df['total_usd'] + 1e-5)).abs()
    df['aggressiveness'] = df['crossed_ratio'] * df['buy_ratio']
    df['smart_money'] = (df['buy_ratio'] * df['total_pnl']).abs()
    df['pnl_per_usd'] = df['total_pnl'] / (df['total_usd'] + 1e-5)
    df['enhanced_fee_efficiency'] = df['fee_efficiency'] * np.log1p(df['total_usd'])
    df['smart_fee_ratio'] = df['smart_money'] / (df['avg_fee'] + 1e-5)
    df['risk_adjusted_efficiency'] = df['fee_efficiency'] * (1 - df['risk_reward'])
    df['aggressive_efficiency'] = df['fee_efficiency'] * df['aggressiveness']
    df['volatility_ratio'] = df['exec_price_volatility'] / (df['fee_volatility'] + 1e-5)
    df['stable_efficiency'] = df['fee_efficiency'] / (df['exec_price_volatility'] + 1e-5)
    df['top_features_interaction'] = df['total_usd'] * df['fee_efficiency'] * df['smart_money']

    return df

# --- Streamlit UI ---
st.title("🚀 Crypto Market Sentiment Predictor")

st.subheader("📊 Enter Trading Feature Values Manually")

# Manual user inputs
mean_exec_price = st.number_input('Mean Execution Price', min_value=0.0, step=0.01)
total_usd = st.number_input('Total USD', min_value=0.0, step=0.01)
total_tokens = st.number_input('Total Tokens', min_value=0.0, step=0.01)
avg_fee = st.number_input('Average Fee', min_value=0.0, step=0.01)
total_pnl = st.number_input('Total PnL', step=0.01)
buy_ratio = st.number_input('Buy Ratio (e.g., 0.5)', min_value=0.0, max_value=1.0, step=0.01)
crossed_ratio = st.number_input('Crossed Ratio (e.g., 0.5)', min_value=0.0, max_value=1.0, step=0.01)

if st.button('🔍 Predict Sentiment'):
    # Prepare input data
    input_data = pd.DataFrame([{
        'mean_exec_price': mean_exec_price,
        'total_usd': total_usd,
        'total_tokens': total_tokens,
        'avg_fee': avg_fee,
        'total_pnl': total_pnl,
        'buy_ratio': buy_ratio,
        'crossed_ratio': crossed_ratio
    }])

    # Apply full feature engineering
    input_data = feature_engineering(input_data)

    # Predict classification
    prediction_encoded = model.predict(input_data)
    prediction = le.inverse_transform(prediction_encoded)[0]

    # Display result
    st.success(f"🎯 Predicted Market Sentiment Classification: **{prediction}**")
