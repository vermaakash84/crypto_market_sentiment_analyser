import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Pre-trained Model and Label Encoder ---
model = joblib.load('production_model_final.pkl')
le = joblib.load('production_label_encoder.pkl')

# --- Streamlit UI ---
st.title("🚀 Crypto Market Sentiment Classification")

st.subheader("📊 Enter Trading Features Manually")

# Manual user inputs
mean_exec_price = st.number_input('Mean Execution Price', min_value=0.0, step=0.01)
total_usd = st.number_input('Total USD', min_value=0.0, step=0.01)
total_tokens = st.number_input('Total Tokens', min_value=0.0, step=0.01)
avg_fee = st.number_input('Average Fee', min_value=0.0, step=0.01)
total_pnl = st.number_input('Total PnL', step=0.01)
buy_ratio = st.number_input('Buy Ratio (e.g., 0.5)', min_value=0.0, max_value=1.0, step=0.01)
crossed_ratio = st.number_input('Crossed Ratio (e.g., 0.5)', min_value=0.0, max_value=1.0, step=0.01)

if st.button('🔍 Predict Sentiment'):
    # Prepare input DataFrame for prediction
    input_data = pd.DataFrame([{
        'mean_exec_price': mean_exec_price,
        'total_usd': total_usd,
        'total_tokens': total_tokens,
        'avg_fee': avg_fee,
        'total_pnl': total_pnl,
        'buy_ratio': buy_ratio,
        'crossed_ratio': crossed_ratio
    }])

    # Predict sentiment
    prediction_encoded = model.predict(input_data)
    prediction = le.inverse_transform(prediction_encoded)[0]

    # Display final classification result
    st.success(f"🎯 Predicted Market Sentiment Classification: **{prediction}**")
