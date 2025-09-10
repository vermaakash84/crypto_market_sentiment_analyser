import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Function to load model, scaler, and label encoder ---
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("production_model_final.pkl")
        scaler = joblib.load("production_scaler.pkl")
        le = joblib.load("production_label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e.filename}. Make sure all .pkl files are uploaded!")
        return None, None, None

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Market Sentiment Analyzer", layout="centered")
    st.title("📈 Market Sentiment Analyzer")
    st.write("Predicting Fear & Greed in Market Behavior")

    # Load model, scaler, label encoder
    model, scaler, le = load_model_files()
    if not model or not scaler or not le:
        st.stop()  # Stop app if model files are missing

    # Sidebar: Input features
    st.sidebar.header("Input Market Features")
    total_usd = st.sidebar.slider("Total USD Volume", -3.0, 3.0, 0.0)
    price_efficiency = st.sidebar.slider("Price Efficiency", -3.0, 3.0, 0.0)
    smart_money = st.sidebar.slider("Smart Money Indicator", -3.0, 3.0, 0.0)
    buy_ratio = st.sidebar.slider("Buy Ratio", -3.0, 3.0, 0.0)

    # Feature vector
    input_features = np.array([[total_usd, price_efficiency, smart_money, buy_ratio]])

    if st.sidebar.button("Analyze Sentiment"):
        # Scale features
        scaled_features = scaler.transform(input_features)

        # Predict sentiment
        prediction = model.predict(scaled_features)
        sentiment = le.inverse_transform(prediction)[0]

        # Prediction confidence
        probabilities = model.predict_proba(scaled_features)
        confidence = np.max(probabilities)

        # Display results
        st.success(f"🎯 Predicted Sentiment: **{sentiment}**")
        st.info(f"📊 Confidence: {confidence:.2%}")

        # Visual indicators
        if "Fear" in sentiment:
            st.error("⚠️ Market caution advised")
        elif "Greed" in sentiment:
            st.warning("💰 Profit-taking opportunity")
        else:
            st.success("✅ Market neutral")

if __name__ == "__main__":
    main()
