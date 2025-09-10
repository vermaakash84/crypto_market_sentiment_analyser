import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your champion model
@st.cache_resource
def load_model():
    model = joblib.load('production_model_final.pkl')
    scaler = joblib.load('production_scaler.pkl')
    le = joblib.load('production_label_encoder.pkl')
    return model, scaler, le

def main():
    st.title("📈 Market Sentiment Analyzer")
    st.write("Predicting Fear & Greed in Market Behavior")

    # Load model
    model, scaler, le = load_model()

    # Input features
    st.sidebar.header("Input Market Features")

    # Create input fields for your top features
    total_usd = st.sidebar.slider("Total USD Volume", -3.0, 3.0, 0.0)
    price_efficiency = st.sidebar.slider("Price Efficiency", -3.0, 3.0, 0.0)
    smart_money = st.sidebar.slider("Smart Money Indicator", -3.0, 3.0, 0.0)
    buy_ratio = st.sidebar.slider("Buy Ratio", -3.0, 3.0, 0.0)

    # Create feature vector
    input_features = np.array([[total_usd, price_efficiency, smart_money, buy_ratio]])

    if st.sidebar.button("Analyze Sentiment"):
        # Scale features
        scaled_features = scaler.transform(input_features)

        # Predict
        prediction = model.predict(scaled_features)
        sentiment = le.inverse_transform(prediction)[0]

        # Get confidence
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
