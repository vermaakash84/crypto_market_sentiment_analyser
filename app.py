import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# --- Load Pre-trained Model and Label Encoder ---
model = joblib.load('champion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# --- Helper Function: Preprocess Data ---
def preprocess_data(df):
    df = df.copy()

    # Trim column names of any leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Debugging: Show cleaned column names
    st.write("🔍 Cleaned Column Names:")
    st.write(df.columns.tolist())

    # Basic feature engineering
    df['fee_efficiency'] = df['Fee'] / (df['Size USD'] + 1e-5)
    df['total_usd'] = df['Size USD']
    df['total_pnl'] = df['Closed PnL']
    df['avg_fee'] = df['Fee']
    
    # Aggregate by 'Timestamp IST'
    daily = df.groupby('Timestamp IST').agg({
        'total_usd': 'sum',
        'total_pnl': 'sum',
        'avg_fee': 'mean',
        'fee_efficiency': 'mean'
    }).reset_index()
    
    return daily[['total_usd', 'total_pnl', 'avg_fee', 'fee_efficiency']]

# --- Helper Function: Convert DataFrame to CSV for Download ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.title("🚀 Crypto Market Sentiment Predictor")

uploaded_file = st.file_uploader("📂 Upload your raw trading data CSV", type="csv")

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.subheader("✅ Raw Data Sample")
    st.dataframe(raw_data.head())

    # Debugging uploaded columns
    st.write("📋 Columns in uploaded file:")
    st.write(raw_data.columns.tolist())

    # Feature Engineering
    try:
        processed_features = preprocess_data(raw_data)
        st.subheader("✅ Processed Features Sample")
        st.dataframe(processed_features.head())

        # Predict sentiment
        preds_encoded = model.predict(processed_features)
        preds = label_encoder.inverse_transform(preds_encoded)

        processed_features['predicted_sentiment'] = preds
        st.subheader("🎯 Predicted Market Sentiment")
        st.dataframe(processed_features[['total_usd', 'total_pnl', 'avg_fee', 'fee_efficiency', 'predicted_sentiment']])

        # Download Button for results
        csv = convert_df(processed_features)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='predicted_sentiments.csv',
            mime='text/csv',
        )

    except KeyError as e:
        st.error(f"⚠️ Missing expected column in uploaded file: {e}")
        st.info("❗ Please check that columns 'Fee', 'Size USD', 'Closed PnL', and 'Timestamp IST' are present.")

else:
    st.info("⚠️ Please upload a CSV file to start predicting market sentiment.")
