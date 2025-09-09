import streamlit as st
import os
import pandas as pd
import joblib
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# File paths
# -----------------------------
FEAR_GREED_FILE = "fear_greed_index.csv"
HISTORICAL_DATA_FILE = "historical_data.csv"
MODEL_FILE = "champion_model.pkl"

st.set_page_config(page_title="Crypto Market Sentiment Analyzer", layout="wide")
st.title("📈 Crypto Market Sentiment Analyzer")

# -----------------------------
# Load fear_greed_index.csv
# -----------------------------
if os.path.exists(FEAR_GREED_FILE):
    df_fgi = pd.read_csv(FEAR_GREED_FILE)
else:
    st.error(f"{FEAR_GREED_FILE} not found! Please upload it to the repo.")
    st.stop()

# -----------------------------
# Download historical_data.csv from Google Drive if not present
# -----------------------------
if not os.path.exists(HISTORICAL_DATA_FILE):
    gdrive_url = "https://drive.google.com/uc?id=1BWlLp26bUZ8prT8L-e6BXhNcBRwCTpgC"
    st.info("Downloading historical_data.csv from Google Drive...")
    gdown.download(gdrive_url, HISTORICAL_DATA_FILE, quiet=False)

# Load historical data
try:
    df_hist = pd.read_csv(HISTORICAL_DATA_FILE)
except pd.errors.ParserError:
    st.error("Error reading historical_data.csv. Check if the file is a valid CSV.")
    st.stop()

# -----------------------------
# Load XGBoost model
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error(f"{MODEL_FILE} not found! Please upload it to the repo.")
    st.stop()

st.success("✅ All data and model loaded successfully!")

# -----------------------------
# Sidebar options
# -----------------------------
st.sidebar.header("Options")
show_fgi = st.sidebar.checkbox("Show Fear & Greed Index Data", value=True)
show_hist = st.sidebar.checkbox("Show Historical Data", value=True)
show_charts = st.sidebar.checkbox("Show Charts", value=True)
predict = st.sidebar.checkbox("Run Predictions", value=True)

# -----------------------------
# Display Fear & Greed Index
# -----------------------------
if show_fgi:
    st.subheader("📊 Fear & Greed Index Data")
    st.dataframe(df_fgi.head(20))

# -----------------------------
# Display Historical Data
# -----------------------------
if show_hist:
    st.subheader("📊 Historical Data")
    st.dataframe(df_hist.head(20))

# -----------------------------
# Charts
# -----------------------------
if show_charts:
    st.subheader("📈 Charts")

    if "Close" in df_hist.columns:
        st.line_chart(df_hist["Close"])
    else:
        st.warning("Column 'Close' not found in historical_data.csv.")

    numeric_cols = df_hist.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        st.write("Correlation Heatmap")
        corr = df_hist[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -----------------------------
# Interactive Predictions
# -----------------------------
if predict:
    st.subheader("🤖 Model Predictions (Interactive)")

    # Identify numeric features
    numeric_features = df_hist.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_features) == 0:
        st.warning("No numeric features found in historical_data.csv for prediction.")
    else:
        st.sidebar.subheader("Input Features for Prediction")

        # Create input sliders for each numeric feature
        user_input = {}
        for feature in numeric_features:
            min_val = float(df_hist[feature].min())
            max_val = float(df_hist[feature].max())
            mean_val = float(df_hist[feature].mean())
            step = (max_val - min_val)/100 if (max_val - min_val) > 0 else 1
            user_input[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val, step=step)

        # Convert user input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict(input_df)[0]
        st.write("### Prediction Result:")
        st.write(prediction)

st.info("App is fully interactive! Use the sidebar to toggle data, charts, and enter features for predictions.")
