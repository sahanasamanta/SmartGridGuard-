import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os

from model_utils import transformer_forecast, train_transformer_model, AddPositionalEncoding
from anomaly_utils import detect_anomalies

# -----------------------
# Config
# -----------------------
MODEL_PATH = "tsfm_model.keras"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
DATA_PATH = "elf_dataset.xlsx"
TARGET_COL = "DEMAND"
TIME_STEPS = 20


# -----------------------
# Load model + scalers
# -----------------------
@st.cache_resource
def load_model_and_scalers():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"AddPositionalEncoding": AddPositionalEncoding}
        )
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, scaler_X, scaler_y
    else:
        return None, None, None


# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="SmartGridGuard", layout="wide")
st.title("⚡ SmartGridGuard – Energy Forecasting & Anomaly Detection")

# File uploader for retraining
uploaded_file = st.file_uploader("Upload new dataset (.xlsx) to retrain model", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.info("Training new Transformer (TSFM) model with uploaded dataset...")
    model, scaler_X, scaler_y, history, results = train_transformer_model(df, TARGET_COL)

    model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    st.success("✅ Model retrained and saved successfully!")
    st.json(results)
else:
    df = load_data()

# Dataset preview
st.markdown("### 📂 Loaded Dataset (Preview)")
st.dataframe(df.head())

# Load pretrained model
model, scaler_X, scaler_y = load_model_and_scalers()
if model is None:
    st.error("No pre-trained Transformer found. Please upload dataset to train.")
    st.stop()

# -----------------------
# Forecasting
# -----------------------
ts_preds, y_true = transformer_forecast(model, scaler_X, scaler_y, df, TARGET_COL, time_steps=TIME_STEPS)

# Detect anomalies
anomalies, iso_model = detect_anomalies(
    df,
    df[TARGET_COL].iloc[-len(ts_preds):],
    ts_preds,
    contamination=None,
    score_percentile=98.5
)

# -----------------------
# Plots
# -----------------------
st.subheader("📊 Forecasting & Anomaly Detection")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Transformer Forecast vs Actual**")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y_true, label="Actual")
    ax.plot(ts_preds, label="Transformer")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown("**Anomaly Detection Results (Isolation Forest)**")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    series_to_plot = df[TARGET_COL].iloc[-len(ts_preds):].values
    ax2.plot(series_to_plot, label="Actual", alpha=0.8)
    ax2.scatter(anomalies["index"], anomalies["actual"], color="red", label="Anomaly", s=20)
    ax2.legend()
    st.pyplot(fig2)
