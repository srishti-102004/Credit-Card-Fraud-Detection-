# ============================================================
#   CREDIT CARD FRAUD DETECTION - STREAMLIT DASHBOARD
#   File: app.py
#   Run:  streamlit run app.py
#   NOTE: Run fraud_detection.py FIRST to generate best_model.pkl
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    data         = load_model()
    model        = data['model']
    scaler       = data['scaler']
    features     = data['features']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ============================================================
# HEADER
# ============================================================
st.title("🔍 Credit Card Fraud Detection")
st.markdown("**Portfolio Project** — Machine Learning + Statistical Analysis")
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model file not found! Please run `python fraud_detection.py` first.")
    st.stop()

# ============================================================
# SIDEBAR — INPUT FORM
# ============================================================
st.sidebar.header("🧾 Enter Transaction Details")

amount   = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
time_val = st.sidebar.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**PCA Features (V1–V10)**")
st.sidebar.caption("Anonymised bank features. Leave at 0 for an average transaction.")

v_vals = {}
for i in range(1, 11):
    v_vals[f'V{i}'] = st.sidebar.slider(f'V{i}', -5.0, 5.0, 0.0, 0.1)

predict_clicked = st.sidebar.button("🔎 Predict Fraud Risk", use_container_width=True)

# ============================================================
# PREDICTION PAGE
# ============================================================
if predict_clicked:

    # Build input row
    input_data = {}
    for i in range(1, 29):
        input_data[f'V{i}'] = [v_vals.get(f'V{i}', 0.0)]

    input_data['Amount_scaled'] = [scaler.transform([[amount]])[0][0]]
    input_data['Time_scaled']   = [scaler.transform([[time_val]])[0][0]]

    input_df = pd.DataFrame(input_data)[features]

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    # --- Top metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{prob * 100:.2f}%")
    col2.metric("Prediction",        "🚨 FRAUD" if pred == 1 else "✅ LEGIT")
    col3.metric("Amount",            f"${amount:.2f}")

    # --- Alert banner ---
    if prob > 0.7:
        st.error("🚨 HIGH RISK — This transaction looks fraudulent!")
    elif prob > 0.4:
        st.warning("🟡 MEDIUM RISK — Manual review recommended")
    else:
        st.success("✅ LOW RISK — Transaction appears legitimate")

    # --- Gauge bar ---
    st.markdown("#### Fraud Risk Gauge")
    bar_color = '#e74c3c' if prob > 0.7 else '#f39c12' if prob > 0.4 else '#2ecc71'

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(['Risk'], [prob],     color=bar_color, height=0.5)
    ax.barh(['Risk'], [1 - prob], left=[prob], color='#ecf0f1', height=0.5)
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.axvline(x=0.7, color='red',    linestyle='--', alpha=0.7, linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(f'Fraud Probability: {prob * 100:.1f}%', fontsize=12)
    ax.text(0.20, 0, 'LOW',    ha='center', va='center', fontsize=9, color='gray')
    ax.text(0.55, 0, 'MEDIUM', ha='center', va='center', fontsize=9, color='gray')
    ax.text(0.85, 0, 'HIGH',   ha='center', va='center', fontsize=9, color='gray')
    st.pyplot(fig)

    # --- Explanation ---
    st.markdown("#### Why this prediction?")
    st.markdown(f"""
- **Transaction amount** entered : ${amount:.2f}
- **Fraud probability**          : {prob * 100:.2f}% (decision boundary at 50%)
- **Model used**                 : Best model from training (XGBoost / Random Forest)
- **Statistical context**        : Average fraud transaction ≈ $122 | Average legit ≈ $88
- **Key insight**                : Model uses 30 features (V1–V28 + Amount + Time) to detect patterns
    """)

# ============================================================
# DEFAULT DASHBOARD (shown before prediction)
# ============================================================
else:

    # --- Summary metrics ---
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases",        "492")
    col3.metric("Fraud Rate",         "0.17%")
    col4.metric("Features Used",      "30")

    st.markdown("---")

    # --- Two column info ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📐 Statistical Methods Used")
        st.markdown("""
| Method | Purpose |
|---|---|
| Z-Score | Detect amount anomalies |
| IQR | Identify outlier transactions |
| t-test (Hypothesis) | Compare fraud vs legit amounts |
| Bayes' Theorem | Update fraud probability |
| SMOTE | Fix class imbalance |
        """)

    with col_right:
        st.subheader("🤖 Models Trained & Compared")
        st.markdown("""
| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Handles non-linear patterns |
| XGBoost | Best overall performance |
        """)

    st.markdown("---")

    # --- Charts in tabs ---
    st.subheader("📈 Project Charts")
    tabs = st.tabs(["EDA", "Statistics", "SMOTE", "Model Evaluation"])

    chart_files = [
        "eda_overview.png",
        "stats_analysis.png",
        "smote_comparison.png",
        "model_evaluation.png"
    ]

    for tab, fname in zip(tabs, chart_files):
        with tab:
            try:
                st.image(fname, use_column_width=True)
            except Exception:
                st.info("Chart not found. Run `python fraud_detection.py` first to generate it.")

    st.markdown("---")
    st.info("👈 Use the sidebar to enter a transaction and get a real-time fraud prediction.")
