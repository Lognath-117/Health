###############################################
# SymptomAtlas - AI-Powered Early Multi-Disease Detection
# Fully deployable standalone Streamlit app (error-free)
###############################################

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import shap

# Synthetic dataset generator
def generate_synthetic_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    data = {
        'sleep_hours': np.random.normal(7, 1.5, num_samples),
        'app_switch_freq': np.random.poisson(15, num_samples),
        'typing_speed': np.random.normal(250, 50, num_samples),
        'typing_errors': np.random.poisson(3, num_samples),
        'sentiment_score': np.random.normal(0, 1, num_samples),
        'speech_pauses': np.random.poisson(5, num_samples),
    }
    df = pd.DataFrame(data)

    # Create synthetic risk labels
    labels = []
    for i in range(num_samples):
        score = 0
        if df.loc[i, 'sleep_hours'] < 5.5: score += 1
        if df.loc[i, 'app_switch_freq'] > 20: score += 1
        if df.loc[i, 'typing_speed'] < 200: score += 1
        if df.loc[i, 'typing_errors'] > 5: score += 1
        if df.loc[i, 'sentiment_score'] < -1: score += 1
        if df.loc[i, 'speech_pauses'] > 8: score += 1

        if score >= 5:
            labels.append("Neurodegenerative Risk")
        elif score >= 3:
            labels.append("Cognitive Decline")
        elif score >= 2:
            labels.append("Anxiety")
        elif score >= 1:
            labels.append("Depression")
        else:
            labels.append("Healthy")

    df['risk_label'] = labels
    return df

# Build ML model pipeline
def train_model(data):
    features = ['sleep_hours', 'app_switch_freq', 'typing_speed', 'typing_errors', 'sentiment_score', 'speech_pauses']
    X = data[features]
    y = data['risk_label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train, check_additivity=False)  # ✅ FIXED LINE

    return model, scaler, report, explainer, shap_values, features, X_train

# Streamlit UI starts here
st.set_page_config(page_title="SymptomAtlas AI", layout="wide")
st.title("🧠 SymptomAtlas — AI-Powered Early Multi-Disease Detection")

st.markdown("""
Detect early risk signals for multiple diseases using behavioral data:
- Sleep patterns
- App switching frequency
- Typing speed and errors
- Text sentiment shifts
- Speech pause patterns
""")

# Sidebar for data input
st.sidebar.header("Data Input Options")
data_source = st.sidebar.radio("Choose Dataset:", ["Use Synthetic Demo Data", "Upload Your Own CSV"])

if data_source == "Use Synthetic Demo Data":
    st.sidebar.success("Using auto-generated synthetic data for demo.")
    data = generate_synthetic_data(1000)
else:
    uploaded_file = st.sidebar.file_uploader("Upload behavioral data CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'risk_label' not in data.columns:
            st.warning("Uploaded data must include 'risk_label' column for supervised training.")
            st.stop()
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Display data preview
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Train model
st.subheader("Model Training")
st.write("Training RandomForestClassifier on behavioral data...")
model, scaler, report, explainer, shap_values, features, X_train = train_model(data)

st.text("Classification Report:")
st.code(report, language='text')

# User prediction section
st.subheader("Make Predictions on New Data")
example_input = {}
columns = ['sleep_hours', 'app_switch_freq', 'typing_speed', 'typing_errors', 'sentiment_score', 'speech_pauses']
col1, col2, col3 = st.columns(3)

for idx, col in enumerate(columns):
    with [col1, col2, col3][idx % 3]:
        example_input[col] = st.number_input(f"{col}", value=float(data[col].mean()), step=0.1)

if st.button("Predict Disease Risk"):
    input_df = pd.DataFrame([example_input])
    input_scaled = scaler.transform(input_df)
    pred_class = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)
    class_probs = dict(zip(model.classes_, pred_proba[0]))

    st.success(f"Predicted Risk: **{pred_class}**")
    st.subheader("Prediction Probabilities")
    st.write(class_probs)

# Visualize behavioral drift
st.subheader("Behavioral Drift Visualizations")
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for idx, feature in enumerate(columns):
    row, col = divmod(idx, 3)
    sns.lineplot(data=data[feature], ax=axs[row][col])
    axs[row][col].set_title(feature)
st.pyplot(fig)

# SHAP Explainability Summary
st.subheader("Explainable AI Insights (SHAP Values)")
st.write("Global feature importance across training data:")
shap.summary_plot(shap_values, features=features, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')

st.markdown("---")
st.markdown("Project developed by **Lognath A.** 🚀 Fully deployable Streamlit AI app.")
