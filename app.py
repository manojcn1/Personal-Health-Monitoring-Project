import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model and feature list
model = joblib.load("body_fat_predictor.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Body Fat % Predictor", layout="centered")
st.title("🧠 Body Fat % Predictor")
st.markdown("""
Enter your current metrics below to predict your **Body Fat % for tomorrow**.
""")

# Sidebar for input
st.sidebar.header("📥 Input Your Current Metrics")
user_input = {}

for feature in features:
    if 'Fat' in feature:
        default = 20.0
        max_val = 50.0
    elif 'BMI' in feature:
        default = 25.0
        max_val = 40.0
    elif 'Weight' in feature:
        default = 75.0
        max_val = 150.0
    else:
        default = 50.0
        max_val = 100.0

    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=0.0,
        max_value=max_val,
        step=0.1,
        value=default
    )

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("🔮 Predict Body Fat %"):
    prediction = model.predict(input_df)[0]
    st.metric(label="Predicted Body Fat % (Tomorrow)", value=f"{prediction:.2f}%")

# Optional: show feature importance
st.markdown("---")
st.subheader("📊 Feature Importance")

# Load feature importances if available
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(importance_df.set_index('Feature'))
else:
    st.write("Feature importance not available for this model.")

st.markdown("""
*Tip: Use this to understand which metrics influence your predicted body fat the most.*
""")
