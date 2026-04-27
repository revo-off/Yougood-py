import streamlit as st
import pandas as pd
import joblib
import os

st.title("Stress Level Prediction")

model_path = os.path.join(os.path.dirname(__file__), "random_forest", "random_forest_regressor.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "random_forest", "random_forest_scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.sidebar.header("Input Features")

feature_names = [ 
    "age","experience_years","daily_work_hours","sleep_hours","caffeine_intake","bugs_per_day","commits_per_day","meetings_per_day","screen_time","exercise_hours"
]

input_features = {}
for feature in feature_names:
    input_features[feature] = st.sidebar.number_input(
        feature, min_value=0.0, max_value=100.0, value=50.0, key=feature 
    )

input_df = pd.DataFrame([input_features])

try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error during input preprocessing: {e}")
    st.stop()

try:
    prediction = model.predict(input_scaled)
    st.subheader("Predicted Stress Level")
    st.write(prediction[0])
except Exception as e:
    st.error(f"Error during prediction: {e}")