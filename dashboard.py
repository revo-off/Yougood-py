import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as st_sns
import torch
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Developer Wellness Dashboard", layout="wide")

# ==========================================
# 1. LOAD DATA & FIT GLOBAL PREPROCESSORS
# ==========================================
@st.cache_data
def load_and_prepare_data():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "developer_burnout_dataset_7000.csv")
    df = pd.read_csv(dataset_path)
    
    if 'burnout_level' in df.columns:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        df.dropna(subset=['stress_level', 'burnout_level'], inplace=True)
    
    return df

df = load_and_prepare_data()

# Prepare Global Scaler & Encoder exactly as in training
X_full = df.drop(columns=['stress_level', 'burnout_level'], errors='ignore')
global_scaler = StandardScaler()
global_scaler.fit(X_full)

le = LabelEncoder()
le.fit(df['burnout_level'])

# ==========================================
# 2. SIDEBAR - INPUT FEATURES
# ==========================================
st.sidebar.header("Developer Profile")

feature_names = [ 
    "age","experience_years","daily_work_hours","sleep_hours",
    "caffeine_intake","bugs_per_day","commits_per_day","meetings_per_day",
    "screen_time","exercise_hours"
]

input_features = {}
for feature in feature_names:
    default_val = float(df[feature].median()) if feature in df.columns else 50.0
    input_features[feature] = st.sidebar.number_input(
        feature.replace('_', ' ').title(), 
        min_value=0.0, 
        max_value=float(df[feature].max()*1.5) if feature in df.columns else 100.0, 
        value=default_val
    )

input_df = pd.DataFrame([input_features])
input_scaled = global_scaler.transform(input_df)

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.title("🧠 Developer Wellness & Burnout Prediction")
st.markdown("Predict the stress level and burnout risk of a developer based on their daily habits and work metrics.")

col1, col2 = st.columns(2)

# ==========================================
# 3. REGRESSION: STRESS LEVEL
# ==========================================
with col1:
    st.header("📈 Stress Level Prediction")
    reg_model_choice = st.selectbox("Select Regression Model:", ["Random Forest", "Support Vector Regressor (SVR)", "TabNet"])
    
    stress_prediction = None
    try:
        if reg_model_choice == "Random Forest":
            path = os.path.join(os.path.dirname(__file__), "random_forest", "random_forest_regressor.pkl")
            model = joblib.load(path)
            stress_prediction = model.predict(input_scaled)[0]
            
        elif reg_model_choice == "Support Vector Regressor (SVR)":
            path = os.path.join(os.path.dirname(__file__), "SVM", "svr_stress_model.pkl")
            model = joblib.load(path)
            stress_prediction = model.predict(input_scaled)[0]
            
        elif reg_model_choice == "TabNet":
            path = os.path.join(os.path.dirname(__file__), "TabNet", "tabnet_stress_model.zip")
            model = TabNetRegressor()
            model.load_model(path)
            stress_prediction = model.predict(input_scaled)[0][0]
            
        if stress_prediction is not None:
            st.success(f"**Predicted Stress Level: {stress_prediction:.2f} / 100**")
            st.progress(min(int(stress_prediction), 100))
            
    except Exception as e:
        st.error(f"Error loading {reg_model_choice}: {e}")

# ==========================================
# 4. CLASSIFICATION: BURNOUT RISK
# ==========================================
with col2:
    st.header("🔥 Burnout Risk Classification")
    clf_model_choice = st.selectbox("Select Classification Model:", ["Support Vector Classifier (SVC)", "TabNet", "KMeans"])
    
    burnout_prediction = None
    try:
        if clf_model_choice == "Support Vector Classifier (SVC)":
            path = os.path.join(os.path.dirname(__file__), "SVM", "svc_burnout_model.pkl")
            model = joblib.load(path)
            pred = model.predict(input_scaled)[0]
            burnout_prediction = pred if isinstance(pred, str) else le.inverse_transform([pred])[0]
            
        elif clf_model_choice == "TabNet":
            path = os.path.join(os.path.dirname(__file__), "TabNet", "tabnet_burnout_model.zip")
            model = TabNetClassifier()
            model.load_model(path)
            pred = model.predict(input_scaled)[0]
            burnout_prediction = le.inverse_transform([pred])[0]
            
        elif clf_model_choice == "KMeans":
            km_path = os.path.join(os.path.dirname(__file__), "kmeans", "kmeans_model.pkl")
            if not os.path.exists(km_path):
                km_path = os.path.join(os.path.dirname(__file__), "kmeans", "kmeans_burnout_model.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "kmeans", "kmeans_scaler.pkl")
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(os.path.dirname(__file__), "kmeans", "kmeans_burnout_scaler.pkl")
                
            model = joblib.load(km_path)
            km_scaler = joblib.load(scaler_path)
            probs_dict = joblib.load(os.path.join(os.path.dirname(__file__), "kmeans", "kmeans_burnout_cluster_probs.pkl"))
            
            km_input_scaled = km_scaler.transform(input_df)
            cluster = model.predict(km_input_scaled)[0]
            pred = np.argmax(probs_dict[cluster])
            burnout_prediction = le.inverse_transform([pred])[0]
            
        if burnout_prediction is not None:
            color = "🟢" if burnout_prediction == "Low" else "🟡" if burnout_prediction == "Medium" else "🔴"
            st.warning(f"**Predicted Burnout Risk: {color} {burnout_prediction}**")
            
    except Exception as e:
        st.error(f"Error loading {clf_model_choice}: {e}")

st.markdown("---")

# ==========================================
# 4.5. GLOBAL BENCHMARK TABLE
# ==========================================
st.header("🏆 Model Benchmark Summary")
st.write("Compare the baseline performance of all trained models based on multiple metrics.")

bench_col1, bench_col2 = st.columns(2)

with bench_col1:
    st.subheader("📈 Stress Level (Regression)")
    reg_bench_df = pd.DataFrame({
        "Model": ["TabNet", "Support Vector Regressor", "Random Forest"],
        "R² Score": [0.7796, 0.7734, 0.7682],
        "MSE": [112.28, 115.45, 118.06],
        "MAE": [8.38, 8.52, 8.66]
    })
    
    st.dataframe(
        reg_bench_df.style.format({
            "R² Score": "{:.4f}",
            "MSE": "{:.2f}",
            "MAE": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

with bench_col2:
    st.subheader("🔥 Burnout Risk (Classification)")
    clf_bench_df = pd.DataFrame({
        "Model": ["TabNet", "Support Vector Classifier", "KMeans"],
        "Accuracy": [0.7777, 0.7682, 0.5117],
        "F1 Score": [0.7775, 0.7669, 0.5045]
    })
    
    st.dataframe(
        clf_bench_df.style.format({
            "Accuracy": "{:.4f}",
            "F1 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==========================================
# 5. DATA INSIGHTS & CORRELATIONS
# ==========================================
st.header("📊 Data Insights: What Impacts Developers the Most?")

st.markdown("### Feature Correlations")
st.write("Determine which daily habits have the strongest impact on Stress and Burnout.")

import seaborn as sns
import matplotlib.patches as mpatches

# Encode burnout specifically for correlation plotting if not already numerical
df_plot = df.copy()
df_plot['burnout_encoded'] = le.transform(df_plot['burnout_level'])

# Calculate correlations
correlations = df_plot.drop(columns=['burnout_level']).corr()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Top Factors for Stress Level")
    stress_corr = correlations['stress_level'].drop(['stress_level', 'burnout_encoded'], errors='ignore').sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5.17))
    colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in stress_corr.values]
    sns.barplot(x=stress_corr.values, y=stress_corr.index, palette=colors, ax=ax)
    ax.set_ylabel("Features")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title("Features vs Stress Level")
    
    # Legend
    pos_patch = mpatches.Patch(color='#ff6b6b', label='Increases Stress')
    neg_patch = mpatches.Patch(color='#4ecdc4', label='Reduces Stress')
    ax.legend(handles=[pos_patch, neg_patch], title="Impact")
    
    st.pyplot(fig)

with col4:
    st.subheader("Average Profiles per Burnout Risk")
    
    # Order columns logically if they exist
    burnout_order = ['Low', 'Medium', 'High']
    valid_cols = [c for c in burnout_order if c in df['burnout_level'].unique()]
    
    # Calculate means grouped by Burnout Level
    feat_means = df.groupby('burnout_level')[feature_names].mean().T
    feat_means = feat_means[valid_cols]
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    normalized_means = feat_means.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8), axis=1)
    
    sns.heatmap(normalized_means, annot=feat_means, cmap="Reds", fmt=".1f", 
                cbar=True, cbar_kws={'label': 'Relative Intensity (Min to Max)'}, linewidths=0.5, ax=ax2)
    
    ax2.set_title("Feature Thresholds (Averages)")
    ax2.set_ylabel("Features")
    ax2.set_xlabel("Burnout Level")
    st.pyplot(fig2)