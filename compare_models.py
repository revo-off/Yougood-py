import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import torch
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import warnings
import os

warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    df.dropna(subset=['burnout_level', 'stress_level'], inplace=True)
    return df

def main():
    print("="*50)
    print(" MODEL COMPARISON TOOL ")
    print("="*50)

    dataset_path = "./data/developer_burnout_dataset_7000.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    df = load_data(dataset_path)

    X = df.drop(columns=['stress_level', 'burnout_level'])
    y_stress = df['stress_level']
    
    le = LabelEncoder()
    y_burnout = le.fit_transform(df['burnout_level'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 1. EVALUATE REGRESSION MODELS (Stress Level)
    # ---------------------------------------------------------
    print("\n[ STRESS LEVEL PREDICTION (Regression) ]")
    print("-" * 55)
    print(f"{'Model':<15} | {'MSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 55)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)

    # SVM (SVR)
    try:
        svr = joblib.load('SVM/svr_stress_model.pkl')
        svr_preds = svr.predict(X_test_r)
        svr_mse = mean_squared_error(y_test_r, svr_preds)
        svr_mae = mean_absolute_error(y_test_r, svr_preds)
        svr_r2 = r2_score(y_test_r, svr_preds)
        print(f"{'SVM (SVR)':<15} | {svr_mse:<10.4f} | {svr_mae:<10.4f} | {svr_r2:<10.4f}")
    except FileNotFoundError:
        print(f"{'SVM (SVR)':<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    # Random Forest Regressor
    try:
        rf_path = 'random_forest/random_forest_regressor.pkl'
        rf_reg = joblib.load(rf_path)
        rf_preds = rf_reg.predict(X_test_r)
        rf_mse = mean_squared_error(y_test_r, rf_preds)
        rf_mae = mean_absolute_error(y_test_r, rf_preds)
        rf_r2 = r2_score(y_test_r, rf_preds)
        print(f"{'Random Forest':<15} | {rf_mse:<10.4f} | {rf_mae:<10.4f} | {rf_r2:<10.4f}")
    except FileNotFoundError:
        print(f"{'Random Forest':<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")

    # TabNet Regressor
    try:
        tabnet_reg = TabNetRegressor()
        tabnet_reg.load_model('TabNet/tabnet_stress_model.zip')
        tab_preds = tabnet_reg.predict(X_test_r)
        tab_mse = mean_squared_error(y_test_r, tab_preds)
        tab_mae = mean_absolute_error(y_test_r, tab_preds)
        tab_r2 = r2_score(y_test_r, tab_preds)
        print(f"{'TabNet':<15} | {tab_mse:<10.4f} | {tab_mae:<10.4f} | {tab_r2:<10.4f}")
    except Exception as e:
        print(f"{'TabNet':<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")


    # ---------------------------------------------------------
    # 2. EVALUATE CLASSIFICATION MODELS (Burnout Level)
    # ---------------------------------------------------------
    print("\n[ BURNOUT RISK PREDICTION (Classification) ]")
    print("-" * 45)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 45)

    # Test split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout)

    # SVM (SVC)
    try:
        svc = joblib.load('SVM/svc_burnout_model.pkl')
        svc_preds = svc.predict(X_test_c)

        if isinstance(svc_preds[0], str) or np.issubdtype(type(svc_preds[0]), np.character):
            svc_preds = le.transform(svc_preds)
            
        svc_acc = accuracy_score(y_test_c, svc_preds)
        svc_f1 = f1_score(y_test_c, svc_preds, average='weighted')
        print(f"{'SVM (SVC)':<15} | {svc_acc:<10.4f} | {svc_f1:<10.4f}")
    except FileNotFoundError:
        print(f"{'SVM (SVC)':<15} | {'N/A':<10} | {'N/A':<10}")

    # TabNet Classifier
    try:
        tabnet_clf = TabNetClassifier()
        tabnet_clf.load_model('TabNet/tabnet_burnout_model.zip')
        tab_c_preds = tabnet_clf.predict(X_test_c)
        tab_c_acc = accuracy_score(y_test_c, tab_c_preds)
        tab_c_f1 = f1_score(y_test_c, tab_c_preds, average='weighted')
        print(f"{'TabNet':<15} | {tab_c_acc:<10.4f} | {tab_c_f1:<10.4f}")
    except Exception as e:
        print(f"{'TabNet':<15} | {'N/A':<10} | {'N/A':<10}")

    # KMeans Classifier
    try:
        km_model_path = 'kmeans/kmeans_model.pkl' 
        km_scaler_path = 'kmeans/kmeans_scaler.pkl'
        
        kmeans = joblib.load(km_model_path)
        kmeans_scaler = joblib.load(km_scaler_path)
        cluster_probs = joblib.load('kmeans/kmeans_burnout_cluster_probs.pkl')
        
        X_km = df.drop(columns=['stress_level', 'burnout_level'])
        X_kmeans_train, X_kmeans_test, y_kmeans_train, y_kmeans_test = train_test_split(
            X_km, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout
        )
        X_kmeans_test_scaled = kmeans_scaler.transform(X_kmeans_test)

        test_clusters = kmeans.predict(X_kmeans_test_scaled)
        km_preds = []
        for c in test_clusters:
            probs = cluster_probs[c]
            km_preds.append(np.argmax(probs))
            
        km_acc = accuracy_score(y_kmeans_test, km_preds)
        km_f1 = f1_score(y_kmeans_test, km_preds, average='weighted')
        print(f"{'KMeans':<15} | {km_acc:<10.4f} | {km_f1:<10.4f}")
        
    except FileNotFoundError:
        print(f"{'KMeans':<15} | {'N/A':<10} | {'N/A':<10}")
        
    print("-" * 45)

if __name__ == "__main__":
    main()