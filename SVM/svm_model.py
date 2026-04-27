import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import joblib

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
        
    df.dropna(subset=['burnout_level', 'stress_level'], inplace=True)
    
    return df

def train_svm_pipeline(data_path):
    print("Loading data...")
    df = load_and_preprocess_data(data_path)
    
    # Features for stress level prediction
    X = df.drop(columns=['stress_level', 'burnout_level'])
    y_stress = df['stress_level']
    y_burnout = df['burnout_level']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Training SVM for Stress Level (Regression) ---")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)
    
    # Train SVR
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_s, y_train_s)
    
    stress_preds = svr.predict(X_test_s)
    print(f"Stress Level MSE: {mean_squared_error(y_test_s, stress_preds):.4f}")
    print(f"Stress Level R2 Score: {r2_score(y_test_s, stress_preds):.4f}")
    
    print("\n--- Training SVM for Burnout Risk (Classification) ---")
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_scaled, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout)
    
    # Train SVC
    svc = SVC(kernel='rbf', C=1.0, probability=True)
    svc.fit(X_train_b, y_train_b)
    
    burnout_preds = svc.predict(X_test_b)
    print(f"Burnout Level Accuracy: {accuracy_score(y_test_b, burnout_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test_b, burnout_preds))
    
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save models
    joblib.dump(svr, os.path.join(base_dir, 'svr_stress_model.pkl'))
    joblib.dump(svc, os.path.join(base_dir, 'svc_burnout_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'svm_scaler.pkl'))
    print("\nModels saved successfully.")

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "data", "developer_burnout_dataset_7000.csv")
    train_svm_pipeline(dataset_path)
