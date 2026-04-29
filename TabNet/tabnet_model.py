import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import torch
import os

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values computationally
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
        
    df.dropna(subset=['burnout_level', 'stress_level'], inplace=True)
    return df

def train_tabnet_pipeline(data_path):
    print("Loading data...")
    df = load_and_preprocess_data(data_path)
    
    # Independent variables
    X = df.drop(columns=['stress_level', 'burnout_level'])
    y_stress = df['stress_level'].values.reshape(-1, 1) 
    
    # Label encoding for classification target
    le = LabelEncoder()
    y_burnout = le.fit_transform(df['burnout_level'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Training TabNet for Stress Level (Regression) ---")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)
    X_train_s, X_valid_s, y_train_s, y_valid_s = train_test_split(X_train_s, y_train_s, test_size=0.2, random_state=42)
    
    # Configure TabNet Regressor
    tabnet_reg = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                                 optimizer_params=dict(lr=2e-2),
                                 scheduler_params={"step_size":10, "gamma":0.9},
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                 mask_type='entmax')
                                 
    tabnet_reg.fit(X_train=X_train_s, y_train=y_train_s,
                   eval_set=[(X_train_s, y_train_s), (X_valid_s, y_valid_s)],
                   eval_name=['train', 'valid'],
                   eval_metric=['rmse'],
                   max_epochs=100, patience=20, batch_size=256, virtual_batch_size=128)
                   
    stress_preds = tabnet_reg.predict(X_test_s)
    print(f"Stress Level MSE: {mean_squared_error(y_test_s, stress_preds):.4f}")
    print(f"Stress Level R2 Score: {r2_score(y_test_s, stress_preds):.4f}")
    
    print("\n--- Training TabNet for Burnout Risk (Classification) ---")
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_scaled, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout)
    X_train_b, X_valid_b, y_train_b, y_valid_b = train_test_split(X_train_b, y_train_b, test_size=0.2, random_state=42, stratify=y_train_b)
    
    # Configure TabNet Classifier
    tabnet_clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                                  optimizer_params=dict(lr=2e-2),
                                  scheduler_params={"step_size":10, "gamma":0.9},
                                  scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                  mask_type='sparsemax')
                                  
    tabnet_clf.fit(X_train=X_train_b, y_train=y_train_b,
                   eval_set=[(X_train_b, y_train_b), (X_valid_b, y_valid_b)],
                   eval_name=['train', 'valid'],
                   eval_metric=['accuracy'],
                   max_epochs=100, patience=20, batch_size=256, virtual_batch_size=128)
                   
    burnout_preds = tabnet_clf.predict(X_test_b)
    print(f"Burnout Risk Accuracy: {accuracy_score(y_test_b, burnout_preds):.4f}")
    
    encoded_classes = le.inverse_transform(np.unique(y_test_b))
    print("Classification Report:")
    print(classification_report(y_test_b, burnout_preds, target_names=encoded_classes))
    
    print("\nNOTE: Base model saving disabled. Use optimize_models.py to generate .zip files.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "data", "developer_burnout_dataset_7000.csv")
    train_tabnet_pipeline(dataset_path)
