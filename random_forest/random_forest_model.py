import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import joblib

print("====================================")
print(" TRAINING RANDOM FOREST MODELS ")
print("====================================\n")

# Load data
data_path = './data/developer_burnout_dataset_7000.csv'
if not os.path.exists(data_path):
    print(f"Dataset not found at {data_path}")
    exit()

df = pd.read_csv(data_path)

# Handle missing values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
df = df.dropna(subset=['stress_level', 'burnout_level'])

# Define features
X = df.drop(columns=['stress_level', 'burnout_level'])

# Target for regression
y_stress = df['stress_level']

# Target for classification
le = LabelEncoder()
y_burnout = le.fit_transform(df['burnout_level'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- REGRESSION: STRESS LEVEL ---
print("Training Random Forest Regressor (Stress Level)...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(
    max_depth=15,
    min_samples_leaf=5,
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.4f}")
print(f"MSE: {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"MAE: {mean_absolute_error(y_test_r, y_pred_r):.4f}")

# Save regressor
os.makedirs('random_forest', exist_ok=True)
joblib.dump(reg_model, 'random_forest/random_forest_regressor.pkl')
print("Model saved to random_forest/random_forest_regressor.pkl\n")


# --- CLASSIFICATION: BURNOUT RISK ---
print("Training Random Forest Classifier (Burnout Risk)...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout)

clf_model = RandomForestClassifier(
    max_depth=15,
    min_samples_leaf=5,
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"F1 Score: {f1_score(y_test_c, y_pred_c, average='weighted'):.4f}")

# Save classifier
joblib.dump(clf_model, 'random_forest/random_forest_classifier.pkl')
print("Model saved to random_forest/random_forest_classifier.pkl\n")

print("Training Complete!")