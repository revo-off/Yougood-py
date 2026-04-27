import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

data = pd.read_csv('./data/developer_burnout_dataset_7000.csv')

target_col = 'burnout_level'
if target_col in data.columns:
    df_imputed = data.copy()

    for col in df_imputed.select_dtypes(include=[np.number]).columns:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    df_imputed.dropna(subset=[target_col], inplace=True)

    X = df_imputed.drop(columns=[target_col])
    y = df_imputed[target_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Original training target shape: {Counter(y_train)}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training target shape: {Counter(y_train_resampled)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R-squared (R²): {r2:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

joblib.dump(model, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")