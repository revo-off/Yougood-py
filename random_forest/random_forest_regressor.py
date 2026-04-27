import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import joblib

data = pd.read_csv('./data/developer_burnout_dataset_7000.csv')

target_col = 'stress_level'

if target_col in data.columns:
    df = data.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        max_depth=15,
        min_samples_leaf=5,
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2: {r2:.3f}")

joblib.dump(model, 'random_forest/random_forest_regressor.pkl')
print("Model saved successfully.")
