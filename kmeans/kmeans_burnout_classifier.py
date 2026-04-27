import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import joblib

data = pd.read_csv('./data/developer_burnout_dataset_7000.csv')

target_col = 'burnout_level'

if target_col in data.columns:
    df = data.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_train)

    train_clusters = kmeans.labels_
    test_clusters = kmeans.predict(X_test)

    cluster_probs = {}

    for c in np.unique(train_clusters):
        idx = np.where(train_clusters == c)[0]
        labels = y_train[idx]

        counts = np.bincount(labels, minlength=len(np.unique(y)))
        probs = counts / counts.sum()

        cluster_probs[c] = probs

        print(f"Cluster {c} probs:", probs)

    y_pred = []

    for c in test_clusters:
        probs = cluster_probs[c]
        pred = np.argmax(probs)
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(kmeans, "kmeans_burnout_model.pkl")