import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from sklearn.model_selection import KFold

# Custom Bootstrap Validation (Out-Of-Bag)
class BootstrapCV:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def split(self, X, y=None, groups=None):
        np.random.seed(self.random_state)
        n_samples = len(X) if isinstance(X, np.ndarray) else X.shape[0]
        for _ in range(self.n_splits):
            train_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))
            yield train_idx, test_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    df.dropna(subset=['burnout_level', 'stress_level'], inplace=True)
    return df

def main():
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

    # Train/Test splits
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_stress, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_burnout, test_size=0.2, random_state=42, stratify=y_burnout)

    print("\n--- 1. OPTIMIZING REGRESSION MODELS (Stress Level) ---")
    
    # 1.1 Random Forest Optimization
    print("\nRandom Forest Regressor (K-Fold CV)...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=10, cv=kf, scoring='r2', n_jobs=-1, verbose=1, random_state=42)
    rf_search.fit(X_train_r, y_train_r)
    print(f"Best RF Params: {rf_search.best_params_}")
    print(f"Best RF Cross-Val R2: {rf_search.best_score_:.4f}")
    # Saving best model
    joblib.dump(rf_search.best_estimator_, 'random_forest/random_forest_regressor_optimized.pkl')

    # 1.2 SVM SVR Optimization
    print("\nSupport Vector Regressor (SVR) (Bootstrapping OOB)...")
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    boot_cv = BootstrapCV(n_splits=5, random_state=42)
    svr_search = RandomizedSearchCV(SVR(), svr_param_grid, n_iter=5, cv=boot_cv, scoring='r2', n_jobs=-1, verbose=1, random_state=42)
    svr_search.fit(X_train_r, y_train_r)
    print(f"Best SVR Params: {svr_search.best_params_}")
    print(f"Best SVR Cross-Val R2: {svr_search.best_score_:.4f}")
    # Saving best model
    joblib.dump(svr_search.best_estimator_, 'SVM/svr_stress_model_optimized.pkl')

    # 1.3 TabNet Regression Optimization (Simple Grid)
    print("\nTabNet Regressor (Simplified)...")
    best_tab_r2 = -1
    best_tab_params = {}
    best_tabnet_reg = None
    for n_a in [8, 16]:
        for lr in [0.02, 0.01]:
            clf = TabNetRegressor(n_d=n_a, n_a=n_a, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=lr), verbose=0)
            clf.fit(
                X_train=X_train_r, y_train=y_train_r.values.reshape(-1, 1),
                eval_set=[(X_test_r, y_test_r.values.reshape(-1, 1))],
                eval_name=['valid'], eval_metric=['mse'],
                max_epochs=30, patience=10, batch_size=256
            )
            preds = clf.predict(X_test_r)
            score = r2_score(y_test_r, preds)
            if score > best_tab_r2:
                best_tab_r2 = score
                best_tab_params = {'n_a': n_a, 'n_d': n_a, 'lr': lr}
                best_tabnet_reg = clf
    print(f"Best TabNet Regressor Params: {best_tab_params}")
    print(f"Best TabNet Test R2: {best_tab_r2:.4f}")
    if best_tabnet_reg:
        best_tabnet_reg.save_model('TabNet/tabnet_stress_model_optimized')

    print("\n--- 2. OPTIMIZING CLASSIFICATION MODELS (Burnout Risk) ---")
    
    # 2.1 Random Forest Optimization (Baseline - No Resampling)
    print("\nRandom Forest Classifier (Baseline)...")
    rf_c_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_c_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_c_param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
    rf_c_search.fit(X_train_c, y_train_c)
    print(f"Best RF Classifier Params: {rf_c_search.best_params_}")
    print(f"Best RF Classifier Cross-Val Acc: {rf_c_search.best_score_:.4f}")
    best_rf_model = rf_c_search.best_estimator_
    joblib.dump(best_rf_model, 'random_forest/random_forest_classifier_optimized.pkl')

    # 2.2 SVC Optimization (Baseline - No Resampling)
    print("\nSupport Vector Classifier (Baseline)...")
    svc_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svc_search = RandomizedSearchCV(SVC(probability=True, random_state=42), svc_param_grid, n_iter=5, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
    svc_search.fit(X_train_c, y_train_c)
    print(f"Best SVC Params: {svc_search.best_params_}")
    print(f"Best SVC Cross-Val Acc: {svc_search.best_score_:.4f}")
    # Saving best model
    best_svc_model = svc_search.best_estimator_
    joblib.dump(best_svc_model, 'SVM/svc_burnout_model_optimized.pkl')

    # 2.3 TabNet Classifier Optimization (Simple Grid with Baseline)
    print("\nTabNet Classifier (Baseline)...")
    best_tab_c_acc = -1
    best_tab_c_params = {}
    best_tabnet_clf = None
    for n_a in [8, 16]:
        for lr in [0.02, 0.01]:
            clf = TabNetClassifier(n_d=n_a, n_a=n_a, optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=lr), verbose=0)
            clf.fit(
                X_train=X_train_c, y_train=y_train_c,
                eval_set=[(X_test_c, y_test_c)],
                eval_name=['valid'], eval_metric=['accuracy'],
                max_epochs=30, patience=10, batch_size=256
            )
            preds = clf.predict(X_test_c)
            acc = accuracy_score(y_test_c, preds)
            if acc > best_tab_c_acc:
                best_tab_c_acc = acc
                best_tab_c_params = {'n_a': n_a, 'n_d': n_a, 'lr': lr}
                best_tabnet_clf = clf
    print(f"Best TabNet Classifier Params: {best_tab_c_params}")
    print(f"Best TabNet Test Accuracy: {best_tab_c_acc:.4f}")
    if best_tabnet_clf:
        best_tabnet_clf.save_model('TabNet/tabnet_burnout_model_optimized')

    # 2.4 KMeans Classifier Optimization (Random Over-Sampling)
    print("\nKMeans Classifier (Random Over-Sampling)...")
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_c_ros, y_train_c_ros = ros.fit_resample(X_train_c, y_train_c)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_c_ros)
    
    # Mapping clusters to probabilities for the classes
    train_clusters = kmeans.predict(X_train_c_ros)
    probs_dict = {}
    for cluster in range(3):
        mask = (train_clusters == cluster)
        if np.sum(mask) > 0:
            counts = np.bincount(y_train_c_ros[mask], minlength=3)
            probs_dict[cluster] = counts / counts.sum()
        else:
            probs_dict[cluster] = np.ones(3) / 3
            
    test_clusters = kmeans.predict(X_test_c)
    test_preds = [np.argmax(probs_dict[c]) for c in test_clusters]
    kmeans_acc = accuracy_score(y_test_c, test_preds)
    print(f"Best KMeans Acc (via mapped probabilities): {kmeans_acc:.4f}")
    
    os.makedirs('kmeans', exist_ok=True)
    joblib.dump(kmeans, 'kmeans/kmeans_burnout_model.pkl')
    joblib.dump(scaler, 'kmeans/kmeans_burnout_scaler.pkl')
    joblib.dump(probs_dict, 'kmeans/kmeans_burnout_cluster_probs.pkl')

if __name__ == '__main__':
    main()
