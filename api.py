import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Lazy loading of models
def load_model(model_name: str):
    model_paths = {
        "random_forest": "random_forest/random_forest_regressor.pkl",
        "svm": "svm/svr_stress_model.pkl",
        "tabnet": "tabNet/tabnet_stress_model.zip",
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Model '{model_name}' not found.")

    try:
        return joblib.load(model_paths[model_name])
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file for '{model_name}' not found at {model_paths[model_name]}")

# Appel de FastAPI pour créer l'API de prédiction de niveau de stress des salariés
app = FastAPI(title="Stress Level Prediction API", version="1.0")

# Définition du modèle de données pour les fonctionnalités d'entrée de l'API
class StressLevelFeatures(BaseModel): 
    age: float = Field(..., description="Age du salarié")
    experience_years: float = Field(..., description="Années d'expérience du salarié")
    daily_work_hours: float = Field(..., description="Heures de travail quotidiennes du salarié")
    sleep_hours: float = Field(..., description="Heures de sommeil du salarié")
    caffeine_intake: float = Field(..., description="Consommation de caféine du salarié")
    bugs_per_day: float = Field(..., description="Nombre de bugs par jour du salarié")
    commits_per_day: float = Field(..., description="Nombre de commits par jour du salarié")
    meetings_per_day: float = Field(..., description="Nombre de réunions par jour du salarié")
    screen_time: float = Field(..., description="Temps d'écran quotidien du salarié")
    exercise_hours: float = Field(..., description="Heures d'exercice du salarié")

# Requete pour vérifier que l'API est opérationnelle
@app.get("/stress/{model_name}")
def stress(model_name: str):
    model_name = model_name.lower()
    try:
        load_model(model_name)
        return {"status": "ok"}
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

# Requete pour la prédiction du niveau de stress du salarié
@app.post("/predict/{model_name}")
def predict(model_name: str, payload: StressLevelFeatures):
    model_name = model_name.lower()
    try:
        model = load_model(model_name)
        X = pd.DataFrame([payload.model_dump()])
        y_pred = model.predict(X)[0]

        return {
            "model_used": model_name,
            "prediction_stress_level": float(y_pred)
        }
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}