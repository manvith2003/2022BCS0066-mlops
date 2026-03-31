from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import pandas as pd
import mlflow
import os

app = FastAPI()

NAME = "Manvith M"
ROLL_NO = "2022BCS0066"

# A basic schema for the wine dataset (all 13 features from load_wine)
class WineFeatures(BaseModel):
    alcohol: float = 13.2
    malic_acid: float = 2.7
    ash: float = 2.4
    alcalinity_of_ash: float = 19.0
    magnesium: float = 100.0
    total_phenols: float = 2.2
    flavanoids: float = 2.0
    nonflavanoid_phenols: float = 0.3
    proanthocyanins: float = 1.3
    color_intensity: float = 4.2
    hue: float = 1.0
    od280_od315_of_diluted_wines: float = 2.8
    proline: float = 800.0

@app.get("/")
@app.get("/health")
def health_check():
    return {"Name": NAME, "Roll No": ROLL_NO, "status": "healthy"}

@app.post("/predict")
def predict(features: WineFeatures):
    # In a full setup, we would load the trained model dynamically from MLflow or local file.
    # We will assume model.pkl is present for inference (will be generated manually or by github actions).
    # Since this is a demo, if model doesn't exist we return a mock prediction.
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        df = pd.DataFrame([features.dict()])
        
        # Identify expected features if it's the reduced model
        expected_features = getattr(model, "feature_names_in_", df.columns)
        df_selected = df[expected_features] if len(expected_features) < len(df.columns) else df
        
        pred = model.predict(df_selected)[0]
    else:
        # Mock prediction if no model found
        pred = 1

    return {
        "Prediction": int(pred),
        "Name": NAME,
        "Roll No": ROLL_NO
    }
