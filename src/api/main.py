from fastapi import FastAPI
from src.api.pydantic_models import RiskRequest, RiskResponse
import mlflow.sklearn

app = FastAPI(title="Credit Risk Scoring API")

# Load model from MLflow
model_name = "RandomForest"
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

@app.post("/predict", response_model=RiskResponse)
def predict_risk(data: RiskRequest):
    features = [[
        data.Frequency, data.Monetary, data.Recency,
        data.TransactionHour, data.TransactionDay, data.TransactionMonth, data.TransactionYear
    ]]
    prob = model.predict_proba(features)[0][1]
    return {"risk_probability": prob}
