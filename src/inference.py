# src/inference.py

"""
Lightweight FastAPI service wrapping a saved sklearn pipeline.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Any, Dict

app = FastAPI(
    title="Kickstarter Success Predictor",
    description="Binary success prediction for Kickstarter projects",
)

# Where to find your pipeline artifact (override via env var if needed)
MODEL_PATH = os.getenv(
    "PIPELINE_PATH",
    "Best_Model/best_kickstarter_model_xgb_optimized_pipeline.pkl"
)

# Try to load at import time; fall back to None so imports/tests don't fail
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception:
    pipeline = None


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    Expects a JSON object mapping feature names to values, e.g.
      { "goal": 5000, "campaign_duration": 30.0, ... }

    Returns:
      {
        "prediction": 0 or 1,
        "probability": float
      }
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build a DataFrame for a single sample
    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # Run prediction
    try:
        pred = pipeline.predict(df)[0]
        proba = pipeline.predict_proba(df)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return {
        "prediction": int(pred),
        "probability": float(proba)
    }