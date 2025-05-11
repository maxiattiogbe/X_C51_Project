import joblib
import numpy as np

# Load the saved model once
_model = joblib.load("XGBoost_policy_model.pkl")

def predict_score(features: list[float]) -> float:
    """Given a feature vector, predict a policy score (lower is better)."""
    return float(_model.predict(np.array(features).reshape(1, -1))[0])
