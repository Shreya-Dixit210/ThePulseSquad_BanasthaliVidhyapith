"""
model_adapter.py
Acts as the interface between the dashboard and the predictive model.

Predicts the probability that a patient's glucose will reach >= 180 mg/dL
within the next 120 minutes.

Current status:
- Features are extracted in `compute_features()`.
- `predict_risk()` currently uses a transparent heuristic placeholder rule.
- A clear TODO block shows how to load and query 'models/glucose_model.joblib'
  once the trained model artifact is plugged in.
"""

from typing import Dict, Any, Tuple
from functools import lru_cache
import os
import numpy as np
import pandas as pd
import joblib

# The exact feature ordering expected by the trained LightGBM model
FEATURE_COLUMNS = [
    "glu_now",
    "glu_change_30",
    "mins_since_meal",
    "meal_carbs",
    "Age",
    "Gender",
    "BMI",
    "A1c",
    "fasting_glu",
    "Insulin",
    "HDL",
    "Triglycerides",
]

# Path to the trained model artifact (when provided)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "glucose_model.joblib")
@lru_cache(maxsize=1)
def _load_artifact():
    """Load the saved model file once and reuse it on every later call."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# True when the trained model file was found; the dashboard can use this for a demo-mode tag
USING_REAL_MODEL = _load_artifact() is not None

def compute_features(
    history: pd.DataFrame,
    static: Dict[str, float],
    mins_since_meal: float,
    meal_carbs: float,
) -> pd.DataFrame:
    """
    Builds the single-row feature vector from patient history and static demographics.
    This function isolates feature preparation so it can be swapped with shared
    training feature pipeline code later.

    Parameters:
    - history: DataFrame containing the 5-minute CGM readings.
    - static: Dictionary containing demographic/clinical variables.
    - mins_since_meal: Minutes elapsed since the most recent meal.
    - meal_carbs: Total grams of carbohydrates in the most recent meal.

    Returns:
    - pd.DataFrame with exactly 1 row and columns matching FEATURE_COLUMNS.
    """
    glu_now = float(history["glucose"].iloc[-1])

    # Glucose 30 minutes ago (6 steps back: 6 * 5 min = 30 min)
    if len(history) >= 7:
        glu_30_ago = float(history["glucose"].iloc[-7])
    else:
        glu_30_ago = glu_now
    glu_change_30 = float(glu_now - glu_30_ago)

    row = {
        "glu_now": glu_now,
        "glu_change_30": glu_change_30,
        "mins_since_meal": float(mins_since_meal),
        "meal_carbs": float(meal_carbs),
        "Age": float(static.get("Age", 50.0)),
        "Gender": float(static.get("Gender", 1.0)),
        "BMI": float(static.get("BMI", 28.0)),
        "A1c": float(static.get("A1c", 7.0)),
        "fasting_glu": float(static.get("fasting_glu", 120.0)),
        "Insulin": float(static.get("Insulin", 15.0)),
        "HDL": float(static.get("HDL", 45.0)),
        "Triglycerides": float(static.get("Triglycerides", 150.0)),
    }

    # Ensure exact column order as specified in MODEL_CARD.md
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


def predict_risk(feature_row: pd.DataFrame) -> float:
    """
    Predicts the probability (0.0 to 1.0) that glucose will reach >= 180 mg/dL
    within the next 2 hours.

    Currently returns a PLACEHOLDER risk calculated using a physiologically grounded
    heuristic rule. When the trained model artifact is ready, replace this with the
    real model call indicated in the TODO below.
    """
        # Use the trained model whenever it is available and the question makes sense.
    # The model was trained only on rows where glucose is below the threshold.
    artifact = _load_artifact()
    glu_now_check = float(feature_row["glu_now"].iloc[0])
    if artifact is not None and glu_now_check < artifact["threshold"]:
        X = feature_row[artifact["features"]]   # same columns, same order as in training
        return round(float(artifact["model"].predict_proba(X)[0, 1]), 3)
    
    # --- PLACEHOLDER HEURISTIC RULE ---
    # Higher baseline glucose, rapid recent rise, high carbs, and recent meals
    # lead to a higher probability of breaching 180 mg/dL.
    glu_now = float(feature_row["glu_now"].iloc[0])
    glu_change_30 = float(feature_row["glu_change_30"].iloc[0])
    mins_since_meal = float(feature_row["mins_since_meal"].iloc[0])
    meal_carbs = float(feature_row["meal_carbs"].iloc[0])
    a1c = float(feature_row["A1c"].iloc[0])

    # If current glucose is already >= 180 mg/dL, breach has already occurred
    if glu_now >= 180.0:
        return 0.95

    # 1. Glucose proximity score (distance to 180 threshold)
    proximity_score = (glu_now - 100.0) / 80.0  # 0 at 100 mg/dL, 1.0 at 180 mg/dL

    # 2. Rate of rise score
    trend_score = glu_change_30 / 30.0  # +30 mg/dL in 30 min adds +1.0

    # 3. Postprandial excursion remaining
    # Meals within 15-75 min are peaking; meals > 150 min have largely cleared
    if mins_since_meal <= 90:
        time_factor = (90.0 - mins_since_meal) / 90.0
    else:
        time_factor = -min((mins_since_meal - 90.0) / 120.0, 0.6)

    carb_score = (meal_carbs / 70.0) * (0.8 + 0.5 * time_factor)

    # 4. Metabolic resistance (A1c baseline)
    a1c_score = (a1c - 6.5) * 0.25

    # Weighted linear combination
    z = (
        -2.2
        + (1.8 * proximity_score)
        + (1.0 * trend_score)
        + (1.1 * carb_score)
        + (0.6 * a1c_score)
    )

    # Sigmoid function to map to a well-behaved probability [0.02, 0.95]
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = float(np.clip(prob, 0.02, 0.95))

    return round(prob, 3)


def get_risk_badge(risk: float) -> Tuple[str, str, str]:
    """
    Returns the classification label, hex color, and badge text based on MODEL_CARD:
    - Green: risk < 0.2
    - Yellow: 0.2 <= risk < 0.4
    - Red: risk >= 0.4
    """
    if risk < 0.2:
        return "Low Risk", "#28a745", "🟢 Low Risk (< 0.20)"
    elif risk < 0.4:
        return "Moderate Risk", "#d39e00", "🟡 Moderate Risk (0.20 – 0.40)"
    else:
        return "High Risk", "#dc3545", "🔴 High Risk (≥ 0.40)"


def get_feature_contributions(feature_row: pd.DataFrame) -> pd.DataFrame:
    """
    Generates placeholder feature importance/contributions for the 'Why' panel.
    PLACEHOLDER: In the future, this will be calculated using TreeSHAP on the LightGBM model.
    """
    glu_now = float(feature_row["glu_now"].iloc[0])
    glu_change_30 = float(feature_row["glu_change_30"].iloc[0])
    mins_since_meal = float(feature_row["mins_since_meal"].iloc[0])
    meal_carbs = float(feature_row["meal_carbs"].iloc[0])
    a1c = float(feature_row["A1c"].iloc[0])
    bmi = float(feature_row["BMI"].iloc[0])

    # Relative directional contributions (positive = increases risk of >=180)
    contributions = [
        {"Feature": "Current Glucose (glu_now)", "Impact": round((glu_now - 120.0) * 0.005, 3)},
        {"Feature": "Recent Meal Carbs (meal_carbs)", "Impact": round((meal_carbs - 45.0) * 0.006, 3)},
        {"Feature": "30-min Trend (glu_change_30)", "Impact": round(glu_change_30 * 0.008, 3)},
        {"Feature": "Time Since Meal (mins_since_meal)", "Impact": round((90.0 - mins_since_meal) * 0.002, 3)},
        {"Feature": "Baseline A1c", "Impact": round((a1c - 7.0) * 0.04, 3)},
        {"Feature": "Body Mass Index (BMI)", "Impact": round((bmi - 28.0) * 0.01, 3)},
    ]

    df = pd.DataFrame(contributions)
    # Sort by absolute impact descending
    df["AbsImpact"] = df["Impact"].abs()
    df = df.sort_values(by="AbsImpact", ascending=True).reset_index(drop=True)
    return df
