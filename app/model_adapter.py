
"""Interface between the dashboard and the trained glucose-risk model."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "glucose_model.joblib"
CALIBRATION_PATH = BASE_DIR / "models" / "calibration_table.json"

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


@lru_cache(maxsize=1)
def _load_artifact():
    """Load the saved model once. Raise a useful error if loading fails."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Model artifact must contain a 'model' entry.")

    if "features" not in artifact:
        raise ValueError("Model artifact is missing its feature list.")

    if not hasattr(artifact["model"], "predict_proba"):
        raise TypeError("Saved model does not support predict_proba().")

    return artifact


def compute_features(
    history: pd.DataFrame,
    static: Dict[str, float],
    mins_since_meal: float,
    meal_carbs: float,
) -> pd.DataFrame:
    """Build one model input row from glucose history and patient inputs."""
    if history.empty or "glucose" not in history.columns:
        raise ValueError("History must contain at least one glucose reading.")

    glu_now = float(history["glucose"].iloc[-1])
    glu_30_ago = (
        float(history["glucose"].iloc[-7])
        if len(history) >= 7
        else glu_now
    )

    row = {
        "glu_now": glu_now,
        "glu_change_30": glu_now - glu_30_ago,
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

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_risk(feature_row: pd.DataFrame) -> float:
    """Return the trained model's estimated probability for the target event."""
    artifact = _load_artifact()
    features = artifact["features"]
    threshold = float(artifact.get("threshold", 180.0))

    missing = [column for column in features if column not in feature_row.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = feature_row.loc[:, features].copy()

    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Model input must contain finite numeric values.")

    # The target is a future threshold event. If the threshold has
    # already been reached, the dashboard should label that separately.
    if float(feature_row["glu_now"].iloc[0]) >= threshold:
        return 1.0

    probability = float(artifact["model"].predict_proba(X)[0, 1])
    return float(np.clip(probability, 0.0, 1.0))


def get_risk_badge(risk: float) -> Tuple[str, str, str]:
    """Return risk category, colour, and readable badge text."""
    if risk < 0.2:
        return "Low Risk", "#28a745", "🟢 Low Risk (< 0.20)"
    if risk < 0.4:
        return "Moderate Risk", "#d39e00", "🟡 Moderate Risk (0.20–0.40)"
    return "High Risk", "#dc3545", "🔴 High Risk (≥ 0.40)"


@lru_cache(maxsize=1)
def _load_calibration():
    """Load the observed-rate calibration bins from the saved JSON file."""
    if not CALIBRATION_PATH.exists():
        return None

    import json

    with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
        bins = json.load(file)

    if not isinstance(bins, list):
        raise ValueError("Calibration file must contain a list of bins.")

    return bins


def validation_rate(score: float):
    """
    Return (observed event rate, row count) for the matching score bin.
    Return None when no calibration bin is available.
    """
    bins = _load_calibration()
    if not bins:
        return None

    score = float(score)


    for index, item in enumerate(bins):
        low = float(item["low"])
        high = float(item["high"])

        # Match pandas pd.cut bins: (low, high]
        matches = low < score <= high

        if matches:
            return float(item["actual_rate"]), int(item["rows"])

    return None


@lru_cache(maxsize=1)
def _get_explainer():
    """Create a TreeSHAP explainer for the loaded model."""
    import shap

    return shap.TreeExplainer(_load_artifact()["model"])


def get_feature_contributions(feature_row: pd.DataFrame) -> pd.DataFrame:
    """
    Return genuine TreeSHAP contributions when SHAP is available.
    Impact values are on the model's raw output scale, not percentages.
    An empty DataFrame means explanations are unavailable.
    """
    try:
        artifact = _load_artifact()
        features = artifact["features"]
        X = feature_row.loc[:, features].copy()

        # Newer SHAP versions may return an Explanation object;
        # older versions may return a list of arrays.
        explanation = _get_explainer()(X)
        values = explanation.values

        values = np.asarray(values)

        # Binary classifiers can return (rows, features, classes).
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        elif values.ndim != 1:
            return pd.DataFrame(columns=["Feature", "Impact", "AbsImpact"])


        if len(values) != len(features):
            return pd.DataFrame(columns=["Feature", "Impact", "AbsImpact"])

        labels = {
            "glu_now": "Current Glucose",
            "glu_change_30": "30-min Glucose Trend",
            "mins_since_meal": "Time Since Meal",
            "meal_carbs": "Meal Carbohydrates",
            "Age": "Age",
            "Gender": "Gender",
            "BMI": "Body Mass Index (BMI)",
            "A1c": "Baseline A1c",
            "fasting_glu": "Fasting Glucose",
            "Insulin": "Insulin",
            "HDL": "HDL",
            "Triglycerides": "Triglycerides",
        }

        result = pd.DataFrame({
            "Feature": [labels.get(name, name) for name in features],
            "Impact": values.astype(float),
        })
        result["AbsImpact"] = result["Impact"].abs()
        return result.sort_values(
            "AbsImpact", ascending=True
        ).reset_index(drop=True)

    except (ImportError, ModuleNotFoundError, ValueError, TypeError):
        # Do not substitute made-up contributions if SHAP is unavailable.
        return pd.DataFrame(columns=["Feature", "Impact", "AbsImpact"])
