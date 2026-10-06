"""
demo_data.py
Generates realistic synthetic patient profiles and 24-hour Continuous Glucose
Monitoring (CGM) time-series data for the Type 2 Diabetes Digital Twin prototype.

IMPORTANT: All data generated here is 100% synthetic. No real clinical dataset
or participant records from the 'data/' folder are used.
"""

from typing import Dict, List, Any
import zlib
import numpy as np
import pandas as pd


# 9 Diverse synthetic patient profiles representing various risk tiers
SYNTHETIC_PATIENTS: List[Dict[str, Any]] = [
    # Tier 1: High Risk (Red >= 0.40) — Recent high-carb meal, elevated baseline, rising trajectory
    {
        "id": "PT-101",
        "name": "Arthur Pendelton",
        "age": 64,
        "gender": 1,  # 1 = Male, 0 = Female
        "bmi": 32.4,
        "a1c": 8.6,
        "fasting_glu": 125.0,
        "insulin": 22.0,
        "hdl": 38.0,
        "triglycerides": 210.0,
        "dinner_carbs": 85.0,
        "mins_since_dinner": 35,
        "breakfast_carbs": 50.0,
        "lunch_carbs": 70.0,
    },
    {
        "id": "PT-102",
        "name": "Brenda Walker",
        "age": 69,
        "gender": 0,
        "bmi": 34.1,
        "a1c": 8.4,
        "fasting_glu": 122.0,
        "insulin": 20.0,
        "hdl": 35.0,
        "triglycerides": 220.0,
        "dinner_carbs": 80.0,
        "mins_since_dinner": 45,
        "breakfast_carbs": 45.0,
        "lunch_carbs": 65.0,
    },
    {
        "id": "PT-103",
        "name": "Jamal Washington",
        "age": 55,
        "gender": 1,
        "bmi": 31.0,
        "a1c": 7.9,
        "fasting_glu": 118.0,
        "insulin": 18.0,
        "hdl": 40.0,
        "triglycerides": 195.0,
        "dinner_carbs": 75.0,
        "mins_since_dinner": 40,
        "breakfast_carbs": 40.0,
        "lunch_carbs": 60.0,
    },

    # Tier 2: Moderate Risk (Yellow 0.20 - 0.40) — Moderate carbs, 1-1.5 hrs ago, plateauing
    {
        "id": "PT-104",
        "name": "Maria Gonzalez",
        "age": 58,
        "gender": 0,
        "bmi": 28.5,
        "a1c": 7.3,
        "fasting_glu": 112.0,
        "insulin": 15.0,
        "hdl": 44.0,
        "triglycerides": 165.0,
        "dinner_carbs": 55.0,
        "mins_since_dinner": 75,
        "breakfast_carbs": 35.0,
        "lunch_carbs": 55.0,
    },
    {
        "id": "PT-105",
        "name": "Robert Taylor",
        "age": 60,
        "gender": 1,
        "bmi": 27.8,
        "a1c": 7.1,
        "fasting_glu": 110.0,
        "insulin": 14.0,
        "hdl": 46.0,
        "triglycerides": 150.0,
        "dinner_carbs": 50.0,
        "mins_since_dinner": 80,
        "breakfast_carbs": 40.0,
        "lunch_carbs": 50.0,
    },
    {
        "id": "PT-106",
        "name": "Thomas Mueller",
        "age": 72,
        "gender": 1,
        "bmi": 29.2,
        "a1c": 7.4,
        "fasting_glu": 115.0,
        "insulin": 16.0,
        "hdl": 42.0,
        "triglycerides": 170.0,
        "dinner_carbs": 48.0,
        "mins_since_dinner": 95,
        "breakfast_carbs": 45.0,
        "lunch_carbs": 55.0,
    },

    # Tier 3: Low Risk (Green < 0.20) — Well controlled, low-to-mod carbs 2+ hours ago, steady/declining
    {
        "id": "PT-107",
        "name": "David Kim",
        "age": 51,
        "gender": 1,
        "bmi": 25.5,
        "a1c": 6.4,
        "fasting_glu": 98.0,
        "insulin": 10.0,
        "hdl": 52.0,
        "triglycerides": 120.0,
        "dinner_carbs": 35.0,
        "mins_since_dinner": 130,
        "breakfast_carbs": 30.0,
        "lunch_carbs": 45.0,
    },
    {
        "id": "PT-108",
        "name": "Grace O'Connor",
        "age": 47,
        "gender": 0,
        "bmi": 24.2,
        "a1c": 6.0,
        "fasting_glu": 94.0,
        "insulin": 8.5,
        "hdl": 58.0,
        "triglycerides": 110.0,
        "dinner_carbs": 30.0,
        "mins_since_dinner": 150,
        "breakfast_carbs": 25.0,
        "lunch_carbs": 40.0,
    },
    {
        "id": "PT-109",
        "name": "Sunita Patel",
        "age": 62,
        "gender": 0,
        "bmi": 26.0,
        "a1c": 6.6,
        "fasting_glu": 102.0,
        "insulin": 11.0,
        "hdl": 50.0,
        "triglycerides": 135.0,
        "dinner_carbs": 38.0,
        "mins_since_dinner": 125,
        "breakfast_carbs": 35.0,
        "lunch_carbs": 45.0,
    },
]


def generate_patient_cgm(patient_meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulates 24 hours of 5-minute CGM readings (288 points) ending at the current time (21:30).
    Includes 3 meals (breakfast, lunch, dinner) and physiologically grounded
    postprandial glucose curves where current glucose remains < 180 mg/dL.
    """
    total_steps = 288  # 24 hours * 12 readings/hour
    step_minutes = 5
   
    rng = np.random.default_rng(seed=zlib.crc32(patient_meta["id"].encode()))   # stable across runs

    # Fasting glucose baseline
    base_glu = patient_meta["fasting_glu"]

    # Minutes ago from 24h ago (1435 min) down to 0 min (now)
    minutes_ago = np.arange(total_steps - 1, -1, -1) * step_minutes

    mins_since_dinner = patient_meta["mins_since_dinner"]
    dinner_carbs = patient_meta["dinner_carbs"]

    meals = [
        {"minutes_ago": 13 * 60, "carbs": patient_meta["breakfast_carbs"], "name": f"Breakfast ({int(patient_meta['breakfast_carbs'])}g)"},
        {"minutes_ago": int(7.5 * 60), "carbs": patient_meta["lunch_carbs"], "name": f"Lunch ({int(patient_meta['lunch_carbs'])}g)"},
        {"minutes_ago": mins_since_dinner, "carbs": dinner_carbs, "name": f"Dinner ({int(dinner_carbs)}g)"},
    ]

    # Start from baseline fasting glucose
    glucose_profile = np.full(total_steps, base_glu, dtype=float)

    meal_carbs_series = np.zeros(total_steps)
    meal_name_series = [""] * total_steps

    for meal in meals:
        t_meal = meal["minutes_ago"]
        carbs = meal["carbs"]

        # Tag the meal marker at the closest 5-minute interval
        meal_idx = int(np.argmin(np.abs(minutes_ago - t_meal)))
        meal_carbs_series[meal_idx] = carbs
        meal_name_series[meal_idx] = meal["name"]

        # Time delta in minutes from meal point onward
        dt = t_meal - minutes_ago
        mask = dt >= 0

        # Postprandial excursion calculation:
        # Excursion height scales with meal carbs and insulin resistance (A1c)
        peak_rise = (carbs * 0.55) * (0.85 + (patient_meta["a1c"] - 6.0) * 0.08)
        tau = 55.0  # absorption & clearance characteristic time (~55 min peak)
        spike = np.zeros(total_steps)
        spike[mask] = peak_rise * (dt[mask] / tau) * np.exp(1.0 - (dt[mask] / tau))
        glucose_profile += spike

    # Add realistic smooth noise (random walk with mean reversion)
    noise = np.zeros(total_steps)
    curr_noise = 0.0
    for i in range(total_steps):
        curr_noise = 0.88 * curr_noise + rng.normal(0, 1.1)
        noise[i] = curr_noise
    glucose_profile += noise

    # Clip to realistic physiological limits and round
    glucose_profile = np.clip(glucose_profile, 70.0, 175.0)
    glucose_profile = np.round(glucose_profile, 1)

    # Format time strings: assume current time is 21:30 (9:30 PM)
    timestamps = []
    base_hour = 21
    base_minute = 30
    current_total_min = base_hour * 60 + base_minute

    for m in minutes_ago:
        point_total_min = (current_total_min - m) % (24 * 60)
        hh = int(point_total_min // 60)
        mm = int(point_total_min % 60)
        timestamps.append(f"{hh:02d}:{mm:02d}")

    df = pd.DataFrame({
        "time_str": timestamps,
        "minutes_ago": minutes_ago,
        "glucose": glucose_profile,
        "meal_carbs": meal_carbs_series,
        "meal_name": meal_name_series,
        "is_meal": meal_carbs_series > 0,
    })

    return df


def get_all_patients() -> List[Dict[str, Any]]:
    """
    Builds the full synthetic patient records with static details, history,
    and dynamically extracted current features for each patient.
    """
    patients = []
    for meta in SYNTHETIC_PATIENTS:
        cgm_df = generate_patient_cgm(meta)
        
        # Current glucose (last row)
        glu_now = float(cgm_df["glucose"].iloc[-1])
        # Glucose 30 minutes ago (6 steps back: 6 * 5 min = 30 min)
        glu_30_ago = float(cgm_df["glucose"].iloc[-7])
        glu_change_30 = round(glu_now - glu_30_ago, 1)

        static_features = {
            "Age": float(meta["age"]),
            "Gender": float(meta["gender"]),
            "BMI": float(meta["bmi"]),
            "A1c": float(meta["a1c"]),
            "fasting_glu": float(meta["fasting_glu"]),
            "Insulin": float(meta["insulin"]),
            "HDL": float(meta["hdl"]),
            "Triglycerides": float(meta["triglycerides"]),
        }

        patient_record = {
            "id": meta["id"],
            "name": meta["name"],
            "static": static_features,
            "history": cgm_df,
            "mins_since_meal": float(meta["mins_since_dinner"]),
            "meal_carbs": float(meta["dinner_carbs"]),
            "glu_now": glu_now,
            "glu_change_30": glu_change_30,
        }
        patients.append(patient_record)

    return patients
