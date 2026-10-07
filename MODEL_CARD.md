MODEL CARD (update after every change)

Task: predict, from past data only, whether a patient's glucose will reach
>= 180 mg/dL within the next 120 minutes. Rows are every 5 minutes, only
rows where current glucose < 180 are used.
Data: CGMacros (PhysioNet, CC BY-NC-SA 4.0), 45 participants.
Never commit dataset rows or derived dataset copies to the repo.

Features (order matters, stored in models/glucose_model.joblib):
- glu_now, glu_change_30: current glucose and its change over the last 30 min
- mins_since_meal, meal_carbs: time since and carbs of the last logged meal
- static: Age, Gender (M=1), BMI, A1c, fasting_glu, Insulin, HDL, Triglycerides

Model: LightGBM (200 trees, lr 0.05, min_child_samples=300). meal_carbs is
constrained so more carbs never lowers risk. Final model trained on all 45 participants.

Evaluation: 5-fold CV split by participant. Mean fold AUC about 0.78,
pooled out-of-fold AUC 0.776.
  glucose only: 0.701 | glucose + meal: 0.766 | + all static: 0.778
  static features alone: 0.650. Static features gave a small, inconsistent gain.
Row level, p >= 0.3: precision 0.44, recall 0.45, false alarm 0.11 (base rate 0.157).
  p >= 0.2: precision 0.37, recall 0.56, false alarm 0.18.
  p >= 0.5: precision 0.56, recall 0.28, false alarm 0.04.
Calibration: scores above 0.4 are over-confident by about 10-15 points
  (score 0.87 -> 75.7% actually reached 180). Table: models/calibration_table.json.

Limitations: 45 participants, US data, no Indian population. Moderate signal only
  (proof of concept). Heart rate and activity are not used, so the what-if has no
  activity slider. The what-if changes only meal inputs and keeps glucose history fixed:
  it shows the model's sensitivity, not what would really happen to the patient.

Rules: English comments in code, doctor-facing disclaimer on the dashboard,
risk shown as 3 levels (green < 0.2, yellow 0.2-0.4, red >= 0.4).