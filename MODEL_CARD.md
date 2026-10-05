MODEL CARD (update after every change)

Task: predict, from past data only, whether a patient's glucose will reach
>= 180 mg/dL within the next 120 minutes. Rows are every 5 minutes, only
rows where current glucose < 180 are used.
Data: CGMacros (PhysioNet, CC BY-NC-SA 4.0), 45 participants.
Never commit dataset rows or derived dataset copies to the repo.

Features (order matters, stored in models/glucose_model.joblib):
- glu_now: current Dexcom glucose (mg/dL)
- glu_change_30: glucose now minus glucose 30 min ago
- mins_since_meal, meal_carbs: time since and carbs of the last logged meal
- static: Age, Gender (M=1), BMI, A1c, fasting_glu, Insulin, HDL, Triglycerides

Model: LightGBM (200 trees, lr 0.05), final model trained on all 45 participants.
Evaluation: 5-fold CV split by participant (no participant in both train and test).
  glucose only: AUC 0.701 | + meal: 0.766 | + meal + static: 0.778
  At p >= 0.3: precision 0.43, recall 0.46, false alarm rate 0.12 (row level)
Limitations: 45 participants, US data, static features gave no consistent gain,
  moderate signal only (proof of concept), what-if is a model estimate, not medical advice.
  The model does not use heart rate or activity, so the what-if has no activity slider.

Rules: English comments in code, doctor-facing disclaimer on the dashboard,
risk shown as 3 levels (green < 0.2, yellow 0.2-0.4, red >= 0.4).