# MODEL CARD

**Task:** Predict whether a patient's glucose will reach ≥180 mg/dL within the next 120 minutes, using past data and current available inputs. Rows are sampled every 5 minutes; only rows with current glucose <180 mg/dL are included.

**Data:** CGMacros (PhysioNet, CC BY-NC-SA 4.0). The final experiment excludes participant 7 because gaps of up to 23 minutes make row-based time windows unreliable. The final training table contains 44 participants.

Never commit dataset rows or derived dataset copies to the repository.

## Features

Feature order is stored in `models/glucose_model.joblib`.

- `glu_now`, `glu_change_30`: current glucose and its change over the previous 30 minutes.
- `mins_since_meal`, `meal_carbs`: time since and carbohydrates in the last logged meal.
- Static features: `Age`, `Gender` (M=1), `BMI`, `A1c`, `fasting_glu`, `Insulin`, `HDL`, `Triglycerides`.

## Model

LightGBM classifier: 200 trees, learning rate 0.05, `min_child_samples=300`.

`meal_carbs` has a positive monotonic constraint, so increasing this feature cannot decrease the model's predicted risk when the other features are held constant.

The final notebook experiment trains on all 44 remaining participants.

## Evaluation

Five-fold cross-validation split by participant. Latest notebook output:

- Pooled out-of-fold AUC: **0.769**
- Positive-event base rate: **0.158**

Out-of-fold alert metrics:

| Score threshold | Alert rate | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|---:|
| ≥0.2 | 0.25 | 0.36 | 0.56 | 0.19 |
| ≥0.3 | 0.17 | 0.42 | 0.45 | 0.11 |
| ≥0.4 | 0.11 | 0.49 | 0.36 | 0.07 |
| ≥0.5 | 0.08 | 0.56 | 0.28 | 0.04 |

These are row-level metrics. Adjacent five-minute readings from the same participant may be correlated; they should not be interpreted as independent patient-level outcomes.

The previous experiment's feature-group AUC comparisons are not included here because their equivalence to the latest 44-participant experiment has not been verified.

## Calibration

Calibration values are derived from out-of-fold predictions and saved in `models/calibration_table.json`.

| Predicted-score range | Rows | Observed event rate |
|---|---:|---:|
| ≤0.1 | 58,462 | 6.9% |
| 0.1–0.2 | 21,300 | 15.6% |
| 0.2–0.3 | 8,719 | 22.0% |
| 0.3–0.4 | 5,520 | 26.8% |
| 0.4–0.5 | 3,783 | 36.0% |
| 0.5–0.6 | 2,934 | 40.4% |
| 0.6–0.8 | 3,699 | 58.3% |
| 0.8–1.0 | 1,750 | 75.1% |

Scores above 0.4 tend to overestimate the observed event rate in these bins. For example, the highest bin has a mean predicted score of 0.865 and an observed event rate of 75.1%. The score is not a guarantee of an individual patient's outcome.

## Limitations

- The final experiment includes 44 participants from the available dataset; the data are not representative of all populations, and performance in an Indian population has not been established.
- This is a proof-of-concept model, not a clinically validated diagnostic or treatment tool.
- Heart rate and activity are not model inputs.
- The what-if interface changes meal inputs while keeping glucose history fixed. It illustrates model sensitivity, not a causal effect or a prediction of what would actually happen to a patient.
- Row-level evaluation may be affected by correlation between readings from the same participant.

## Dashboard rules

- Include a clear doctor-facing disclaimer.
- Display three score bands: green (<0.2), yellow (0.2–<0.4), and red (≥0.4).
- Keep code comments in English.
- Do not commit dataset rows or derived dataset copies.
