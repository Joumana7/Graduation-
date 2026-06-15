# Model Evaluation Report
## Explainable AI Learning Analytics Platform — Zewail City

---

## Target 1: GPA Prediction (Regression)

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| linear_regression | 0.9864 | 0.0767 | 0.1326 |
| xgboost | 0.9989 | 0.0288 | 0.0381 |
| lightgbm | 0.9990 | 0.0273 | 0.0360 |

**Production Model: XGBoost**
- R²: 0.9989 (explains 99.9% of GPA variance)
- MAE: 0.0288 GPA points average error
- RMSE: 0.0381

### Top Feature Importances (GPA Model)
| Feature | Importance |
|---------|-----------|
| avg_overall | 0.5264 |
| failed_course_ratio | 0.1637 |
| avg_final | 0.1180 |
| avg_midterm | 0.0966 |
| credit_completion_ratio | 0.0690 |
| failed_courses | 0.0123 |
| avg_labs | 0.0090 |
| avg_assignments | 0.0032 |
| avg_quizzes | 0.0008 |
| credits_passed | 0.0002 |

---

## Target 2: Risk Classification (Low / Medium / High)

| Model | Accuracy | F1 (weighted) | AUC |
|-------|----------|---------------|-----|
| logistic_regression | 0.9733 | 0.9734 | N/A |
| xgboost | 0.9887 | 0.9888 | 0.9995 |
| lightgbm | 0.9887 | 0.9888 | 0.9996 |

**Production Model: XGBoost**

### Classification Report (XGBoost)
```
              precision    recall  f1-score   support

    Low Risk       1.00      0.99      1.00       640
 Medium Risk       0.94      0.99      0.96       226
   High Risk       1.00      0.99      0.99       634

    accuracy                           0.99      1500
   macro avg       0.98      0.99      0.98      1500
weighted avg       0.99      0.99      0.99      1500

```

### Confusion Matrix (XGBoost)
```
Predicted: Low   Med   High
Low Risk :   635     5     0
Med Risk :     0   223     3
High Risk:     0     9   625
```

### Top Feature Importances (Risk Model)
| Feature | Importance |
|---------|-----------|
| avg_overall | 0.3954 |
| failed_courses | 0.1062 |
| avg_final | 0.0955 |
| avg_midterm | 0.0852 |
| avg_attendance | 0.0654 |
| credit_completion_ratio | 0.0602 |
| failed_course_ratio | 0.0461 |
| attendance_risk_score | 0.0450 |
| avg_labs | 0.0288 |
| avg_quizzes | 0.0188 |

---

## Conclusion

Both XGBoost models achieve strong performance and are selected as production models.
SHAP explainability is applied in Phase 6.
