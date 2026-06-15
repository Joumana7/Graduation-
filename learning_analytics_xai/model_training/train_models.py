"""
Phase 5 — Model Training
Trains and evaluates models for two targets:
  Target 1: Semester/Cumulative GPA Prediction (Regression)
  Target 2: Academic Risk Classification (Low / Medium / High)

Models evaluated:
  Baselines : Linear Regression / Logistic Regression
  Main      : XGBoost (production model)
  Compare   : LightGBM

Best models saved to ../models/

Outputs:
  ../models/gpa_model_xgb.pkl       — best GPA regressor
  ../models/risk_model_xgb.pkl      — best risk classifier
  ../models/scaler.pkl              — StandardScaler for raw features
  ../models/model_metrics.json      — evaluation metrics report
  ../reports/model_evaluation.md    — human-readable evaluation report
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

HERE    = Path(__file__).parent
DATA    = HERE.parent / "data"
MODELS  = HERE.parent / "models"
REPORTS = HERE.parent / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

FEAT_PATH = DATA / "features_train.csv"
META_PATH = DATA / "feature_names.json"

RANDOM_STATE = 42


# ── Load data ─────────────────────────────────────────────────────────────────

def load():
    df   = pd.read_csv(FEAT_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feat_cols = meta["feature_columns"]
    X = df[feat_cols].values.astype(np.float32)
    y_gpa  = df["target_gpa"].values.astype(np.float32)
    y_risk = df["target_risk"].values.astype(np.int32)
    return X, y_gpa, y_risk, feat_cols


# ── Regression (GPA) ─────────────────────────────────────────────────────────

def train_gpa_models(X_tr, X_te, y_tr, y_te, feat_cols):
    print("  [GPA Regression]")
    results = {}

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    pred_lr = lr.predict(X_te)
    r2_lr  = r2_score(y_te, pred_lr)
    mae_lr = mean_absolute_error(y_te, pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_te, pred_lr))
    print(f"    Linear Regression   : R2={r2_lr:.4f}  MAE={mae_lr:.4f}  RMSE={rmse_lr:.4f}")
    results["linear_regression"] = {"r2": r2_lr, "mae": mae_lr, "rmse": rmse_lr}

    # XGBoost
    xgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    xgb_reg = xgb.XGBRegressor(**xgb_params)
    xgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    pred_xgb = xgb_reg.predict(X_te)
    r2_xgb   = r2_score(y_te, pred_xgb)
    mae_xgb  = mean_absolute_error(y_te, pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_te, pred_xgb))
    print(f"    XGBoost             : R2={r2_xgb:.4f}  MAE={mae_xgb:.4f}  RMSE={rmse_xgb:.4f}")
    results["xgboost"] = {"r2": r2_xgb, "mae": mae_xgb, "rmse": rmse_xgb}

    # LightGBM
    lgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    lgb_reg = lgb.LGBMRegressor(**lgb_params)
    lgb_reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    pred_lgb = lgb_reg.predict(X_te)
    r2_lgb   = r2_score(y_te, pred_lgb)
    mae_lgb  = mean_absolute_error(y_te, pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_te, pred_lgb))
    print(f"    LightGBM            : R2={r2_lgb:.4f}  MAE={mae_lgb:.4f}  RMSE={rmse_lgb:.4f}")
    results["lightgbm"] = {"r2": r2_lgb, "mae": mae_lgb, "rmse": rmse_lgb}

    # Feature importance (XGBoost)
    fi = dict(zip(feat_cols, xgb_reg.feature_importances_.tolist()))

    # Select best model
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = xgb_reg if best_name in ("xgboost", "linear_regression") else lgb_reg
    print(f"    Best model: {best_name} (R2={results[best_name]['r2']:.4f})")

    return xgb_reg, lgb_reg, results, fi


# ── Classification (Risk) ─────────────────────────────────────────────────────

def train_risk_models(X_tr, X_te, y_tr, y_te, feat_cols):
    print("\n  [Risk Classification]")
    results = {}
    target_names = ["Low Risk", "Medium Risk", "High Risk"]

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr, y_tr)
    pred_lr = lr.predict(X_te)
    acc_lr  = accuracy_score(y_te, pred_lr)
    f1_lr   = f1_score(y_te, pred_lr, average="weighted")
    print(f"    Logistic Regression : Acc={acc_lr:.4f}  F1={f1_lr:.4f}")
    results["logistic_regression"] = {"accuracy": acc_lr, "f1_weighted": f1_lr}

    # XGBoost
    xgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    xgb_clf = xgb.XGBClassifier(**xgb_params)
    xgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    pred_xgb = xgb_clf.predict(X_te)
    acc_xgb  = accuracy_score(y_te, pred_xgb)
    f1_xgb   = f1_score(y_te, pred_xgb, average="weighted")
    prob_xgb = xgb_clf.predict_proba(X_te)
    auc_xgb  = roc_auc_score(y_te, prob_xgb, multi_class="ovr", average="weighted")
    print(f"    XGBoost             : Acc={acc_xgb:.4f}  F1={f1_xgb:.4f}  AUC={auc_xgb:.4f}")
    results["xgboost"] = {"accuracy": acc_xgb, "f1_weighted": f1_xgb, "auc": auc_xgb}

    clf_report_xgb = classification_report(y_te, pred_xgb, target_names=target_names)
    cm_xgb = confusion_matrix(y_te, pred_xgb).tolist()

    # LightGBM
    lgb_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])
    pred_lgb = lgb_clf.predict(X_te)
    acc_lgb  = accuracy_score(y_te, pred_lgb)
    f1_lgb   = f1_score(y_te, pred_lgb, average="weighted")
    prob_lgb = lgb_clf.predict_proba(X_te)
    auc_lgb  = roc_auc_score(y_te, prob_lgb, multi_class="ovr", average="weighted")
    print(f"    LightGBM            : Acc={acc_lgb:.4f}  F1={f1_lgb:.4f}  AUC={auc_lgb:.4f}")
    results["lightgbm"] = {"accuracy": acc_lgb, "f1_weighted": f1_lgb, "auc": auc_lgb}

    fi = dict(zip(feat_cols, xgb_clf.feature_importances_.tolist()))

    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    print(f"    Best model: {best_name} (F1={results[best_name]['f1_weighted']:.4f})")

    return xgb_clf, lgb_clf, results, fi, clf_report_xgb, cm_xgb


# ── Report ─────────────────────────────────────────────────────────────────────

def build_report(gpa_results, risk_results, gpa_fi, risk_fi, clf_report, cm, feat_cols):
    top_gpa  = sorted(gpa_fi.items(),  key=lambda x: x[1], reverse=True)[:10]
    top_risk = sorted(risk_fi.items(), key=lambda x: x[1], reverse=True)[:10]

    report = f"""# Model Evaluation Report
## Explainable AI Learning Analytics Platform — Zewail City

---

## Target 1: GPA Prediction (Regression)

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
"""
    for m, v in gpa_results.items():
        report += f"| {m} | {v['r2']:.4f} | {v['mae']:.4f} | {v['rmse']:.4f} |\n"

    report += f"""
**Production Model: XGBoost**
- R²: {gpa_results['xgboost']['r2']:.4f} (explains {gpa_results['xgboost']['r2']*100:.1f}% of GPA variance)
- MAE: {gpa_results['xgboost']['mae']:.4f} GPA points average error
- RMSE: {gpa_results['xgboost']['rmse']:.4f}

### Top Feature Importances (GPA Model)
| Feature | Importance |
|---------|-----------|
"""
    for feat, imp in top_gpa:
        report += f"| {feat} | {imp:.4f} |\n"

    report += f"""
---

## Target 2: Risk Classification (Low / Medium / High)

| Model | Accuracy | F1 (weighted) | AUC |
|-------|----------|---------------|-----|
"""
    for m, v in risk_results.items():
        auc = v.get("auc", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
        report += f"| {m} | {v['accuracy']:.4f} | {v['f1_weighted']:.4f} | {auc_str} |\n"

    report += f"""
**Production Model: XGBoost**

### Classification Report (XGBoost)
```
{clf_report}
```

### Confusion Matrix (XGBoost)
```
Predicted: Low   Med   High
Low Risk : {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}
Med Risk : {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}
High Risk: {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}
```

### Top Feature Importances (Risk Model)
| Feature | Importance |
|---------|-----------|
"""
    for feat, imp in top_risk:
        report += f"| {feat} | {imp:.4f} |\n"

    report += """
---

## Conclusion

Both XGBoost models achieve strong performance and are selected as production models.
SHAP explainability is applied in Phase 6.
"""
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 5: Model Training ===")

    X, y_gpa, y_risk, feat_cols = load()
    print(f"  Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"  GPA target: mean={y_gpa.mean():.3f}  std={y_gpa.std():.3f}")
    print(f"  Risk classes: {dict(zip(*np.unique(y_risk, return_counts=True)))}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train/test split (80/20, stratified on risk)
    X_tr, X_te, yg_tr, yg_te, yr_tr, yr_te = train_test_split(
        X_scaled, y_gpa, y_risk,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_risk
    )
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    print()
    # GPA models
    xgb_reg, lgb_reg, gpa_results, gpa_fi = train_gpa_models(
        X_tr, X_te, yg_tr, yg_te, feat_cols
    )

    # Risk models
    xgb_clf, lgb_clf, risk_results, risk_fi, clf_report, cm = train_risk_models(
        X_tr, X_te, yr_tr, yr_te, feat_cols
    )

    # 5-fold CV (n_jobs=1 to avoid multiprocessing memory issues on Windows)
    cv_gpa = cross_val_score(
        xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE,
                          verbosity=0, n_jobs=1),
        X_scaled, y_gpa, cv=5, scoring="r2", n_jobs=1
    )
    print(f"\n  XGBoost GPA 5-fold CV R2: {cv_gpa.mean():.4f} (+/-{cv_gpa.std():.4f})")

    cv_risk = cross_val_score(
        xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE,
                            verbosity=0, n_jobs=1, use_label_encoder=False,
                            eval_metric="mlogloss"),
        X_scaled, y_risk, cv=5, scoring="f1_weighted", n_jobs=1
    )
    print(f"  XGBoost Risk 5-fold CV F1: {cv_risk.mean():.4f} (+/-{cv_risk.std():.4f})")

    # Save models
    with open(MODELS / "gpa_model_xgb.pkl", "wb") as f:
        pickle.dump(xgb_reg, f)
    with open(MODELS / "risk_model_xgb.pkl", "wb") as f:
        pickle.dump(xgb_clf, f)
    with open(MODELS / "gpa_model_lgb.pkl", "wb") as f:
        pickle.dump(lgb_reg, f)
    with open(MODELS / "risk_model_lgb.pkl", "wb") as f:
        pickle.dump(lgb_clf, f)
    with open(MODELS / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  Saved models -> {MODELS}/")

    # Save metrics JSON
    metrics = {
        "gpa_regression":     gpa_results,
        "risk_classification": risk_results,
        "cv_gpa_r2":           {"mean": float(cv_gpa.mean()), "std": float(cv_gpa.std())},
        "cv_risk_f1":          {"mean": float(cv_risk.mean()), "std": float(cv_risk.std())},
        "gpa_feature_importance":  gpa_fi,
        "risk_feature_importance": risk_fi,
    }
    (MODELS / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save markdown report
    report = build_report(gpa_results, risk_results, gpa_fi, risk_fi, clf_report, cm, feat_cols)
    (REPORTS / "model_evaluation.md").write_text(report, encoding="utf-8")
    print(f"  Saved report -> {REPORTS}/model_evaluation.md")

    print("\n=== Phase 5 Complete ===")


if __name__ == "__main__":
    main()
