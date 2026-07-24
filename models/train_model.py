"""
Phase 2.1 — Approval Prediction Model Training
AI-Driven Prior Authorization Assistant (Diploma Project)

This script trains a machine learning model to predict whether an insurance claim
will be approved or rejected based on patient and claim features.

How it works:
  1. Loads the synthetic claims dataset (claims.csv)
  2. Encodes categorical features (drug names, insurance companies, diagnosis codes)
     into numerical values that ML models can process
  3. Trains two models — Random Forest and XGBoost — and picks the best one
  4. Evaluates performance using accuracy, precision, recall, F1 score, and
     a confusion matrix
  5. Saves the trained model + encoders for use by the Streamlit dashboard

Key ML concepts used (for diploma defense):
  - Binary Classification: predicting approved (1) vs. rejected (0)
  - Feature Engineering: converting text categories to numbers
  - Class Imbalance Handling: the dataset has ~65% approvals, so we use
    class_weight='balanced' to prevent the model from just predicting "approved"
  - Train/Test Split: 80% train, 20% test (stratified to maintain class ratio)
  - Feature Importance: shows which factors most influence approval decisions
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# XGBoost is a gradient boosting library — generally more accurate than
# Random Forest for structured/tabular data like insurance claims
import xgboost as xgb
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 2.1 — TRAINING APPROVAL PREDICTION MODEL")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))
print(f"\nLoaded {len(df)} claims from claims.csv")
print(f"Approval rate: {df['approved'].mean():.1%}")
print(f"Class distribution:\n{df['approved'].value_counts()}\n")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
# Select the features we'll use for prediction.
# These mirror what a pharmacist would know at the point of dispensing:
#   - patient_age: patient's age
#   - drug_name: the medication being prescribed
#   - insurance_name: which insurance company
#   - diagnosis_code: the ICD-10 code for the condition
#   - prior_drug_tried: whether the patient tried a required prior drug
#   - had_lab_results: whether lab results are available
#   - days_supply_requested: how many days of medication requested

FEATURE_COLS = [
    "patient_age",
    "drug_name",
    "insurance_name",
    "diagnosis_code",
    "prior_drug_tried",
    "had_lab_results",
    "days_supply_requested",
]
TARGET_COL = "approved"

# Handle missing/empty boolean columns — convert to numeric
# In the CSV, these may be empty strings for claims where the field
# doesn't apply (e.g., prior_drug_tried is empty when no step therapy required)
df["prior_drug_tried"] = df["prior_drug_tried"].apply(
    lambda x: 1 if x is True or x == "True" else (0 if x is False or x == "False" else -1)
)
df["had_lab_results"] = df["had_lab_results"].apply(
    lambda x: 1 if x is True or x == "True" else (0 if x is False or x == "False" else -1)
)

# Encode categorical columns using LabelEncoder
# LabelEncoder converts text labels like "CareFirst" into numbers like 0, 1, 2...
# We save these encoders so the dashboard can use the same encoding at prediction time
encoders = {}
categorical_cols = ["drug_name", "insurance_name", "diagnosis_code"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Build final feature matrix
feature_cols_final = [
    "patient_age",
    "drug_name_encoded",
    "insurance_name_encoded",
    "diagnosis_code_encoded",
    "prior_drug_tried",
    "had_lab_results",
    "days_supply_requested",
]

X = df[feature_cols_final].values
y = df[TARGET_COL].values

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
# stratify=y ensures both sets have the same approval/rejection ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

# ---------------------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------------------

# --- Model A: Random Forest ---
# Random Forest builds many decision trees and averages their predictions.
# class_weight='balanced' adjusts for the 65/35 imbalance automatically.
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=10,            # max depth per tree (prevents overfitting)
    class_weight="balanced", # handles class imbalance
    random_state=42,
    n_jobs=-1,               # use all CPU cores
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average="weighted")
print(f"  Accuracy: {rf_acc:.4f}")
print(f"  F1 Score: {rf_f1:.4f}")

# --- Model B: XGBoost ---
# XGBoost uses gradient boosting — it builds trees sequentially, each one
# correcting the errors of the previous one. Often more accurate than RF.
print("\n--- Training XGBoost ---")
# Calculate scale_pos_weight for class imbalance
n_negative = sum(y_train == 0)
n_positive = sum(y_train == 1)
scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",

)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds, average="weighted")
print(f"  Accuracy: {xgb_acc:.4f}")
print(f"  F1 Score: {xgb_f1:.4f}")

# ---------------------------------------------------------------------------
# 5. Select best model
# ---------------------------------------------------------------------------
if xgb_f1 >= rf_f1:
    best_model = xgb_model
    best_name = "XGBoost"
    best_acc = xgb_acc
    best_f1 = xgb_f1
    best_preds = xgb_preds
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_acc = rf_acc
    best_f1 = rf_f1
    best_preds = rf_preds

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_name}")
print(f"  Accuracy: {best_acc:.4f}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"{'=' * 60}")

# ---------------------------------------------------------------------------
# 6. Detailed evaluation (for diploma report/defense)
# ---------------------------------------------------------------------------
print("\n--- Classification Report ---")
print(classification_report(
    y_test, best_preds,
    target_names=["Rejected (0)", "Approved (1)"]
))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, best_preds)
print(f"  True Negatives  (correctly predicted rejected):  {cm[0][0]}")
print(f"  False Positives (predicted approved, was rejected): {cm[0][1]}")
print(f"  False Negatives (predicted rejected, was approved): {cm[1][0]}")
print(f"  True Positives  (correctly predicted approved):  {cm[1][1]}")

# Feature importance — shows which factors matter most for approval decisions
print("\n--- Feature Importance ---")
feature_names = [
    "Patient Age",
    "Drug",
    "Insurance Company",
    "Diagnosis Code",
    "Prior Drug Tried",
    "Lab Results Available",
    "Days Supply Requested",
]
if best_name == "XGBoost":
    importances = best_model.feature_importances_
else:
    importances = best_model.feature_importances_

importance_pairs = sorted(
    zip(feature_names, importances), key=lambda x: x[1], reverse=True
)
for name, imp in importance_pairs:
    bar = "#" * int(imp * 50)
    print(f"  {name:<25s} {imp:.4f} {bar}")

# ---------------------------------------------------------------------------
# 7. Save model and encoders
# ---------------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "approval_model.joblib")
encoders_path = os.path.join(MODEL_DIR, "encoders.joblib")
metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")

joblib.dump(best_model, model_path)
joblib.dump(encoders, encoders_path)

# Save metadata for the dashboard to display
metadata = {
    "model_name": best_name,
    "accuracy": round(best_acc, 4),
    "f1_score": round(best_f1, 4),
    "feature_names": feature_names,
    "feature_importance": {name: round(float(imp), 4) for name, imp in importance_pairs},
    "confusion_matrix": cm.tolist(),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "classification_report": classification_report(
        y_test, best_preds,
        target_names=["Rejected (0)", "Approved (1)"],
        output_dict=True,
    ),
}
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to:    {model_path}")
print(f"Encoders saved to: {encoders_path}")
print(f"Metadata saved to: {metadata_path}")
print("\n[DONE] Training complete!")
