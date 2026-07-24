"""
Phase 2.1 — Prediction Module (Inference)
AI-Driven Prior Authorization Assistant (Diploma Project)

This module loads the trained ML model and provides a simple function
that the Streamlit dashboard calls to predict approval likelihood.

Usage:
    from models.predict import predict_approval

    result = predict_approval(
        patient_age=54,
        drug_name="Ozempic",
        insurance_name="CareFirst",
        diagnosis_code="E11.9",
        prior_drug_tried=True,
        had_lab_results=True,
        days_supply=90,
    )
    # result = {
    #     "probability": 0.78,
    #     "approved_prediction": True,
    #     "risk_level": "LOW",
    #     "risk_factors": [...],
    #     "recommendation": "..."
    # }
"""

import os
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Load model and encoders (loaded once when module is imported)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved")

_model = None
_encoders = None


def _load_model():
    """Lazy-load the model and encoders on first prediction call."""
    global _model, _encoders
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "approval_model.joblib")
        encoders_path = os.path.join(MODEL_DIR, "encoders.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python models/train_model.py' first."
            )

        _model = joblib.load(model_path)
        _encoders = joblib.load(encoders_path)


def _safe_encode(encoder, value):
    """
    Encode a categorical value safely.

    If the value wasn't seen during training (e.g., a new drug name),
    return -1 instead of crashing. The model will still make a prediction,
    just with lower confidence — this is a graceful degradation strategy.
    """
    try:
        return encoder.transform([str(value)])[0]
    except ValueError:
        return -1


def predict_approval(
    patient_age: int,
    drug_name: str,
    insurance_name: str,
    diagnosis_code: str,
    prior_drug_tried: bool,
    had_lab_results: bool,
    days_supply: int,
    insurance_rules_df: pd.DataFrame = None,
) -> dict:
    """
    Predict the likelihood of an insurance claim being approved.

    Parameters
    ----------
    patient_age : int
        Patient's age in years.
    drug_name : str
        Name of the prescribed medication (e.g., "Ozempic").
    insurance_name : str
        Insurance company name (e.g., "CareFirst").
    diagnosis_code : str
        ICD-10 diagnosis code (e.g., "E11.9" for Type 2 Diabetes).
    prior_drug_tried : bool
        Whether the patient has tried the required prior/alternative drug.
    had_lab_results : bool
        Whether recent lab results are available.
    days_supply : int
        Number of days' supply requested.
    insurance_rules_df : pd.DataFrame, optional
        The insurance rules table — used to generate specific risk factors.

    Returns
    -------
    dict with keys:
        - probability (float): 0.0 to 1.0 approval probability
        - approved_prediction (bool): True if model predicts approval
        - risk_level (str): "LOW", "MEDIUM", or "HIGH" risk of rejection
        - risk_factors (list[str]): specific reasons that might cause rejection
        - recommendation (str): plain-language recommendation for the pharmacist
    """
    _load_model()

    # Encode categorical features using the same encoders from training
    drug_encoded = _safe_encode(_encoders["drug_name"], drug_name)
    insurance_encoded = _safe_encode(_encoders["insurance_name"], insurance_name)
    diagnosis_encoded = _safe_encode(_encoders["diagnosis_code"], diagnosis_code)

    # Convert booleans to the same numeric format used during training
    prior_drug_val = 1 if prior_drug_tried else 0
    lab_results_val = 1 if had_lab_results else 0

    # Build the feature vector (same order as training!)
    features = np.array([[
        patient_age,
        drug_encoded,
        insurance_encoded,
        diagnosis_encoded,
        prior_drug_val,
        lab_results_val,
        days_supply,
    ]])

    # Get probability from the model
    # predict_proba returns [[prob_rejected, prob_approved]]
    probabilities = _model.predict_proba(features)[0]
    approval_prob = float(probabilities[1])  # probability of class 1 (approved)
    prediction = bool(approval_prob >= 0.5)

    # Determine risk level based on approval probability
    if approval_prob >= 0.75:
        risk_level = "LOW"
    elif approval_prob >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # ---------------------------------------------------------------------------
    # Generate specific, actionable risk factors
    # This is where we combine the ML prediction with rule-based logic
    # to give the pharmacist concrete reasons, not just a number
    # ---------------------------------------------------------------------------
    risk_factors = []

    if insurance_rules_df is not None:
        # Look up the specific rules for this drug + insurance combination
        rule_match = insurance_rules_df[
            (insurance_rules_df["insurance_name"] == insurance_name) &
            (insurance_rules_df["drug_name"] == drug_name)
        ]

        if not rule_match.empty:
            rule = rule_match.iloc[0]

            # Check step therapy requirement
            if rule.get("requires_step_therapy") in [True, "True"]:
                required_drug = rule.get("required_prior_drug", "an alternative")
                if not prior_drug_tried:
                    risk_factors.append(
                        f"⚠️ Step therapy required: patient must try {required_drug} first"
                    )
                else:
                    risk_factors.append(
                        f"✅ Step therapy satisfied: {required_drug} was tried"
                    )

            # Check lab results requirement
            if rule.get("requires_lab_results") in [True, "True"]:
                if not had_lab_results:
                    risk_factors.append(
                        "⚠️ Lab results required but not provided"
                    )
                else:
                    risk_factors.append(
                        "✅ Required lab results are available"
                    )

            # Check days supply limit
            max_days = rule.get("max_days_supply_without_pa", 90)
            if isinstance(max_days, str):
                max_days = int(max_days)
            if days_supply > max_days:
                risk_factors.append(
                    f"⚠️ Requested {days_supply} days exceeds plan limit of {max_days} days without PA"
                )
            else:
                risk_factors.append(
                    f"✅ Days supply ({days_supply}) within plan limit ({max_days})"
                )

    # Generate recommendation
    if risk_level == "LOW":
        recommendation = (
            "High likelihood of approval. Submit the claim directly — "
            "no prior authorization likely needed."
        )
    elif risk_level == "MEDIUM":
        recommendation = (
            "Moderate risk of rejection. Review the risk factors above and "
            "ensure all requirements are met before submitting."
        )
    else:
        recommendation = (
            "High risk of rejection. Address all flagged risk factors before "
            "submitting. Consider initiating a prior authorization request proactively."
        )

    return {
        "probability": round(approval_prob, 4),
        "approved_prediction": prediction,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
    }
