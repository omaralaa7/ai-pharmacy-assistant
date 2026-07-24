# 💊 AI-Driven Prior Authorization Assistant

An AI-powered proof-of-concept system that helps pharmacists navigate insurance prior authorization (PA) workflows using machine learning and NLP.

> **Diploma Project** — Built with synthetic data to demonstrate the full AI pipeline (prediction, NLP interpretation, workflow assistance) with a clear integration path for real claims data.

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| **Approval Prediction** | ML model predicts the likelihood of insurance claim approval based on patient, drug, and insurance data |
| **Rejection Interpreter** | Translates cryptic rejection codes into plain-language explanations with step-by-step action checklists |
| **PA Letter Generator** | Auto-generates prior authorization justification letter drafts |
| **Analytics Dashboard** | Visualizes approval rates, rejection patterns, and claim trends |
| **Model Performance** | Displays ML evaluation metrics (accuracy, F1, confusion matrix, feature importance) |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend / Dashboard | Streamlit |
| ML Model | XGBoost / Random Forest (scikit-learn) |
| NLP Interpreter | Rule-based + fuzzy string matching |
| Data Visualization | Plotly |
| Data | Synthetic (pandas, Faker) |

## 📁 Project Structure

```
AI Pharmacy/
├── app.py                      # Main Streamlit dashboard (run this)
├── requirements.txt            # Python dependencies
├── generate_dataset.py         # Synthetic data generator
├── data/
│   ├── claims.csv              # 2,500 synthetic insurance claims
│   ├── insurance_rules.csv     # 60 insurance rules (5 insurers × 12 drugs)
│   └── rejection_codes.csv     # 6 rejection codes with explanations
├── models/
│   ├── train_model.py          # ML model training script
│   ├── predict.py              # Prediction inference module
│   ├── rejection_interpreter.py # Rejection code interpreter
│   └── saved/                  # Trained model files (generated)
│       ├── approval_model.joblib
│       ├── encoders.joblib
│       └── model_metadata.json
└── .streamlit/
    └── config.toml             # Dashboard theme configuration
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model
```bash
python models/train_model.py
```
This trains the approval prediction model and saves it to `models/saved/`.

### 3. Run the Dashboard
```bash
streamlit run app.py
```
The dashboard will open at `http://localhost:8501`.

## 📊 Dashboard Pages

1. **🏥 New Claim Check** — Enter patient/drug/insurance details → get approval probability + risk analysis
2. **🔍 Rejection Assistant** — Look up rejection codes → get explanations + action checklists + generate PA letters
3. **📊 Analytics Dashboard** — Charts showing approval rates by insurer, drug, age, and rejection patterns
4. **🧪 Model Performance** — ML evaluation metrics for academic validation (confusion matrix, feature importance)

## 🧠 How the AI Works

### Predictive Approval Engine
- **Algorithm:** XGBoost (gradient boosting) with Random Forest fallback
- **Features:** Patient age, drug, insurance, diagnosis code, step therapy status, lab results, days supply
- **Output:** Approval probability (0-100%) with risk level (LOW/MEDIUM/HIGH)
- **Class Imbalance:** Handled via `scale_pos_weight` (XGBoost) or `class_weight='balanced'` (RF)

### Rejection Code Interpreter
- **Approach:** Rule-based lookup table + fuzzy string matching
- **Exact match:** Code → explanation + checklist (100% accuracy)
- **Fuzzy match:** Free-text description → best-matching code (SequenceMatcher)

## ⚠️ Disclaimer

This is a **proof-of-concept** built on synthetic data for educational/academic purposes. It does not process real patient data and is not intended for clinical use. Real-world deployment would require HIPAA compliance, real claims data, and PMS integration.
