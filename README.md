# 💊 AI-Driven Insurance Chatbot for Pharmacies

### مساعد التأمين الذكي للصيدليات

An AI-powered chatbot that provides Egyptian pharmacists with fast, accurate, and interactive access to medical insurance dispensing policies using NLP and Machine Learning.

> **Diploma Project** — Built with real Egyptian insurance data (77 companies, 766 policy rules) and a TF-IDF based natural language search engine.

---

## 🎯 What It Does

| Module | Description |
|--------|-------------|
| **💬 AI Chatbot** | Ask questions in Arabic or English about any insurance company's policies |
| **📖 Policy Directory** | Browse all 77 companies and 14 policy categories |
| **⚡ Dispensing Check** | Verify exclusions, stamp requirements, max duration, and co-pay rules |
| **📊 Analytics** | Visual insights across all Egyptian insurance providers |
| **🏥 Approval Prediction** | ML model predicts claim approval likelihood (synthetic data demo) |
| **🔍 Rejection Assistant** | Interprets rejection codes into actionable checklists + PA letter generator |
| **🧪 Model Performance** | ML evaluation metrics (accuracy, F1, confusion matrix, feature importance) |

---

## 📊 Real Dataset

| Metric | Value |
|--------|-------|
| Insurance Companies | 77 |
| Policy Rules | 766 |
| Rule Categories | 14 |
| Data Source | Real Egyptian insurance dispensing rules |

### Policy Categories Covered
نماذج الصرف، المحظورات، التحمل، التشخيص، صلاحية النموذج، صورة البطاقة، صورة الكارنية، الختم/إمضاء العميل، أقصى مدة للصرف، الحد الأقصى، التواصل للموافقات، لينك الأونلاين سيستم، البدائل، ملاحظات

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend / Dashboard | Streamlit |
| NLP Search Engine | TF-IDF + Cosine Similarity (scikit-learn) |
| Arabic NLP | Custom preprocessing (diacritics, alef/ya normalization) |
| ML Model | XGBoost / Random Forest |
| Data Visualization | Plotly |
| Knowledge Base | JSON (from Excel via openpyxl) |

---

## 📁 Project Structure

```
AI Pharmacy/
├── app.py                              # Main Streamlit dashboard (7 pages)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── generate_dataset.py                 # Synthetic data generator
├── نظم صرف شركات التأمين-2.xlsx        # Real Egyptian insurance dataset
├── data/
│   ├── process_excel.py                # Excel → JSON knowledge base
│   ├── insurance_knowledge_base.json   # Structured knowledge base (77 companies)
│   ├── claims.csv                      # 2,500 synthetic claims
│   ├── insurance_rules.csv             # Synthetic insurance rules
│   └── rejection_codes.csv             # Rejection codes lookup
├── models/
│   ├── chatbot_engine.py               # TF-IDF NLP chatbot engine
│   ├── predict.py                      # ML approval prediction
│   ├── rejection_interpreter.py        # Rejection code interpreter
│   ├── train_model.py                  # ML model training script
│   └── saved/                          # Trained model files
├── notebooks/                          # Jupyter notebooks (Colab-ready)
│   ├── 01_data_exploration.ipynb       # Data analysis & knowledge base
│   ├── 02_nlp_chatbot_development.ipynb # NLP engine development
│   └── 03_evaluation_and_results.ipynb  # System evaluation & KPIs
└── .streamlit/
    └── config.toml                     # Dashboard theme
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process the Excel dataset (if knowledge base doesn't exist)
python data/process_excel.py

# 3. Train the ML model (if not already trained)
python models/train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

Dashboard runs at: **http://localhost:8501**

---

## 🧠 How the AI Works

### NLP Chatbot (TF-IDF Search)
1. **Arabic Preprocessing:** Remove diacritics, normalize alef/ya/ta-marbuta
2. **TF-IDF Indexing:** Build weighted term vectors over all policy text
3. **Multi-Strategy Retrieval:**
   - Direct lookup (exact company + category match)
   - Fuzzy matching (handles typos and partial names)
   - TF-IDF cosine similarity (semantic search fallback)
4. **Bilingual:** Supports Arabic and English queries

### ML Approval Prediction
- Algorithm: Random Forest (92.4% accuracy on synthetic data)
- Features: Patient age, drug, insurance, diagnosis, step therapy, lab results, days supply

---

## ⚠️ Disclaimer

This is a **diploma project / proof-of-concept**. The insurance policy data is used for educational purposes. Always verify dispensing rules with the insurance company directly before dispensing medications.
