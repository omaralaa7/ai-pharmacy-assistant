# 💊 Medical Insurance Policy Assistant for Pharmacies

### نظام الاستعلام عن سياسات ومحظورات التأمين الطبي للصيدليات

A digital decision support system providing Egyptian pharmacists with fast, accurate, and structured policy verification across medical insurance providers and Third Party Administrators (TPAs).

> 🌐 **Live Web Application:** [ai-pharmacy-assistant.streamlit.app](https://ai-pharmacy-assistant.streamlit.app)

---

## 🎯 Key Capabilities

| Module | Description |
|--------|-------------|
| **💬 Policy Search** | Search policy rules, exclusion lists, and contact information |
| **📖 Policy Directory** | Browse all insurance providers and policy categories |
| **⚡ Dispensing Check** | Verify item exclusions and dispensing conditions prior to dispensing |
| **🏥 Approval Check** | Evaluate rule compliance and approval likelihood score |
| **📝 PA Letter Generator** | Draft official approval request letters with company contacts |

---

## 📊 Dataset Overview

* **Insurance Entities:** 77 Egyptian insurance providers and TPAs
* **Indexed Policy Rules:** 766 rule categories
* **Policy Attributes Covered:**
  * Exclusions & Prohibited Items (*المحظورات*)
  * Maximum Dispensing Duration (*أقصى مدة للصرف*)
  * Stamp & Signature Requirements (*الختم / إمضاء العميل*)
  * Form Validity Period (*صلاحية النموذج*)
  * Card & Identity Copy Requirements (*صورة الكارنية / البطاقة*)
  * Approval Contact Endpoints (*التواصل للموافقات*)

---

## 🛠️ System Architecture

* **Frontend:** Streamlit
* **Search Engine:** Vector space retrieval model (TF-IDF & Cosine Similarity)
* **Arabic Text Normalization:** Diacritic stripping, Alef/Ya normalization, stop-word filtering
* **Knowledge Store:** Structured JSON database derived from Egyptian TPA rules

---

## 📁 Repository Structure

```
AI Pharmacy/
├── app.py                              # Streamlit Web Application
├── requirements.txt                    # Python dependencies
├── README.md                           # Documentation & system guide
├── نظم صرف شركات التأمين-2.xlsx        # Primary Egyptian insurance dataset
├── data/
│   └── insurance_knowledge_base.json   # Structured JSON knowledge base (77 companies)
├── models/
│   ├── __init__.py                  # Python package marker
│   ├── chatbot_engine.py               # Policy search & retrieval engine
│   └── real_approval_engine.py         # Compliance evaluation & PA letter generator
├── docs/                               # Thesis methodology documentation (.docx & .md)
└── notebooks/                          # Academic evaluation notebooks
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the application
streamlit run app.py
```

Dashboard runs locally at: **http://localhost:8501**
