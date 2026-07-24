# Implementation Plan: AI-Driven Prior Authorization Assistant (Diploma MVP)

## 1. Project Scope (Realistic Version)

The original proposal is a full SaaS product (real insurance scraping, real claims data, real PMS integration). For a diploma project, we build a **proof-of-concept** that demonstrates the same AI capabilities using synthetic/simulated data, in a self-contained demo app.

**What we keep from the original idea:**
- Predicting approval likelihood
- Interpreting rejection codes into actionable steps
- Auto-drafting PA justification text
- A pharmacist-facing dashboard

**What we drop (for now):**
- Live web scraping of insurance portals
- Real HIPAA-protected claims data
- Real Pharmacy Management System (PMS) integration
- Live pilot testing with real pharmacists

---

## 2. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Synthetic Data  │ --> │  ML/NLP Models    │ --> │  Streamlit Dashboard│
│  (CSV/DB)        │     │  (Prediction +     │     │  (Pharmacist UI)   │
│                  │     │   NLP interpreter) │     │                    │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

- **Backend:** Python (FastAPI, optional — Streamlit alone can call functions directly for a demo)
- **ML layer:** scikit-learn / XGBoost for approval prediction
- **NLP layer:** rule-based + small transformer model (or even a well-prompted LLM) for rejection code interpretation
- **Storage:** SQLite or PostgreSQL (you already know pgvector/Postgres from Squire — reuse that stack)
- **Frontend:** Streamlit (fastest way to get a clean pharmacist dashboard without building a full web app)

---

## 3. Phase-by-Phase Plan

### Phase 1 — Synthetic Data Design (Week 1)

Build a fake but realistic dataset that mimics insurance rules.

**Table: `insurance_rules`**
| Column | Example |
|---|---|
| insurance_name | "CareFirst", "BlueShield-X" |
| drug_name | "Ozempic" |
| requires_step_therapy | true/false |
| required_prior_drugs | ["Metformin"] |
| requires_lab_results | true/false |
| max_days_supply_without_PA | 30 |

**Table: `claims`** (used to train the prediction model)
| Column | Example |
|---|---|
| patient_age | 54 |
| drug_name | "Ozempic" |
| insurance_name | "CareFirst" |
| diagnosis_code | "E11.9" |
| prior_drugs_tried | ["Metformin"] |
| had_lab_results | true |
| **approved** (target) | 1 / 0 |

Generate ~1,000–3,000 synthetic rows (you've already done this scale of dataset work for your NER project — same approach: define rules, then generate rows that follow or intentionally violate them, so the model has real signal to learn).

**Table: `rejection_codes`**
| Column | Example |
|---|---|
| code | "DNC-004" |
| raw_message | "Drug Not Covered — Step Therapy Required" |
| plain_explanation | "Patient must try a cheaper alternative first" |
| action_checklist | ["Check if Metformin was tried", "Attach last 3 months lab results"] |

---

### Phase 2 — Model Development (Weeks 2–3)

**2.1 Predictive Approval Engine**
- Model: XGBoost or Random Forest (binary classification: approved / rejected)
- Features: patient age, drug, insurance, diagnosis code, prior drugs tried, lab results present
- Output: probability score (e.g., "78% likely to be approved")
- Since this is synthetic data, you control the "ground truth" rules — this makes it easy to demonstrate the model actually learned something meaningful (e.g., test that it correctly penalizes missing step therapy).

**2.2 Rejection Code Interpreter**
- Simplest approach: a lookup table (code → plain explanation → checklist) — legitimate for a diploma project scope, and honest about being rule-based rather than "AI" if you present it that way.
- Stronger version: fine-tune a small NLP classifier, or use an LLM prompt template that takes the raw rejection message and outputs a structured checklist. Given your NLP background, this is a good place to show more depth — e.g., a lightweight intent/slot extraction model similar to what you built for your notes app classifier.

**2.3 Smart Form Filler (optional stretch goal)**
- Take structured claim data + rejection reason → generate a draft PA justification paragraph using a template or LLM call.
- Keep it template-based first; only add LLM generation if time allows.

---

### Phase 3 — Dashboard (Week 4)

Streamlit app with three views:

1. **New Claim Check** — pharmacist enters patient/drug/insurance info → gets approval probability + risk factors
2. **Rejection Assistant** — pharmacist pastes/selects a rejection code → gets plain-language explanation + checklist
3. **Analytics view** — simple charts: approval rate by insurance, most common rejection reasons (good for the "predictive analytics" section of your defense)

---

### Phase 4 — Validation & Presentation Prep (Week 5)

- **Model evaluation:** accuracy, precision/recall, confusion matrix on held-out synthetic test set (you've already dealt with class imbalance issues before — apply the same care here if approvals/rejections aren't 50/50)
- **Synthetic "pilot":** simulate 20–30 example claims flowing through the whole pipeline, screenshot/record the demo
- **Clearly document in your report:** this is a proof-of-concept on synthetic data, with a documented path to real data integration (mention HIPAA, real PMS APIs, etc. as "future work" — this is honest and actually strengthens the academic framing)

---

## 4. Suggested Tech Stack Summary

| Component | Tool |
|---|---|
| Data generation | Python (pandas, Faker library for realistic names/patient data) |
| ML model | scikit-learn / XGBoost |
| NLP interpreter | Rule-based lookup + optional small transformer or LLM prompt |
| Database | PostgreSQL or SQLite |
| Dashboard | Streamlit |
| Deployment (optional) | Hugging Face Spaces (you already deploy there) |

---

## 5. Realistic Timeline

| Week | Deliverable |
|---|---|
| 1 | Synthetic dataset + insurance rules table finalized |
| 2 | Approval prediction model trained + evaluated |
| 3 | Rejection interpreter + checklist logic done |
| 4 | Streamlit dashboard connecting everything |
| 5 | Testing, polish, report writing, defense prep |

---

## 6. What to Say in the Defense

Be upfront and confident about the scope decision — it reads as good engineering judgment, not a shortcut:
- "We designed a synthetic dataset modeling real-world insurance PA rules based on published payer criteria patterns."
- "The system demonstrates the full AI pipeline — prediction, NLP interpretation, and workflow assistance — validated on controlled data, with a clear integration path for real claims data and PMS systems as future work."

---

## 7. Next Steps If You Take This On

1. Confirm with your relative: solo build or does she need to contribute code/analysis herself for the diploma requirements?
2. Decide: rule-based rejection interpreter (faster, safer) vs. NLP model (more impressive, more time)
3. I can generate the synthetic dataset schema + generation script first — that unblocks everything else
