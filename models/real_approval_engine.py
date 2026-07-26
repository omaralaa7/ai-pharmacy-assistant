"""
Real-Data Approval & Compliance Engine
AI-Driven Insurance Assistant (Diploma Project Prototype)

Evaluates prescription requests against REAL Egyptian insurance policy rules
from insurance_knowledge_base.json and generates official approval letters.
"""

import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
KB_PATH = os.path.join(PROJECT_DIR, "data", "insurance_knowledge_base.json")


def _load_kb():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_real_approval(company_key, item_name, days_requested, has_doctor_stamp=True, has_diagnosis=True):
    """
    Evaluate approval probability & compliance risk based on real company policy rules.

    Parameters
    ----------
    company_key : str
        Company key in knowledge base.
    item_name : str
        Medication or item name requested.
    days_requested : int
        Days supply requested (e.g. 14, 30, 60).
    has_doctor_stamp : bool
        Whether the prescription has a valid doctor/hospital stamp.
    has_diagnosis : bool
        Whether diagnosis is written on prescription.

    Returns
    -------
    dict with score, risk_level, risk_factors, recommendation, details
    """
    kb = _load_kb()
    if company_key not in kb:
        return {"error": "Company not found"}

    company_data = kb[company_key]
    policies = company_data.get("policies", {})

    score = 100
    risk_factors = []

    # 1. Check Exclusion List (المحظورات)
    exclusion_policy = policies.get("المحظورات", {})
    exclusion_text = exclusion_policy.get("details", "")
    if exclusion_text and item_name:
        item_lower = item_name.strip().lower()
        excl_lower = exclusion_text.lower()
        if item_lower in excl_lower:
            score -= 50
            risk_factors.append(f"🚫 الصنف '{item_name}' يندرج ضمن قائمة المحظورات الصريحة للشركة.")

    # 2. Check Max Duration (أقصى مدة للصرف)
    duration_policy = policies.get("أقصى مدة للصرف", {})
    duration_text = duration_policy.get("details", "")
    if duration_text and days_requested:
        # Check if duration mentions 14 days / week / fortnight / 2 weeks
        if ("اسبوعين" in duration_text or "أسبوعين" in duration_text or "14" in duration_text or "7 أيام" in duration_text) and days_requested > 14:
            score -= 30
            risk_factors.append(f"⏱️ الفترة المطلوبة ({days_requested} يوم) تتجاوز الحد الأقصى المسموح ({duration_text}).")

    # 3. Check Stamp Requirement (الختم / إمضاء العميل)
    stamp_policy = policies.get("الختم / إمضاء العميل", {})
    stamp_text = stamp_policy.get("details", "")
    if stamp_text:
        requires_stamp = "يشترط" in stamp_text or "لابد من وجود ختم" in stamp_text or "مختوم" in stamp_text
        if requires_stamp and not has_doctor_stamp:
            score -= 20
            risk_factors.append("✍️ الروشتة غير مختومة، والشركة تشترط وجود ختم الطبيب/المستشفى.")

    # 4. Check Diagnosis Requirement (التشخيص)
    diag_policy = policies.get("التشخيص", {})
    diag_text = diag_policy.get("details", "")
    if diag_text:
        requires_diag = "يشترط" in diag_text or "لابد" in diag_text
        if requires_diag and not has_diagnosis:
            score -= 15
            risk_factors.append("🩺 التشخيص غير مدون، والشركة تشترط وجود التشخيص على الروشتة.")

    # Ensure score stays between 0 and 100
    score = max(0, min(100, score))

    if score >= 80:
        risk_level = "LOW"
        recommendation = "✅ المطالبة متوافقة بشكل مرتفع مع ضوابط وقواعد الشركة. يمكن الصرف مباشرة."
    elif score >= 50:
        risk_level = "MEDIUM"
        recommendation = "⚠️ المطالبة تتضمن بعض الملاحظات. يرجى استكمال الأختام أو تعديل الكمية المحددة."
    else:
        risk_level = "HIGH"
        recommendation = "🚫 المطالبة عالية المخاطر وقد تتطلب موافقة مسبقة تليفونية أو إعادة الروشتة للطبيب."

    return {
        "company_name": company_data.get("company_name", company_key),
        "score": score / 100.0,
        "score_percent": score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "policies": policies,
    }


def generate_real_pa_letter(company_key, patient_name, item_name, diagnosis, pharmacist_notes=""):
    """
    Generate an official Prior Authorization / Approval Request Letter
    populated with real company details, contacts, and rules.
    """
    kb = _load_kb()
    company_data = kb.get(company_key, {})
    policies = company_data.get("policies", {})

    company_name = company_data.get("company_name", company_key)
    contact_info = policies.get("التواصل للموافقات", {}).get("details", "غير متاح")
    forms_info = policies.get("نماذج الصرف", {}).get("details", "روشتة معتمدة")

    letter_ar = f"""خطاب طلب موافقة طبية مسبقة (Prior Authorization Request)
----------------------------------------------------------------------
إلى: إدارة الرعاية الطبية - {company_name}
التاريخ: اليوم
بيانات المريض: {patient_name or 'المريض المحترم'}
التشخيص الطبي: {diagnosis or 'غير مدون'}
المستحضر المطلوب: {item_name or 'المستحضر الطبي'}

تحية طيبة وبعد،،،

نود إفادتكم بضرورة صرف المستحضر الطبي الموضح أعلاه للمريض المذكور طبقاً للتعليمات والوصفة الطبية المدونة.

بيانات الاتصال وإجراءات الموافقة المسجلة لدى الشركة:
{contact_info}

نظام نموذج الصرف المعتمد:
{forms_info}

ملاحظات الصيدلي:
{pharmacist_notes or 'يرجى التكرم بالإفادة بالرقم المرجعي للموافقة لتسهيل الصرف للمريض.'}

شاكرين لكم حسن التعاون،،،
صيدلية: _____________________
توقيع الصيدلي: __________________
----------------------------------------------------------------------"""

    return letter_ar
