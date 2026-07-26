"""
Phase 3 — Pharmacist Dashboard (Streamlit Application)
AI-Driven Insurance Chatbot (Diploma Project)

Integrated application providing:
  1. AI Chatbot — natural language queries about insurance policies
  2. Policy Directory — browse all company policy categories
  3. Dispensing Quick-Check — verify exclusions and requirements
  4. Analytics — visual insights across insurance providers
  5. Approval Prediction — ML model for claims approval
  6. Rejection Assistant — code interpreter & letter generator
  7. Model Performance — evaluation metrics

Run locally with: streamlit run app.py
"""

import os
import sys
import json
import html
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Auto-train ML model if not found
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "saved", "approval_model.joblib")
if not os.path.exists(MODEL_PATH):
    import subprocess
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_DIR, "models", "train_model.py")],
        check=True,
    )

from models.predict import predict_approval
from models.rejection_interpreter import (
    interpret_rejection,
    get_all_codes,
    generate_pa_letter,
)
from models.chatbot_engine import (
    chat_query,
    get_all_companies,
    get_company_policies,
    get_all_categories,
    check_exclusions,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_DIR, "data")


@st.cache_data
def load_claims():
    return pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))


@st.cache_data
def load_insurance_rules():
    return pd.read_csv(os.path.join(DATA_DIR, "insurance_rules.csv"))


@st.cache_data
def load_model_metadata():
    metadata_path = os.path.join(PROJECT_DIR, "models", "saved", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_knowledge_base():
    kb_path = os.path.join(DATA_DIR, "insurance_knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# RTL & Text Formatter Helper
# ---------------------------------------------------------------------------
def format_arabic_html(text):
    """
    Renders Arabic/English policy text with strict RTL alignment,
    cleaning punctuation artifacts (e.g. colons at line starts) and bullets.
    """
    if not text:
        return '<div dir="rtl" style="direction: rtl; text-align: right; color: rgba(250,250,250,0.5); font-style: italic;">لا توجد تفاصيل متاحة</div>'

    lines = [line.strip() for line in str(text).split("\n") if line.strip()]
    formatted_elements = []

    for line in lines:
        # Fix leading colon export artifact
        if line.startswith(":"):
            line = line[1:].strip() + " :"
        elif line.startswith(":-"):
            line = line[2:].strip() + " :-"

        safe_line = html.escape(line)

        # Bullet item check
        match_bullet = re.match(r"^([\-\•\*\d+[\.\-]]+)\s*(.+)", safe_line)
        if match_bullet:
            bullet_mark = match_bullet.group(1)
            content = match_bullet.group(2)
            formatted_elements.append(
                f'<div style="display: flex; flex-direction: row-reverse; justify-content: flex-end; align-items: flex-start; gap: 0.6rem; margin: 0.4rem 0; line-height: 1.8;">'
                f'<span style="color: #00D4AA; font-weight: 600; flex-shrink: 0;">{bullet_mark}</span>'
                f'<span style="flex: 1; text-align: right; direction: rtl; unicode-bidi: isolate;">{content}</span>'
                f'</div>'
            )
        else:
            formatted_elements.append(
                f'<p style="margin: 0.4rem 0; line-height: 1.8; text-align: right; direction: rtl; unicode-bidi: isolate;">{safe_line}</p>'
            )

    return f'''<div dir="rtl" style="direction: rtl; text-align: right; font-family: 'Noto Sans Arabic', 'Inter', sans-serif; unicode-bidi: isolate; background: rgba(26, 31, 46, 0.95); border: 1px solid rgba(0, 212, 170, 0.2); border-radius: 12px; padding: 1.2rem; margin: 0.6rem 0; color: #e2e8f0; font-size: 0.95rem;">
{''.join(formatted_elements)}
</div>'''


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Insurance Assistant | مساعد التأمين الطبي",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Arabic:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', 'Noto Sans Arabic', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #059669 50%, #047857 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.15);
    }
    .main-header h1 { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.12);
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #00D4AA; line-height: 1.2; }
    .kpi-label {
        font-size: 0.8rem; color: rgba(250,250,250,0.6);
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem;
    }

    .risk-low { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 600; display: inline-block; }
    .risk-medium { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 600; display: inline-block; }
    .risk-high { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 600; display: inline-block; }

    .chat-bubble-user {
        background: rgba(0, 212, 170, 0.12);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 12px 12px 4px 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        direction: auto;
    }

    .checklist-item {
        background: rgba(0, 212, 170, 0.05);
        border-right: 3px solid #00D4AA;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px 0 0 8px;
        direction: rtl;
        text-align: right;
    }

    .status-box {
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #00D4AA;
        border-bottom: 2px solid rgba(0, 212, 170, 0.2);
        padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }

    .excluded-alert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        direction: rtl;
        text-align: right;
    }
    .approved-alert {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        direction: rtl;
        text-align: right;
    }

    .pa-letter {
        background: #1a1f2e; border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px; padding: 1.5rem;
        font-family: 'Courier New', monospace; font-size: 0.85rem;
        white-space: pre-wrap; line-height: 1.6;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💊 AI Insurance Assistant")
    st.markdown("##### نظام مساعد التأمين الطبي الذكي")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "💬 AI Chatbot",
            "📖 Policy Directory",
            "⚡ Dispensing Check",
            "📊 Analytics",
            "🏥 Approval Prediction",
            "🔍 Rejection Assistant",
            "🧪 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("AI-Driven Medical Insurance System")
    st.caption("Pharmacy Decision Support Prototype")


# ===========================================================================
# PAGE 1: AI CHATBOT
# ===========================================================================
if page == "💬 AI Chatbot":
    st.markdown("""
    <div class="main-header">
        <h1>💬 AI Insurance Assistant</h1>
        <p>مساعد الاستعلام الذكي عن سياسات وقواعد صرف التأمين الطبي</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Example Queries / أمثلة على الاستعلامات"):
        st.markdown("""
        - **ما هي محظورات يونايتد؟** — قائمة المواد والمستحضرات المحظور صرفها
        - **أقصى مدة صرف لشركة ويبكو** — أقصى فترة علاجية مسموح بصرفها
        - **رقم تليفون موافقات دريم مشرق** — أرقام التواصل للموافقات الطبية
        - **هل يشترط ختم لشركة جلوبميد؟** — ضوابط وأختام الروشتات
        - **What is the copay for ALICO?** — نسبة التحمل ورسوم الخدمة
        """)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask about any insurance company...",
            placeholder="اكتب استفسارك هنا... (مثال: ما هي محظورات شركة يونايتد؟)",
            key="chat_input",
            label_visibility="collapsed",
        )
    with col_btn:
        send_btn = st.button("🔍 بحث", type="primary", use_container_width=True)

    if send_btn and user_question:
        result = chat_query(user_question)
        st.session_state.chat_history.append({"question": user_question, "result": result})

    for entry in reversed(st.session_state.chat_history):
        q = entry["question"]
        r = entry["result"]

        st.markdown(f'<div class="chat-bubble-user">🧑‍⚕️ {q}</div>', unsafe_allow_html=True)

        if r.get("found"):
            # Only show company & category header if it's an actual policy (not a greeting)
            if r.get("category") != "ترحيب":
                header_text = f"🏢 <strong>{r.get('company_name', '')}</strong> — {r.get('category', '')}"
                st.markdown(f'<div style="direction: rtl; text-align: right; color: #00D4AA; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem;">{header_text}</div>', unsafe_allow_html=True)

            st.markdown(format_arabic_html(r.get("answer", "")), unsafe_allow_html=True)

            if r.get("notes"):
                st.caption(f"📝 ملاحظات إضافية: {r['notes']}")

            if r.get("other_results"):
                with st.expander("نتائج إضافية ذات صلة"):
                    for other in r["other_results"]:
                        st.markdown(f"**{other['company_name']}** — {other['category']}")
                        st.markdown(format_arabic_html(other["answer"][:200]), unsafe_allow_html=True)
        else:
            error_msg = r.get("error", "لم يتم العثور على نتائج مطابقة.")
            st.markdown(f'<div class="chat-bubble-bot" dir="rtl" style="text-align: right;">❌ {error_msg}</div>', unsafe_allow_html=True)

    if st.button("🗑️ مسح السجل"):
        st.session_state.chat_history = []
        st.rerun()


# ===========================================================================
# PAGE 2: POLICY DIRECTORY
# ===========================================================================
elif page == "📖 Policy Directory":
    st.markdown("""
    <div class="main-header">
        <h1>📖 Policy Directory</h1>
        <p>دليل وسجل سياسات التأمين الطبي لجميع الشركات الهيئات</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']}": c["key"] for c in companies}

    selected_label = st.selectbox(
        "Select Insurance Company / اختر الجهة الضامنة",
        options=list(company_options.keys()),
    )

    if selected_label:
        selected_key = company_options[selected_label]
        company_data = get_company_policies(selected_key)

        if company_data:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size:1.2rem;">{company_data.get('company_name_ar', '')}</div>
                    <div class="kpi-label">الاسم بالعربية</div>
                </div>
                """, unsafe_allow_html=True)
            with col_info2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size:1.2rem;">{company_data.get('company_name_en', '').upper() or 'N/A'}</div>
                    <div class="kpi-label">English Name</div>
                </div>
                """, unsafe_allow_html=True)
            with col_info3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{len(company_data.get('policies', {}))}</div>
                    <div class="kpi-label">عدد بنود السياسة</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            all_cats = list(company_data.get("policies", {}).keys())
            filter_cat = st.multiselect("تصفية حسب البند (اختياري)", all_cats)
            cats_to_show = filter_cat if filter_cat else all_cats

            for category in cats_to_show:
                policy = company_data["policies"][category]
                cat_en = policy.get("category_en", "")

                with st.expander(f"📋 {category} — {cat_en}", expanded=len(cats_to_show) <= 3):
                    st.markdown(format_arabic_html(policy.get("details", "")), unsafe_allow_html=True)
                    if policy.get("notes"):
                        st.info(f"📝 ملاحظات: {policy['notes']}")


# ===========================================================================
# PAGE 3: DISPENSING QUICK-CHECK
# ===========================================================================
elif page == "⚡ Dispensing Check":
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Dispensing Quick-Check</h1>
        <p>نظام التحقق السريع من المحظورات وضوابط الصرف قبل الدواء</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']}": c["key"] for c in companies}

    col_check1, col_check2 = st.columns([1, 1.5])

    with col_check1:
        st.markdown('<div class="section-header">🔍 بيانات الفحص</div>', unsafe_allow_html=True)

        selected_company = st.selectbox(
            "جهة التأمين / الشركة",
            options=list(company_options.keys()),
            key="check_company",
        )

        item_to_check = st.text_input(
            "اسم المستحضر / الصنف",
            placeholder="مثال: كريمات تفتيح, فيتامينات, clexan...",
            key="check_item",
        )

        check_btn = st.button("⚡ بدء الفحص", type="primary", use_container_width=True)

    with col_check2:
        if check_btn and selected_company and item_to_check:
            company_key = company_options[selected_company]
            company_data = get_company_policies(company_key)

            result = check_exclusions(company_key, item_to_check)

            if result.get("is_excluded"):
                st.markdown(f"""
                <div class="excluded-alert">
                    <h3 style="color: #ef4444; margin: 0;">🚫 صنف محظور الصرف</h3>
                    <p style="margin: 0.5rem 0 0 0;">
                        المستحضر <strong>{item_to_check}</strong> يندرج ضمن قائمة المحظورات لشركة <strong>{selected_company}</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="approved-alert">
                    <h3 style="color: #10b981; margin: 0;">✅ غير مدرج ضمن المحظورات المباشرة</h3>
                    <p style="margin: 0.5rem 0 0 0;">
                        المستحضر <strong>{item_to_check}</strong> لم يظهر في قائمة المحظورات الصريحة لـ <strong>{selected_company}</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if result.get("full_exclusions"):
                with st.expander("📋 قائمة المحظورات الكاملة للشركة"):
                    st.markdown(format_arabic_html(result["full_exclusions"]), unsafe_allow_html=True)

            if company_data:
                st.markdown('<div class="section-header">📋 قائمة ضوابط الصرف</div>', unsafe_allow_html=True)

                policies = company_data.get("policies", {})
                checklist_items = [
                    ("أقصى مدة للصرف", "⏱️ أقص مدة للصرف"),
                    ("صلاحية النموذج", "📅 صلاحية النموذج"),
                    ("الختم / إمضاء العميل", "✍️ الأختام والتوقيعات"),
                    ("التحمل", "💰 نسبة التحمل"),
                    ("صورة الكارنية", "🎴 صورة الكارنيه"),
                    ("صورة البطاقة", "🪪 صورة البطاقة الشخصية"),
                    ("التشخيص", "🩺 اشتراطات التشخيص"),
                ]

                for cat_ar, label in checklist_items:
                    if cat_ar in policies:
                        detail = policies[cat_ar].get("details", "")
                        st.markdown(f'<div class="checklist-item"><strong>{label}:</strong></div>', unsafe_allow_html=True)
                        st.markdown(format_arabic_html(detail), unsafe_allow_html=True)

                if "التواصل للموافقات" in policies:
                    st.markdown('<div class="section-header">📞 التواصل للموافقات</div>', unsafe_allow_html=True)
                    st.markdown(format_arabic_html(policies["التواصل للموافقات"].get("details", "")), unsafe_allow_html=True)

        elif check_btn:
            st.warning("يرجى اختيار شركة التأمين وإدخال اسم الصنف للفحص.")


# ===========================================================================
# PAGE 4: ANALYTICS
# ===========================================================================
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>تحليلات توزيع قواعد وسياسات التأمين الطبي</p>
    </div>
    """, unsafe_allow_html=True)

    kb = load_knowledge_base()

    col1, col2, col3 = st.columns(3)
    total_companies = len(kb)
    companies_with_exclusions = sum(1 for c in kb.values() if "المحظورات" in c.get("policies", {}))
    companies_with_contacts = sum(1 for c in kb.values() if "التواصل للموافقات" in c.get("policies", {}))

    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_companies}</div><div class="kpi-label">عدد الجهات والشركات</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{companies_with_exclusions}</div><div class="kpi-label">جهات تتضمن قائمة محظورات</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{companies_with_contacts}</div><div class="kpi-label">جهات توفر خطوط موافقات</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="section-header">تغطية البنود عبر الشركات</div>', unsafe_allow_html=True)
        cat_coverage = {}
        for data in kb.values():
            for cat in data.get("policies", {}):
                cat_coverage[cat] = cat_coverage.get(cat, 0) + 1

        coverage_df = pd.DataFrame(
            sorted(cat_coverage.items(), key=lambda x: x[1], reverse=True),
            columns=["البند", "عدد الشركات"],
        )

        fig2 = px.bar(coverage_df, x="عدد الشركات", y="البند", orientation="h",
                      color="عدد الشركات", color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#fafafa"), height=450, margin=dict(l=10,r=10,t=10,b=10),
                          showlegend=False, coloraxis_showscale=False,
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig2, use_container_width=True)

    with col_c2:
        claims_df = load_claims()
        rejected_claims = claims_df[claims_df["approved"] == 0]
        code_dist = rejected_claims["rejection_code"].value_counts().reset_index()
        code_dist.columns = ["الكود", "العدد"]
        st.markdown('<div class="section-header">توزيع أكواد الرفض (بيانات نموذجية)</div>', unsafe_allow_html=True)
        fig_s2 = px.pie(code_dist, values="العدد", names="الكود",
                        color_discrete_sequence=["#ef4444","#f59e0b","#3b82f6","#8b5cf6","#ec4899","#06b6d4"],
                        hole=0.4)
        fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#fafafa"),
                            height=450, margin=dict(l=10,r=10,t=10,b=10))
        fig_s2.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_s2, use_container_width=True)


# ===========================================================================
# PAGE 5: APPROVAL PREDICTION
# ===========================================================================
elif page == "🏥 Approval Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Approval Prediction (Synthetic Data Demo)</h1>
        <p>نموذج الذكاء الاصطناعي للتنبؤ بنسبة قبول المطالبات</p>
    </div>
    """, unsafe_allow_html=True)

    rules_df = load_insurance_rules()

    col_form, col_spacer, col_result = st.columns([1, 0.05, 1.2])

    with col_form:
        st.markdown('<div class="section-header">📋 بيانات المطالبة</div>', unsafe_allow_html=True)
        drug_options = sorted(rules_df["drug_name"].unique())
        drug_name = st.selectbox("Medication", drug_options, key="claim_drug")
        insurance_options = sorted(rules_df["insurance_name"].unique())
        insurance_name = st.selectbox("Insurance Company", insurance_options, key="claim_insurance")
        drug_diagnosis = rules_df[rules_df["drug_name"] == drug_name]["diagnosis_code"].iloc[0]
        diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", value=drug_diagnosis, key="claim_diag")
        patient_age = st.slider("Patient Age", 18, 90, 45, key="claim_age")

        st.markdown('<div class="section-header">📄 الإجراءات السريرية</div>', unsafe_allow_html=True)
        rule_match = rules_df[(rules_df["insurance_name"]==insurance_name) & (rules_df["drug_name"]==drug_name)]

        if not rule_match.empty:
            rule = rule_match.iloc[0]
            requires_step = rule.get("requires_step_therapy") in [True, "True"]
            requires_lab = rule.get("requires_lab_results") in [True, "True"]
            if requires_step:
                prior_drug = rule.get("required_prior_drug", "alternative")
                st.info(f"ℹ️ يتطلب العلاج المتدرج: تجربة **{prior_drug}** أولاً")
                prior_drug_tried = st.checkbox(f"تم تجربة دواء {prior_drug}", key="claim_prior")
            else:
                st.success("✅ لا يتطلب علاج متدرج"); prior_drug_tried = True
            if requires_lab:
                st.info("ℹ️ يتطلب نتائج تحاليل معملية")
                had_lab_results = st.checkbox("التحاليل المعملية متوفرة", key="claim_lab")
            else:
                st.success("✅ لا يتطلب تحاليل معملية"); had_lab_results = True
        else:
            prior_drug_tried = st.checkbox("Prior drug tried", key="cpf")
            had_lab_results = st.checkbox("Lab results available", key="clf")

        days_supply = st.select_slider("Days Supply", options=[14,30,60,90], value=30, key="claim_days")
        predict_btn = st.button("🔮 بدء التنبؤ", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            result = predict_approval(patient_age, drug_name, insurance_name, diagnosis_code,
                                      prior_drug_tried, had_lab_results, days_supply, rules_df)
            prob = result["probability"]
            risk = result["risk_level"]
            color = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#ef4444"}[risk]
            risk_class = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}[risk]

            st.markdown(f'<div style="text-align:center;padding:1rem;"><div style="font-size:3.5rem;font-weight:700;color:{color};">{prob:.0%}</div><div style="font-size:0.85rem;color:rgba(250,250,250,0.6);text-transform:uppercase;">احتمالية القبول</div><div style="margin-top:0.8rem;"><span class="{risk_class}">{risk} RISK</span></div></div>', unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={"suffix":"%","font":{"size":40}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":color,"thickness":0.3},"bgcolor":"#1a1f2e",
                "steps":[{"range":[0,50],"color":"rgba(239,68,68,0.15)"},{"range":[50,75],"color":"rgba(245,158,11,0.15)"},{"range":[75,100],"color":"rgba(16,185,129,0.15)"}]}))
            fig_g.update_layout(height=200,margin=dict(l=30,r=30,t=30,b=10),paper_bgcolor="rgba(0,0,0,0)",font={"color":"#fafafa"})
            st.plotly_chart(fig_g, use_container_width=True)

            if result["risk_factors"]:
                st.markdown('<div class="section-header">⚡ تحليلات المخاطر</div>', unsafe_allow_html=True)
                for f in result["risk_factors"]:
                    st.markdown(f'<div class="checklist-item">{f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-box">{result["recommendation"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:4rem 2rem;opacity:0.5;"><div style="font-size:4rem;">🔮</div><h3 style="color:#888;">أدخل بيانات المطالبة للتنبؤ</h3></div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 6: REJECTION ASSISTANT
# ===========================================================================
elif page == "🔍 Rejection Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Rejection Assistant</h1>
        <p>مساعد تفسير أكواد الرفض وصياغة خطابات الموافقة المسبقة</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔎 البحث في كود الرفض", "📝 إنشاء خطاب موافقة مسبقة (PA Letter)"])

    with tab1:
        col_input, col_result = st.columns([1, 1.5])
        with col_input:
            all_codes = get_all_codes()
            query_type = st.radio("البحث حسب:", ["Rejection Code", "Description"], horizontal=True)
            if query_type == "Rejection Code":
                code_options = [f"{c['code']} — {c['raw_message']}" for c in all_codes]
                selected = st.selectbox("اختر الكود", code_options)
                query = selected.split(" — ")[0] if selected else ""
            else:
                query = st.text_input("وصف سبب الرفض", placeholder="مثال: step therapy...")
            search_btn = st.button("🔍 تفسير الكود", type="primary", use_container_width=True)

        with col_result:
            if search_btn and query:
                result = interpret_rejection(query)
                if result["found"]:
                    if result["match_type"] == "fuzzy":
                        st.info(f"🔗 مطابقة تقريبية ({result['match_confidence']:.0%})")
                    st.markdown(f'<div class="kpi-card" style="text-align:left;margin-bottom:1rem;"><div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{result["code"]}</div><div style="color:rgba(250,250,250,0.8);margin-top:0.3rem;">{result["raw_message"]}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="status-box">{result["plain_explanation"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">✅ خطوات العمل المطلوبة</div>', unsafe_allow_html=True)
                    for i, action in enumerate(result["action_checklist"], 1):
                        st.checkbox(action, key=f"act_{i}_{result['code']}")
                else:
                    st.error(result.get("error", "لم يتم العثور على الكود."))

        st.markdown("---")
        ref_data = [{"Code": c["code"], "Message": c["raw_message"], "Explanation": c["plain_explanation"]} for c in all_codes]
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    with tab2:
        rules_df = load_insurance_rules()
        all_codes = get_all_codes()
        col_pa1, col_pa2 = st.columns(2)
        with col_pa1:
            pa_drug = st.selectbox("Medication", sorted(rules_df["drug_name"].unique()), key="pa_drug")
            pa_insurance = st.selectbox("Insurance", sorted(rules_df["insurance_name"].unique()), key="pa_ins")
            pa_age = st.number_input("Patient Age", 18, 90, 50, key="pa_age")
        with col_pa2:
            drug_diag = rules_df[rules_df["drug_name"]==pa_drug]["diagnosis_code"].iloc[0]
            pa_diag = st.text_input("Diagnosis Code", value=drug_diag, key="pa_diag")
            pa_rejection = st.selectbox("Rejection Code", [c["code"] for c in all_codes], key="pa_reject")
            pa_prior = st.checkbox("Prior drug tried", key="pa_pt"); pa_lab = st.checkbox("Lab results available", key="pa_la")

        if st.button("📄 توليد الخطاب", type="primary", use_container_width=True):
            letter = generate_pa_letter(pa_age, pa_drug, pa_insurance, pa_diag, pa_rejection, pa_prior, pa_lab)
            st.markdown(f'<div class="pa-letter">{letter}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ تحميل الخطاب (.txt)", letter, file_name=f"PA_{pa_drug}_{pa_insurance}.txt")


# ===========================================================================
# PAGE 7: MODEL PERFORMANCE
# ===========================================================================
elif page == "🧪 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Model Performance</h1>
        <p>مؤشرات وتقييم أداء نموذج التنبؤ لأغراض التقييم الأكاديمي</p>
    </div>
    """, unsafe_allow_html=True)

    metadata = load_model_metadata()
    if metadata is None:
        st.error("⚠️ Model not trained yet."); st.stop()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metadata["accuracy"]:.1%}</div><div class="kpi-label">دقة النموذج Accuracy</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metadata["f1_score"]:.1%}</div><div class="kpi-label">معامل F1-Score</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.3rem;">{metadata["model_name"]}</div><div class="kpi-label">خوارزمية النموذج</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix (مصفوفة الإرباك)</div>', unsafe_allow_html=True)
        cm = np.array(metadata["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred Rejected","Pred Approved"], y=["Actual Rejected","Actual Approved"],
            text=cm, texttemplate="%{text}", textfont={"size":20,"color":"white"},
            colorscale=[[0,"#1a1f2e"],[1,"#00D4AA"]], showscale=False))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"), height=350, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown('<div class="section-header">Feature Importance (أهمية المتغيرات)</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame(sorted(metadata["feature_importance"].items(), key=lambda x: x[1], reverse=True), columns=["Feature","Importance"])
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1a1f2e","#00D4AA"])
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"), height=350, margin=dict(l=10,r=10,t=10,b=10),
            showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Classification Report التفصيلي</div>', unsafe_allow_html=True)
    report = metadata["classification_report"]
    report_data = []
    for label in ["Rejected (0)", "Approved (1)"]:
        if label in report:
            r = report[label]
            report_data.append({"Class":label, "Precision":f"{r['precision']:.4f}", "Recall":f"{r['recall']:.4f}", "F1-Score":f"{r['f1-score']:.4f}", "Support":int(r["support"])})
    if "weighted avg" in report:
        r = report["weighted avg"]
        report_data.append({"Class":"Weighted Avg", "Precision":f"{r['precision']:.4f}", "Recall":f"{r['recall']:.4f}", "F1-Score":f"{r['f1-score']:.4f}", "Support":int(r["support"])})
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)
