"""
Medical Insurance Policy Assistant (Streamlit Application)
Pharmacist Decision Support System

Professional policy query and verification system running on real Egyptian insurance policy data:
  1. Policy Search Engine — natural language search across company rules
  2. Policy Directory — browse all company policy categories
  3. Dispensing Quick-Check — verify exclusions and requirements
  4. Approval Compliance Check — rule compliance evaluation
  5. PA Letter Generator — generate official prior authorization letters

Run locally with: streamlit run app.py
"""

import os
import sys
import json
import html
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models.chatbot_engine import (
    chat_query,
    get_all_companies,
    get_company_policies,
    get_all_categories,
    check_exclusions,
)
from models.real_approval_engine import (
    evaluate_real_approval,
    generate_real_pa_letter,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_DIR, "data")


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
    cleaning punctuation artifacts and formatting bullets cleanly.
    """
    if not text:
        return '<div dir="rtl" style="direction: rtl; text-align: right; color: rgba(250,250,250,0.5); font-style: italic;">لا توجد تفاصيل متاحة</div>'

    lines = [line.strip() for line in str(text).split("\n") if line.strip()]
    formatted_elements = []

    for line in lines:
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
    page_title="Medical Insurance Assistant | مساعد التأمين الطبي",
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
        font-family: 'Courier New', monospace; font-size: 0.9rem;
        white-space: pre-wrap; line-height: 1.8;
        direction: rtl; text-align: right;
    }

    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', 'Inter', sans-serif;
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
    st.markdown("## 💊 Insurance Assistant")
    st.markdown("##### نظام مساعد التأمين الطبي")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "💬 Policy Search",
            "📖 Policy Directory",
            "⚡ Dispensing Check",
            "🏥 Approval Check",
            "📝 PA Letter Generator",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Medical Insurance Management System")
    st.caption("Pharmacy Decision Support System")


# ===========================================================================
# PAGE 1: POLICY SEARCH
# ===========================================================================
if page == "💬 Policy Search":
    st.markdown("""
    <div class="main-header">
        <h1>💬 Policy Search Engine</h1>
        <p>نظام الاستعلام الرقمي عن سياسات وقواعد صرف التأمين الطبي</p>
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
        <p>دليل وسجل سياسات التأمين الطبي لجميع الشركات والهيئات</p>
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
        <p>نظام التحقق السريع من المحظورات وضوابط الصرف قبل صرف الدواء</p>
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
                    ("أقصى مدة للصرف", "⏱️ أقصى مدة للصرف"),
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
# PAGE 4: APPROVAL CHECK
# ===========================================================================
elif page == "🏥 Approval Check":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Approval & Compliance Check</h1>
        <p>تقييم نسبة قبول المطالبة والتوافق مع القواعد الحقيقية لشركة التأمين</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']}": c["key"] for c in companies}

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown('<div class="section-header">📋 بيانات الروشتة والمطالبة</div>', unsafe_allow_html=True)

        eval_company_label = st.selectbox(
            "شركة التأمين / الجهة الضامنة",
            options=list(company_options.keys()),
            key="eval_company",
        )

        eval_item = st.text_input(
            "اسم المستحضر المطلوب",
            value="Clexane 4000",
            key="eval_item",
        )

        eval_days = st.select_slider(
            "فترة الصرف المطلوبة (بالأيام)",
            options=[7, 14, 21, 30, 60, 90],
            value=30,
            key="eval_days",
        )

        has_stamp = st.checkbox("الروشتة مختومة بختم الطبيب / المستشفى معتمد", value=True, key="eval_stamp")
        has_diag = st.checkbox("التشخيص الطبي مدون على الروشتة", value=True, key="eval_diag")

        eval_btn = st.button("🔮 تقييم نسبة القبول", type="primary", use_container_width=True)

    with col_result:
        if eval_btn and eval_company_label:
            comp_key = company_options[eval_company_label]
            eval_res = evaluate_real_approval(comp_key, eval_item, eval_days, has_stamp, has_diag)

            prob = eval_res["score"]
            score_pct = eval_res["score_percent"]
            risk = eval_res["risk_level"]

            color = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}[risk]
            risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[risk]

            st.markdown(f'<div style="text-align:center;padding:1rem;"><div style="font-size:3.5rem;font-weight:700;color:{color};">{score_pct}%</div><div style="font-size:0.85rem;color:rgba(250,250,250,0.6);text-transform:uppercase;">نسبة التوافق وقبول الصرف</div><div style="margin-top:0.8rem;"><span class="{risk_class}">{risk} RISK</span></div></div>', unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_pct,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#1a1f2e",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                        {"range": [50, 75], "color": "rgba(245,158,11,0.15)"},
                        {"range": [75, 100], "color": "rgba(16,185,129,0.15)"},
                    ],
                },
            ))
            fig_g.update_layout(height=200, margin=dict(l=30, r=30, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"})
            st.plotly_chart(fig_g, use_container_width=True)

            if eval_res["risk_factors"]:
                st.markdown('<div class="section-header">⚡ تحليلات المخاطر وعدم التوافق</div>', unsafe_allow_html=True)
                for factor in eval_res["risk_factors"]:
                    st.markdown(f'<div class="checklist-item">{factor}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="status-box" dir="rtl" style="text-align:right;">{eval_res["recommendation"]}</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 5: PA LETTER GENERATOR
# ===========================================================================
elif page == "📝 PA Letter Generator":
    st.markdown("""
    <div class="main-header">
        <h1>📝 PA Letter Generator</h1>
        <p>صياغة خطاب رسمي لطلب الموافقة المسبقة باستعمال أرقام وبيانات الشركة الحقيقية</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']}": c["key"] for c in companies}

    col_pa1, col_pa2 = st.columns([1, 1.2])

    with col_pa1:
        st.markdown('<div class="section-header">📋 بيانات الخطاب</div>', unsafe_allow_html=True)

        pa_company_label = st.selectbox(
            "شركة التأمين / الجهة الضامنة",
            options=list(company_options.keys()),
            key="pa_company",
        )

        patient_name = st.text_input("اسم المريض/المريضة", value="سارة محمود السيد", key="pa_patient")
        pa_item = st.text_input("اسم المستحضر / العلاج", value="Clexane 4000 IU", key="pa_item")
        pa_diag = st.text_input("التشخيص الطبي", value="متابعة حمل خطِر / الوقاية من التجلطات", key="pa_diag")
        pharmacist_notes = st.text_area("ملاحظات الصيدلي / حالة طارئة", value="يرجى الموافقة لضرورة الحالة الطبية العاجلة للمريضة.", key="pa_notes")

        gen_btn = st.button("📄 توليد الخطاب الرسمى", type="primary", use_container_width=True)

    with col_pa2:
        if gen_btn and pa_company_label:
            comp_key = company_options[pa_company_label]
            letter_text = generate_real_pa_letter(comp_key, patient_name, pa_item, pa_diag, pharmacist_notes)

            st.markdown('<div class="section-header">📄 نص الخطاب المولد</div>', unsafe_allow_html=True)
            st.markdown(format_arabic_html(letter_text), unsafe_allow_html=True)

            st.download_button(
                "⬇️ تحميل الخطاب (.txt)",
                data=letter_text,
                file_name=f"PA_Letter_{patient_name}.txt",
                mime="text/plain",
                use_container_width=True,
            )
