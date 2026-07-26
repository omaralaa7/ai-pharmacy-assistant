"""
Phase 3 — Pharmacist Dashboard (Streamlit Application)
AI-Driven Insurance Chatbot (Diploma Project)

This is the main application — the pharmacist-facing interface that integrates:
  1. AI Chatbot — natural language queries about insurance policies
  2. Policy Directory — browse all 77 companies and 14 categories
  3. Dispensing Quick-Check — verify exclusions and requirements
  4. Analytics — visual insights across all insurers
  5. (Kept) Approval Prediction — ML model from synthetic data
  6. (Kept) Rejection Assistant — code interpreter
  7. Model Performance — ML evaluation metrics

Run with: streamlit run app.py
"""

import os
import sys
import json
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Insurance Assistant | مساعد التأمين الذكي",
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
    .kpi-value { font-size: 2rem; font-weight: 700; color: #00D4AA; line-height: 1.2; }
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
    .chat-bubble-bot {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px 12px 12px 4px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        direction: auto;
        white-space: pre-wrap;
        line-height: 1.8;
    }

    .checklist-item {
        background: rgba(0, 212, 170, 0.05);
        border-left: 3px solid #00D4AA;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    .policy-card {
        background: #1a1f2e;
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        white-space: pre-wrap;
        direction: auto;
        line-height: 1.8;
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
    }
    .approved-alert {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💊 AI Insurance Assistant")
    st.markdown("##### مساعد التأمين الذكي")
    st.markdown("---")

    page = st.radio(
        "Navigate",
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

    kb = load_knowledge_base()
    total_companies = len(kb)
    total_policies = sum(len(c.get("policies", {})) for c in kb.values())

    st.markdown(f"""
    <div class="kpi-card" style="margin-bottom: 0.8rem;">
        <div class="kpi-value">{total_companies}</div>
        <div class="kpi-label">Insurance Companies</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{total_policies}</div>
        <div class="kpi-label">Policy Rules</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("AI-Driven Insurance Chatbot")
    st.caption("Diploma Project — Real Egyptian Data")


# ===========================================================================
# PAGE 1: AI CHATBOT
# ===========================================================================
if page == "💬 AI Chatbot":
    st.markdown("""
    <div class="main-header">
        <h1>💬 AI Insurance Assistant</h1>
        <p>اسأل عن سياسات أي شركة تأمين باللغة العربية أو الإنجليزية</p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions
    with st.expander("💡 Example Questions / أمثلة على الأسئلة"):
        st.markdown("""
        - **ما هي محظورات يونايتد؟** — Shows United's excluded items list
        - **أقصى مدة صرف لشركة ويبكو** — Shows WEPCO's max dispensing duration
        - **رقم تليفون موافقات دريم مشرق** — Shows Dream Mashreq approval contacts
        - **هل يشترط ختم لشركة جلوبميد؟** — Shows GlobeMed stamp requirements
        - **What is the copay for ALICO?** — Shows ALICO's co-payment rules
        - **يونايتد** — Shows all policy categories for United
        """)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask about any insurance company...",
            placeholder="مثال: ما هي محظورات يونايتد؟ / What are the exclusions for United?",
            key="chat_input",
            label_visibility="collapsed",
        )
    with col_btn:
        send_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

    if send_btn and user_question:
        result = chat_query(user_question)
        st.session_state.chat_history.append({"question": user_question, "result": result})

    # Display chat history (newest first)
    for entry in reversed(st.session_state.chat_history):
        q = entry["question"]
        r = entry["result"]

        st.markdown(f'<div class="chat-bubble-user">🧑‍⚕️ {q}</div>', unsafe_allow_html=True)

        if r.get("found"):
            method_badge = {"direct": "✅ Direct Match", "fuzzy": "🔗 Fuzzy Match", "tfidf": "🔎 AI Search"}.get(r.get("method", ""), "")
            confidence = r.get("confidence", 0)

            header = f"**{r.get('company_name', '')}** — {r.get('category', '')} ({r.get('category_en', '')})"
            st.markdown(f'<div class="chat-bubble-bot">{method_badge} (Confidence: {confidence:.0%})\n\n{header}\n\n{r.get("answer", "")}</div>', unsafe_allow_html=True)

            if r.get("notes"):
                st.caption(f"📝 {r['notes']}")

            # Show other TF-IDF results if available
            if r.get("other_results"):
                with st.expander("More related results"):
                    for other in r["other_results"]:
                        st.markdown(f"**{other['company_name']}** — {other['category']}")
                        st.markdown(other["answer"][:200] + "...")
        else:
            error_msg = r.get("error", "No results found.")
            st.markdown(f'<div class="chat-bubble-bot">❌ {error_msg}</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ===========================================================================
# PAGE 2: POLICY DIRECTORY
# ===========================================================================
elif page == "📖 Policy Directory":
    st.markdown("""
    <div class="main-header">
        <h1>📖 Policy Directory</h1>
        <p>تصفح سياسات جميع شركات التأمين — Browse all insurance company policies</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']} ({c['id']})": c["key"] for c in companies}

    selected_label = st.selectbox(
        "Select Insurance Company / اختر شركة التأمين",
        options=list(company_options.keys()),
    )

    if selected_label:
        selected_key = company_options[selected_label]
        company_data = get_company_policies(selected_key)

        if company_data:
            # Company header
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size:1.3rem;">{company_data.get('company_name_ar', '')}</div>
                    <div class="kpi-label">Arabic Name</div>
                </div>
                """, unsafe_allow_html=True)
            with col_info2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="font-size:1.3rem;">{company_data.get('company_name_en', '').upper() or 'N/A'}</div>
                    <div class="kpi-label">English Name</div>
                </div>
                """, unsafe_allow_html=True)
            with col_info3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{len(company_data.get('policies', {}))}</div>
                    <div class="kpi-label">Policy Categories</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Category filter
            all_cats = list(company_data.get("policies", {}).keys())
            filter_cat = st.multiselect("Filter by category (optional)", all_cats)
            cats_to_show = filter_cat if filter_cat else all_cats

            # Display policies
            for category in cats_to_show:
                policy = company_data["policies"][category]
                cat_en = policy.get("category_en", "")

                with st.expander(f"📋 {category} — {cat_en}", expanded=len(cats_to_show) <= 3):
                    st.markdown(f'<div class="policy-card">{policy.get("details", "No details available.")}</div>', unsafe_allow_html=True)
                    if policy.get("notes"):
                        st.info(f"📝 Notes: {policy['notes']}")


# ===========================================================================
# PAGE 3: DISPENSING QUICK-CHECK
# ===========================================================================
elif page == "⚡ Dispensing Check":
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Dispensing Quick-Check</h1>
        <p>تحقق من محظورات ومتطلبات الصرف — Verify exclusions and dispensing requirements</p>
    </div>
    """, unsafe_allow_html=True)

    companies = get_all_companies()
    company_options = {f"{c['name']}": c["key"] for c in companies}

    col_check1, col_check2 = st.columns([1, 1.5])

    with col_check1:
        st.markdown('<div class="section-header">🔍 Check Details</div>', unsafe_allow_html=True)

        selected_company = st.selectbox(
            "Insurance Company / شركة التأمين",
            options=list(company_options.keys()),
            key="check_company",
        )

        item_to_check = st.text_input(
            "Item / Medication Name",
            placeholder="e.g., sun screen, فيتامينات, clexan...",
            key="check_item",
        )

        check_btn = st.button("⚡ Check", type="primary", use_container_width=True)

    with col_check2:
        if check_btn and selected_company and item_to_check:
            company_key = company_options[selected_company]
            company_data = get_company_policies(company_key)

            # Check exclusions
            result = check_exclusions(company_key, item_to_check)

            if result.get("is_excluded"):
                st.markdown(f"""
                <div class="excluded-alert">
                    <h3 style="color: #ef4444; margin: 0;">🚫 EXCLUDED / محظور</h3>
                    <p style="margin: 0.5rem 0 0 0;">
                        <strong>{item_to_check}</strong> appears to be in <strong>{selected_company}</strong>'s exclusion list.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="approved-alert">
                    <h3 style="color: #10b981; margin: 0;">✅ NOT IN EXCLUSION LIST</h3>
                    <p style="margin: 0.5rem 0 0 0;">
                        <strong>{item_to_check}</strong> was not found in <strong>{selected_company}</strong>'s exclusion list.
                        <br><em>Note: Always verify with the full policy details.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Show exclusions list
            if result.get("full_exclusions"):
                with st.expander("📋 Full Exclusion List / قائمة المحظورات الكاملة"):
                    st.markdown(f'<div class="policy-card">{result["full_exclusions"]}</div>', unsafe_allow_html=True)

            # Show dispensing summary
            if company_data:
                st.markdown('<div class="section-header">📋 Dispensing Checklist</div>', unsafe_allow_html=True)

                policies = company_data.get("policies", {})
                checklist_items = [
                    ("أقصى مدة للصرف", "⏱️ Max Duration", "Maximum Dispensing Duration"),
                    ("صلاحية النموذج", "📅 Form Validity", "Form Validity Period"),
                    ("الختم / إمضاء العميل", "✍️ Stamp/Signature", "Stamp Requirements"),
                    ("التحمل", "💰 Co-payment", "Co-payment / Deductible"),
                    ("صورة الكارنية", "🎴 Card Copy", "Insurance Card Requirement"),
                    ("صورة البطاقة", "🪪 ID Copy", "National ID Requirement"),
                    ("التشخيص", "🩺 Diagnosis", "Diagnosis Requirements"),
                ]

                for cat_ar, icon_label, cat_en in checklist_items:
                    if cat_ar in policies:
                        detail = policies[cat_ar].get("details", "—")
                        st.markdown(f'<div class="checklist-item"><strong>{icon_label}:</strong> {detail}</div>', unsafe_allow_html=True)

                # Contact info
                if "التواصل للموافقات" in policies:
                    st.markdown('<div class="section-header">📞 Approval Contacts</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="policy-card">{policies["التواصل للموافقات"].get("details", "")}</div>', unsafe_allow_html=True)

        elif check_btn:
            st.warning("Please select a company and enter an item name.")


# ===========================================================================
# PAGE 4: ANALYTICS
# ===========================================================================
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Insurance Analytics</h1>
        <p>Visual insights across all 77 Egyptian insurance providers</p>
    </div>
    """, unsafe_allow_html=True)

    kb = load_knowledge_base()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_companies = len(kb)
    total_policies = sum(len(c.get("policies", {})) for c in kb.values())
    companies_with_exclusions = sum(1 for c in kb.values() if "المحظورات" in c.get("policies", {}))
    companies_with_contacts = sum(1 for c in kb.values() if "التواصل للموافقات" in c.get("policies", {}))

    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_companies}</div><div class="kpi-label">Companies</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_policies}</div><div class="kpi-label">Policy Rules</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{companies_with_exclusions}</div><div class="kpi-label">With Exclusion Lists</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{companies_with_contacts}</div><div class="kpi-label">With Contact Info</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1: Categories per company
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="section-header">Policy Categories per Company</div>', unsafe_allow_html=True)
        cat_counts = []
        for key, data in kb.items():
            cat_counts.append({
                "Company": data.get("company_name_ar", key)[:20],
                "Categories": len(data.get("policies", {})),
            })
        cat_df = pd.DataFrame(cat_counts).sort_values("Categories", ascending=True).tail(20)

        fig1 = px.bar(cat_df, x="Categories", y="Company", orientation="h",
                      color="Categories", color_continuous_scale=["#1a1f2e", "#00D4AA"])
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#fafafa"), height=500, margin=dict(l=10,r=10,t=10,b=10),
                          showlegend=False, coloraxis_showscale=False,
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig1, use_container_width=True)

    with col_c2:
        st.markdown('<div class="section-header">Category Coverage Across All Companies</div>', unsafe_allow_html=True)
        cat_coverage = {}
        for data in kb.values():
            for cat in data.get("policies", {}):
                cat_coverage[cat] = cat_coverage.get(cat, 0) + 1

        coverage_df = pd.DataFrame(
            sorted(cat_coverage.items(), key=lambda x: x[1], reverse=True),
            columns=["Category", "Companies"],
        )

        fig2 = px.bar(coverage_df, x="Companies", y="Category", orientation="h",
                      color="Companies", color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#fafafa"), height=500, margin=dict(l=10,r=10,t=10,b=10),
                          showlegend=False, coloraxis_showscale=False,
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 2: Max dispensing duration distribution
    st.markdown('<div class="section-header">Maximum Dispensing Duration Distribution</div>', unsafe_allow_html=True)
    duration_data = []
    for key, data in kb.items():
        pol = data.get("policies", {}).get("أقصى مدة للصرف", {})
        detail = pol.get("details", "")
        if detail:
            duration_data.append({
                "Company": data.get("company_name_ar", key)[:25],
                "Duration": detail[:50],
            })

    if duration_data:
        dur_df = pd.DataFrame(duration_data)
        st.dataframe(dur_df, use_container_width=True, hide_index=True, height=300)

    # Also show synthetic claims analytics
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Synthetic Claims Analytics (ML Training Data)</div>', unsafe_allow_html=True)

    claims_df = load_claims()
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        approval_by_insurer = claims_df.groupby("insurance_name")["approved"].mean().reset_index()
        approval_by_insurer.columns = ["Insurance", "Approval Rate"]
        fig_s1 = px.bar(approval_by_insurer.sort_values("Approval Rate"), x="Approval Rate", y="Insurance",
                        orientation="h", color="Approval Rate",
                        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"], range_color=[0,1])
        fig_s1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#fafafa"), height=300, margin=dict(l=10,r=10,t=10,b=10),
                            showlegend=False, coloraxis_showscale=False,
                            xaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_s1, use_container_width=True)

    with col_s2:
        rejected_claims = claims_df[claims_df["approved"] == 0]
        code_dist = rejected_claims["rejection_code"].value_counts().reset_index()
        code_dist.columns = ["Code", "Count"]
        fig_s2 = px.pie(code_dist, values="Count", names="Code",
                        color_discrete_sequence=["#ef4444","#f59e0b","#3b82f6","#8b5cf6","#ec4899","#06b6d4"],
                        hole=0.4)
        fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#fafafa"),
                            height=300, margin=dict(l=10,r=10,t=10,b=10))
        fig_s2.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig_s2, use_container_width=True)


# ===========================================================================
# PAGE 5: APPROVAL PREDICTION (kept from original)
# ===========================================================================
elif page == "🏥 Approval Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Approval Prediction (Synthetic Data Demo)</h1>
        <p>ML model predicting insurance claim approval likelihood</p>
    </div>
    """, unsafe_allow_html=True)

    rules_df = load_insurance_rules()

    col_form, col_spacer, col_result = st.columns([1, 0.05, 1.2])

    with col_form:
        st.markdown('<div class="section-header">📋 Claim Details</div>', unsafe_allow_html=True)
        drug_options = sorted(rules_df["drug_name"].unique())
        drug_name = st.selectbox("Medication", drug_options, key="claim_drug")
        insurance_options = sorted(rules_df["insurance_name"].unique())
        insurance_name = st.selectbox("Insurance Company", insurance_options, key="claim_insurance")
        drug_diagnosis = rules_df[rules_df["drug_name"] == drug_name]["diagnosis_code"].iloc[0]
        diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", value=drug_diagnosis, key="claim_diag")
        patient_age = st.slider("Patient Age", 18, 90, 45, key="claim_age")

        st.markdown('<div class="section-header">📄 Clinical Information</div>', unsafe_allow_html=True)
        rule_match = rules_df[(rules_df["insurance_name"]==insurance_name) & (rules_df["drug_name"]==drug_name)]

        if not rule_match.empty:
            rule = rule_match.iloc[0]
            requires_step = rule.get("requires_step_therapy") in [True, "True"]
            requires_lab = rule.get("requires_lab_results") in [True, "True"]
            if requires_step:
                prior_drug = rule.get("required_prior_drug", "alternative")
                st.info(f"ℹ️ Requires step therapy: try **{prior_drug}** first")
                prior_drug_tried = st.checkbox(f"Patient has tried {prior_drug}", key="claim_prior")
            else:
                st.success("✅ No step therapy required"); prior_drug_tried = True
            if requires_lab:
                st.info("ℹ️ Lab results required")
                had_lab_results = st.checkbox("Lab results available", key="claim_lab")
            else:
                st.success("✅ No lab results required"); had_lab_results = True
        else:
            prior_drug_tried = st.checkbox("Prior drug tried", key="cpf")
            had_lab_results = st.checkbox("Lab results available", key="clf")

        days_supply = st.select_slider("Days Supply", options=[14,30,60,90], value=30, key="claim_days")
        predict_btn = st.button("🔮 Predict Approval", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            result = predict_approval(patient_age, drug_name, insurance_name, diagnosis_code,
                                      prior_drug_tried, had_lab_results, days_supply, rules_df)
            prob = result["probability"]
            risk = result["risk_level"]
            color = {"LOW":"#10b981","MEDIUM":"#f59e0b","HIGH":"#ef4444"}[risk]
            risk_class = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}[risk]

            st.markdown(f'<div style="text-align:center;padding:1rem;"><div style="font-size:3.5rem;font-weight:700;color:{color};">{prob:.0%}</div><div style="font-size:0.85rem;color:rgba(250,250,250,0.6);text-transform:uppercase;">Approval Probability</div><div style="margin-top:0.8rem;"><span class="{risk_class}">{risk} RISK</span></div></div>', unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, number={"suffix":"%","font":{"size":40}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":color,"thickness":0.3},"bgcolor":"#1a1f2e",
                "steps":[{"range":[0,50],"color":"rgba(239,68,68,0.15)"},{"range":[50,75],"color":"rgba(245,158,11,0.15)"},{"range":[75,100],"color":"rgba(16,185,129,0.15)"}]}))
            fig_g.update_layout(height=200,margin=dict(l=30,r=30,t=30,b=10),paper_bgcolor="rgba(0,0,0,0)",font={"color":"#fafafa"})
            st.plotly_chart(fig_g, use_container_width=True)

            if result["risk_factors"]:
                st.markdown('<div class="section-header">⚡ Risk Analysis</div>', unsafe_allow_html=True)
                for f in result["risk_factors"]:
                    st.markdown(f'<div class="checklist-item">{f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-box">{result["recommendation"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;padding:4rem 2rem;opacity:0.5;"><div style="font-size:4rem;">🔮</div><h3 style="color:#888;">Enter Claim Details</h3></div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 6: REJECTION ASSISTANT (kept from original)
# ===========================================================================
elif page == "🔍 Rejection Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Rejection Assistant</h1>
        <p>Interpret rejection codes and get actionable next steps</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔎 Look Up Code", "📝 Generate PA Letter"])

    with tab1:
        col_input, col_result = st.columns([1, 1.5])
        with col_input:
            all_codes = get_all_codes()
            query_type = st.radio("Search by:", ["Rejection Code", "Description"], horizontal=True)
            if query_type == "Rejection Code":
                code_options = [f"{c['code']} — {c['raw_message']}" for c in all_codes]
                selected = st.selectbox("Select code", code_options)
                query = selected.split(" — ")[0] if selected else ""
            else:
                query = st.text_input("Describe the rejection", placeholder="e.g., step therapy...")
            search_btn = st.button("🔍 Interpret", type="primary", use_container_width=True)

        with col_result:
            if search_btn and query:
                result = interpret_rejection(query)
                if result["found"]:
                    if result["match_type"] == "fuzzy":
                        st.info(f"🔗 Fuzzy match ({result['match_confidence']:.0%})")
                    st.markdown(f'<div class="kpi-card" style="text-align:left;margin-bottom:1rem;"><div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{result["code"]}</div><div style="color:rgba(250,250,250,0.8);margin-top:0.3rem;">{result["raw_message"]}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="status-box">{result["plain_explanation"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">✅ Action Checklist</div>', unsafe_allow_html=True)
                    for i, action in enumerate(result["action_checklist"], 1):
                        st.checkbox(action, key=f"act_{i}_{result['code']}")
                else:
                    st.error(result.get("error", "No match found."))

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

        if st.button("📄 Generate Letter", type="primary", use_container_width=True):
            letter = generate_pa_letter(pa_age, pa_drug, pa_insurance, pa_diag, pa_rejection, pa_prior, pa_lab)
            st.markdown(f'<div class="pa-letter">{letter}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Download (.txt)", letter, file_name=f"PA_{pa_drug}_{pa_insurance}.txt")


# ===========================================================================
# PAGE 7: MODEL PERFORMANCE
# ===========================================================================
elif page == "🧪 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 Model Performance</h1>
        <p>ML model evaluation metrics for academic validation</p>
    </div>
    """, unsafe_allow_html=True)

    metadata = load_model_metadata()
    if metadata is None:
        st.error("⚠️ Model not trained yet. Run `python models/train_model.py` first."); st.stop()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metadata["accuracy"]:.1%}</div><div class="kpi-label">Accuracy</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{metadata["f1_score"]:.1%}</div><div class="kpi-label">F1 Score</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.5rem;">{metadata["model_name"]}</div><div class="kpi-label">Best Model</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(metadata["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred Rejected","Pred Approved"], y=["Actual Rejected","Actual Approved"],
            text=cm, texttemplate="%{text}", textfont={"size":20,"color":"white"},
            colorscale=[[0,"#1a1f2e"],[1,"#00D4AA"]], showscale=False))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"), height=350, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame(sorted(metadata["feature_importance"].items(), key=lambda x: x[1], reverse=True), columns=["Feature","Importance"])
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1a1f2e","#00D4AA"])
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"), height=350, margin=dict(l=10,r=10,t=10,b=10),
            showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
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
