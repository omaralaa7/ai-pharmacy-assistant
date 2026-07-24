"""
Phase 3 — Pharmacist Dashboard (Streamlit Application)
AI-Driven Prior Authorization Assistant (Diploma Project)

This is the main application file — the pharmacist-facing interface that
ties together the ML prediction model, the rejection code interpreter,
and the analytics visualization.

Three main pages:
  1. New Claim Check — predict approval likelihood before submitting
  2. Rejection Assistant — interpret rejection codes + generate PA letters
  3. Analytics Dashboard — visualize claim patterns and insights

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add project root to path so we can import our modules
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Auto-train model if not found (needed for Streamlit Cloud fresh deploys)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data loading (cached so it only loads once)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_DIR, "data")


@st.cache_data
def load_claims():
    """Load the claims dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))


@st.cache_data
def load_insurance_rules():
    """Load the insurance rules dataset."""
    return pd.read_csv(os.path.join(DATA_DIR, "insurance_rules.csv"))


@st.cache_data
def load_model_metadata():
    """Load the saved model's evaluation metadata."""
    metadata_path = os.path.join(PROJECT_DIR, "models", "saved", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Prior Authorization Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0d9488 0%, #059669 50%, #047857 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.15);
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* KPI card styling */
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
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: rgba(250, 250, 250, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Risk level badges */
    .risk-low {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Probability gauge */
    .gauge-container {
        text-align: center;
        padding: 1rem;
    }
    .gauge-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .gauge-label {
        font-size: 0.85rem;
        color: rgba(250,250,250,0.6);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Checklist items */
    .checklist-item {
        background: rgba(0, 212, 170, 0.05);
        border-left: 3px solid #00D4AA;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* PA Letter box */
    .pa-letter {
        background: #1a1f2e;
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151b28 100%);
    }

    /* Status box */
    .status-box {
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00D4AA;
        border-bottom: 2px solid rgba(0, 212, 170, 0.2);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💊 AI PA Assistant")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏥 New Claim Check", "🔍 Rejection Assistant", "📊 Analytics Dashboard", "🧪 Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick stats in sidebar
    claims_df = load_claims()
    total_claims = len(claims_df)
    approval_rate = claims_df["approved"].mean()

    st.markdown(f"""
    <div class="kpi-card" style="margin-bottom: 0.8rem;">
        <div class="kpi-value">{total_claims:,}</div>
        <div class="kpi-label">Total Claims</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{approval_rate:.0%}</div>
        <div class="kpi-label">Approval Rate</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("AI-Driven Prior Authorization Assistant")
    st.caption("Diploma Project — Proof of Concept")


# ===========================================================================
# PAGE 1: NEW CLAIM CHECK
# ===========================================================================
if page == "🏥 New Claim Check":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 New Claim Check</h1>
        <p>Predict insurance approval likelihood before submitting a claim</p>
    </div>
    """, unsafe_allow_html=True)

    rules_df = load_insurance_rules()

    # --- Input Form ---
    col_form, col_spacer, col_result = st.columns([1, 0.05, 1.2])

    with col_form:
        st.markdown('<div class="section-header">📋 Claim Details</div>', unsafe_allow_html=True)

        # Drug selection
        drug_options = sorted(rules_df["drug_name"].unique())
        drug_name = st.selectbox("Medication", drug_options, key="claim_drug")

        # Insurance selection
        insurance_options = sorted(rules_df["insurance_name"].unique())
        insurance_name = st.selectbox("Insurance Company", insurance_options, key="claim_insurance")

        # Auto-fill diagnosis code based on drug
        drug_diagnosis = rules_df[rules_df["drug_name"] == drug_name]["diagnosis_code"].iloc[0]
        diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", value=drug_diagnosis, key="claim_diag")

        # Patient age
        patient_age = st.slider("Patient Age", 18, 90, 45, key="claim_age")

        # Look up if this drug/insurance combo requires step therapy
        rule_match = rules_df[
            (rules_df["insurance_name"] == insurance_name) &
            (rules_df["drug_name"] == drug_name)
        ]

        st.markdown('<div class="section-header">📄 Clinical Information</div>', unsafe_allow_html=True)

        if not rule_match.empty:
            rule = rule_match.iloc[0]
            requires_step = rule.get("requires_step_therapy") in [True, "True"]
            requires_lab = rule.get("requires_lab_results") in [True, "True"]
            max_days = int(rule.get("max_days_supply_without_pa", 90))

            if requires_step:
                prior_drug = rule.get("required_prior_drug", "alternative")
                st.info(f"ℹ️ {insurance_name} requires step therapy: patient must try **{prior_drug}** first")
                prior_drug_tried = st.checkbox(f"Patient has tried {prior_drug}", key="claim_prior")
            else:
                st.success("✅ No step therapy required for this combination")
                prior_drug_tried = True

            if requires_lab:
                st.info("ℹ️ Lab results are required for this medication")
                had_lab_results = st.checkbox("Lab results available (within 3 months)", key="claim_lab")
            else:
                st.success("✅ No lab results required")
                had_lab_results = True

            st.caption(f"Max days supply without PA: **{max_days}** days")
        else:
            prior_drug_tried = st.checkbox("Prior alternative drug tried", key="claim_prior_fallback")
            had_lab_results = st.checkbox("Lab results available", key="claim_lab_fallback")
            max_days = 90

        days_supply = st.select_slider(
            "Days Supply Requested",
            options=[14, 30, 60, 90],
            value=30,
            key="claim_days",
        )

        predict_btn = st.button("🔮 Predict Approval", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            with st.spinner("Analyzing claim..."):
                result = predict_approval(
                    patient_age=patient_age,
                    drug_name=drug_name,
                    insurance_name=insurance_name,
                    diagnosis_code=diagnosis_code,
                    prior_drug_tried=prior_drug_tried,
                    had_lab_results=had_lab_results,
                    days_supply=days_supply,
                    insurance_rules_df=rules_df,
                )

            prob = result["probability"]
            risk_level = result["risk_level"]

            # Color based on risk
            if risk_level == "LOW":
                color = "#10b981"
                risk_class = "risk-low"
            elif risk_level == "MEDIUM":
                color = "#f59e0b"
                risk_class = "risk-medium"
            else:
                color = "#ef4444"
                risk_class = "risk-high"

            # --- Probability display ---
            st.markdown(f"""
            <div class="gauge-container">
                <div class="gauge-value" style="color: {color};">{prob:.0%}</div>
                <div class="gauge-label">Approval Probability</div>
                <div style="margin-top: 0.8rem;">
                    <span class="{risk_class}">{risk_level} RISK</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- Visual probability bar ---
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#1a1f2e",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [50, 75], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [75, 100], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#fafafa"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- Risk factors ---
            st.markdown('<div class="section-header">⚡ Risk Analysis</div>', unsafe_allow_html=True)

            if result["risk_factors"]:
                for factor in result["risk_factors"]:
                    st.markdown(f'<div class="checklist-item">{factor}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="checklist-item">✅ No specific risk factors identified</div>',
                           unsafe_allow_html=True)

            # --- Recommendation ---
            st.markdown('<div class="section-header">💡 Recommendation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-box">{result["recommendation"]}</div>',
                       unsafe_allow_html=True)

        else:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; opacity: 0.5;">
                <div style="font-size: 4rem;">🔮</div>
                <h3 style="color: #888; margin-top: 1rem;">Enter Claim Details</h3>
                <p style="color: #666;">Fill in the form and click <strong>Predict Approval</strong>
                to see the AI-powered analysis</p>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: REJECTION ASSISTANT
# ===========================================================================
elif page == "🔍 Rejection Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Rejection Assistant</h1>
        <p>Interpret rejection codes and get actionable next steps</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔎 Look Up Code", "📝 Generate PA Letter"])

    # --- Tab 1: Code Lookup ---
    with tab1:
        col_input, col_result = st.columns([1, 1.5])

        with col_input:
            st.markdown('<div class="section-header">Enter Rejection Code</div>', unsafe_allow_html=True)

            # Show available codes for reference
            all_codes = get_all_codes()

            query_type = st.radio(
                "Search by:",
                ["Rejection Code", "Description / Keyword"],
                horizontal=True,
            )

            if query_type == "Rejection Code":
                code_options = [f"{c['code']} — {c['raw_message']}" for c in all_codes]
                selected = st.selectbox("Select a rejection code", code_options)
                query = selected.split(" — ")[0] if selected else ""
            else:
                query = st.text_input(
                    "Describe the rejection",
                    placeholder="e.g., step therapy, lab results, quantity limit...",
                )

            search_btn = st.button("🔍 Interpret", type="primary", use_container_width=True)

        with col_result:
            if search_btn and query:
                result = interpret_rejection(query)

                if result["found"]:
                    # Match info
                    if result["match_type"] == "fuzzy":
                        st.info(f"🔗 Fuzzy match (confidence: {result['match_confidence']:.0%})")

                    # Code and message
                    st.markdown(f"""
                    <div class="kpi-card" style="text-align: left; margin-bottom: 1rem;">
                        <div style="font-size: 1.4rem; font-weight: 700; color: #ef4444;">
                            {result['code']}
                        </div>
                        <div style="color: rgba(250,250,250,0.8); margin-top: 0.3rem;">
                            {result['raw_message']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Plain explanation
                    st.markdown('<div class="section-header">📖 What This Means</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="status-box">{result["plain_explanation"]}</div>',
                               unsafe_allow_html=True)

                    # Action checklist
                    st.markdown('<div class="section-header">✅ Action Checklist</div>', unsafe_allow_html=True)
                    for i, action in enumerate(result["action_checklist"], 1):
                        st.checkbox(f"{action}", key=f"action_{i}_{result['code']}")

                else:
                    st.error(result.get("error", "No match found."))
            elif search_btn:
                st.warning("Please enter a rejection code or description.")

        # --- Reference table ---
        st.markdown("---")
        st.markdown('<div class="section-header">📚 All Rejection Codes Reference</div>', unsafe_allow_html=True)

        ref_data = []
        for code in all_codes:
            ref_data.append({
                "Code": code["code"],
                "Message": code["raw_message"],
                "Explanation": code["plain_explanation"],
            })
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    # --- Tab 2: PA Letter Generator ---
    with tab2:
        st.markdown('<div class="section-header">📝 Auto-Generate PA Justification Letter</div>',
                   unsafe_allow_html=True)

        rules_df = load_insurance_rules()

        col_pa1, col_pa2 = st.columns(2)
        with col_pa1:
            pa_drug = st.selectbox("Medication", sorted(rules_df["drug_name"].unique()), key="pa_drug")
            pa_insurance = st.selectbox("Insurance", sorted(rules_df["insurance_name"].unique()), key="pa_ins")
            pa_age = st.number_input("Patient Age", 18, 90, 50, key="pa_age")

        with col_pa2:
            drug_diag = rules_df[rules_df["drug_name"] == pa_drug]["diagnosis_code"].iloc[0]
            pa_diag = st.text_input("Diagnosis Code", value=drug_diag, key="pa_diag")
            pa_codes = [c["code"] for c in all_codes]
            pa_rejection = st.selectbox("Rejection Code", pa_codes, key="pa_reject")
            pa_prior = st.checkbox("Prior drug was tried", key="pa_prior_tried")
            pa_lab = st.checkbox("Lab results available", key="pa_lab_avail")

        if st.button("📄 Generate Letter", type="primary", use_container_width=True):
            letter = generate_pa_letter(
                patient_age=pa_age,
                drug_name=pa_drug,
                insurance_name=pa_insurance,
                diagnosis_code=pa_diag,
                rejection_code=pa_rejection,
                prior_drug_tried=pa_prior,
                had_lab_results=pa_lab,
            )
            st.markdown(f'<div class="pa-letter">{letter}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Letter (.txt)",
                letter,
                file_name=f"PA_Letter_{pa_drug}_{pa_insurance}.txt",
                mime="text/plain",
            )


# ===========================================================================
# PAGE 3: ANALYTICS DASHBOARD
# ===========================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Insights and patterns from insurance claims data</p>
    </div>
    """, unsafe_allow_html=True)

    claims_df = load_claims()

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)

    total = len(claims_df)
    approved = claims_df["approved"].sum()
    rejected = total - approved
    approval_pct = approved / total

    # Most problematic insurer
    rejection_by_insurer = claims_df[claims_df["approved"] == 0].groupby("insurance_name").size()
    worst_insurer = rejection_by_insurer.idxmax() if len(rejection_by_insurer) > 0 else "N/A"

    # Most rejected drug
    rejection_by_drug = claims_df[claims_df["approved"] == 0].groupby("drug_name").size()
    worst_drug = rejection_by_drug.idxmax() if len(rejection_by_drug) > 0 else "N/A"

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">Total Claims</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #10b981;">{approval_pct:.1%}</div>
            <div class="kpi-label">Approval Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #ef4444;">{rejected}</div>
            <div class="kpi-label">Rejections</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #f59e0b; font-size: 1.3rem;">{worst_drug}</div>
            <div class="kpi-label">Most Rejected Drug</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts row 1 ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown('<div class="section-header">Approval Rate by Insurance Company</div>',
                   unsafe_allow_html=True)
        approval_by_insurer = claims_df.groupby("insurance_name")["approved"].mean().reset_index()
        approval_by_insurer.columns = ["Insurance", "Approval Rate"]
        approval_by_insurer = approval_by_insurer.sort_values("Approval Rate")

        fig1 = px.bar(
            approval_by_insurer,
            x="Approval Rate",
            y="Insurance",
            orientation="h",
            color="Approval Rate",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[0, 1],
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.markdown('<div class="section-header">Rejection Codes Distribution</div>',
                   unsafe_allow_html=True)
        rejected_claims = claims_df[claims_df["approved"] == 0]
        code_dist = rejected_claims["rejection_code"].value_counts().reset_index()
        code_dist.columns = ["Code", "Count"]

        fig2 = px.pie(
            code_dist,
            values="Count",
            names="Code",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4"],
            hole=0.4,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(size=11)),
        )
        fig2.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Charts row 2 ---
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown('<div class="section-header">Approval Rate by Medication</div>',
                   unsafe_allow_html=True)
        approval_by_drug = claims_df.groupby("drug_name").agg(
            total=("approved", "count"),
            approved=("approved", "sum"),
        ).reset_index()
        approval_by_drug["rate"] = approval_by_drug["approved"] / approval_by_drug["total"]
        approval_by_drug = approval_by_drug.sort_values("rate")

        fig3 = px.bar(
            approval_by_drug,
            x="drug_name",
            y="rate",
            color="rate",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[0, 1],
            labels={"drug_name": "Drug", "rate": "Approval Rate"},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-45),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_chart4:
        st.markdown('<div class="section-header">Patient Age vs. Approval</div>',
                   unsafe_allow_html=True)

        # Create age bins
        claims_df["age_group"] = pd.cut(
            claims_df["patient_age"],
            bins=[17, 30, 45, 60, 75, 90],
            labels=["18-30", "31-45", "46-60", "61-75", "76-90"],
        )
        age_approval = claims_df.groupby("age_group", observed=True)["approved"].mean().reset_index()
        age_approval.columns = ["Age Group", "Approval Rate"]

        fig4 = px.line(
            age_approval,
            x="Age Group",
            y="Approval Rate",
            markers=True,
            line_shape="spline",
        )
        fig4.update_traces(
            line=dict(color="#00D4AA", width=3),
            marker=dict(size=10, color="#00D4AA"),
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Claims Data Explorer ---
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Claims Data Explorer</div>', unsafe_allow_html=True)

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        filter_insurance = st.multiselect(
            "Filter by Insurance",
            claims_df["insurance_name"].unique(),
            key="filter_ins",
        )
    with col_filter2:
        filter_drug = st.multiselect(
            "Filter by Drug",
            claims_df["drug_name"].unique(),
            key="filter_drug",
        )
    with col_filter3:
        filter_status = st.selectbox(
            "Status",
            ["All", "Approved Only", "Rejected Only"],
            key="filter_status",
        )

    filtered = claims_df.copy()
    if filter_insurance:
        filtered = filtered[filtered["insurance_name"].isin(filter_insurance)]
    if filter_drug:
        filtered = filtered[filtered["drug_name"].isin(filter_drug)]
    if filter_status == "Approved Only":
        filtered = filtered[filtered["approved"] == 1]
    elif filter_status == "Rejected Only":
        filtered = filtered[filtered["approved"] == 0]

    st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
    st.caption(f"Showing {len(filtered):,} of {total:,} claims")


# ===========================================================================
# PAGE 4: MODEL PERFORMANCE (for diploma defense)
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
        st.error("⚠️ Model not trained yet. Run `python models/train_model.py` first.")
        st.stop()

    # --- Model Overview ---
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{metadata['accuracy']:.1%}</div>
            <div class="kpi-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{metadata['f1_score']:.1%}</div>
            <div class="kpi-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="font-size: 1.5rem;">{metadata['model_name']}</div>
            <div class="kpi-label">Best Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Confusion Matrix ---
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

        cm = np.array(metadata["confusion_matrix"])
        labels = ["Rejected", "Approved"]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Predicted<br>{l}" for l in labels],
            y=[f"Actual<br>{l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            colorscale=[[0, "#1a1f2e"], [1, "#00D4AA"]],
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_fi:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

        fi = metadata["feature_importance"]
        fi_df = pd.DataFrame(
            sorted(fi.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Importance"],
        )

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1a1f2e", "#00D4AA"],
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa"),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # --- Classification Report ---
    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report = metadata["classification_report"]

    report_data = []
    for label in ["Rejected (0)", "Approved (1)"]:
        if label in report:
            r = report[label]
            report_data.append({
                "Class": label,
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1-Score": f"{r['f1-score']:.4f}",
                "Support": int(r["support"]),
            })

    if "weighted avg" in report:
        r = report["weighted avg"]
        report_data.append({
            "Class": "Weighted Average",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1-score']:.4f}",
            "Support": int(r["support"]),
        })

    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    # --- Training Details ---
    st.markdown('<div class="section-header">Training Details</div>', unsafe_allow_html=True)
    col_td1, col_td2 = st.columns(2)
    with col_td1:
        st.markdown(f"""
        - **Model Type:** {metadata['model_name']}
        - **Training Samples:** {metadata['train_samples']:,}
        - **Test Samples:** {metadata['test_samples']:,}
        - **Test/Train Split:** 80% / 20% (stratified)
        """)
    with col_td2:
        st.markdown(f"""
        - **Accuracy:** {metadata['accuracy']:.4f}
        - **F1 Score (weighted):** {metadata['f1_score']:.4f}
        - **Class Imbalance Handling:** {'scale_pos_weight' if metadata['model_name'] == 'XGBoost' else 'class_weight=balanced'}
        - **Data:** Synthetic (proof-of-concept)
        """)
