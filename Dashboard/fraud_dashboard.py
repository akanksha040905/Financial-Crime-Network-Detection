"""
Automated Financial Crime and Anti-Money Laundering (AML) Detection Dashboard.

Enterprise Intelligence Platform featuring 10 Dedicated Sections:
  1. Dataset Upload & Column Mapping
  2. Data Cleaning & Preprocessing Report
  3. Machine Learning Results & Model Performance
  4. Transaction-Level Risk Scoring
  5. Account-Level Risk Register
  6. Transaction Network Graph (Interactive Plotly)
  7. Graph Analytics & AML Typology Detection
  8. Interactive Account Investigation Dashboard
  9. SHAP Explainability Studio
  10. Reports & Audit Downloads

Palette: Navy (#12305E), Teal (#1F9D8B), Alert Red (#D7263D), Amber (#E69F00), Ground (#F4F6F9)
"""

from __future__ import annotations

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "aml_mpl")

import base64
import datetime
import hashlib
import html
import io
import json
import warnings
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import explainability as exp
import fraud_core as fc
import graph_analytics as ga
import report_generator as rg
from sample_data_generator import generate_aml_demo_dataset

# -----------------------------------------------------------------------------
# App Configuration & Enterprise Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AML & Financial Crime Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY = "#0F172A"
TEAL = "#0D9488"
ALERT = "#E11D48"
AMBER = "#D97706"
GROUND = "#F8FAFC"
PANEL = "#FFFFFF"
LINE = "#E2E8F0"
HAIR = "#F1F5F9"
INK = "#0F172A"
MUTED = "#64748B"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --ground: #F8FAFC;
    --panel: #FFFFFF;
    --line: #E2E8F0;
    --hair: #F1F5F9;
    --ink: #0F172A;
    --muted: #64748B;
    --navy: #0F172A;
    --indigo: #4F46E5;
    --alert: #E11D48;
    --amber: #D97706;
    --ok: #0D9488;
}

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--ground);
    color: var(--ink);
}

.stMarkdown, .stButton button, [data-testid="stWidgetLabel"], .stSelectbox label, .stSlider label, .stCheckbox label,
button[data-baseweb="tab"] p, [data-baseweb="select"] div, [data-testid="stFileUploader"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

code, pre, .mono, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

header[data-testid="stHeader"] { background: transparent; z-index: 999 !important; }
#MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton { display: none; }
.block-container { padding-top: 0.8rem; padding-bottom: 3rem; max-width: 1440px; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--line) !important;
    box-shadow: 2px 0 12px rgba(15, 23, 42, 0.03);
    z-index: 100 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem !important;
}

/* Sidebar Toggle */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: linear-gradient(135deg, #0F172A, #1E293B) !important;
    color: #FFFFFF !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 8px 12px !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25) !important;
    top: 16px !important;
    left: 0 !important;
    z-index: 999999 !important;
    cursor: pointer !important;
    transition: all 0.2s ease;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #4F46E5 !important;
}
[data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Premium Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    border: 1px solid var(--line);
    background: #FFFFFF;
    color: var(--ink);
    padding: 6px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.stButton>button:hover {
    border-color: var(--indigo);
    color: var(--indigo);
    background: #F8FAFC;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(79, 70, 229, 0.1);
}
.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    border: 1px solid #0F172A;
    color: #FFFFFF;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.2);
}
.stButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #1E293B, #334155);
    border-color: #334155;
    color: #FFFFFF;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.3);
}

/* Enterprise Tabs */
div[data-baseweb="tab-list"] {
    gap: 4px;
    background: #F1F5F9;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--line);
    margin-bottom: 14px;
}
button[data-baseweb="tab"] {
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 8px 14px;
    border-radius: 6px;
    border: none !important;
    background: transparent;
    color: var(--muted);
    transition: all 0.2s ease;
}
button[data-baseweb="tab"]:hover {
    color: var(--ink);
    background: rgba(255, 255, 255, 0.6);
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0F172A !important;
    background: #FFFFFF !important;
    font-weight: 700;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.06);
}
div[data-baseweb="tab-highlight"] { display: none; }

/* Top Header Bar */
.brand-strip {
    background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #1E293B 100%);
    border-radius: 12px;
    padding: 16px 22px;
    margin-bottom: 12px;
    color: #FFFFFF;
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}
.brand-strip::after {
    content: '';
    position: absolute;
    top: 0; right: 0; width: 400px; height: 100%;
    background: radial-gradient(circle, rgba(79, 70, 229, 0.15) 0%, transparent 70%);
    pointer-events: none;
}
.brand-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.brand-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-sub {
    color: #94A3B8;
    font-size: 12.5px;
    margin-top: 4px;
    line-height: 1.5;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10B981;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
    animation: pulse 2s infinite ease-in-out;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.9); }
}

/* Quick Bar */
.quick-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* Elevated KPI Cards Grid */
.kpis-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.08);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: #CBD5E1;
}
.kpi-card.accent-indigo::before { background: linear-gradient(90deg, #4F46E5, #818CF8); }
.kpi-card.accent-rose::before { background: linear-gradient(90deg, #E11D48, #F43F5E); }
.kpi-card.accent-amber::before { background: linear-gradient(90deg, #D97706, #F59E0B); }
.kpi-card.accent-teal::before { background: linear-gradient(90deg, #0D9488, #14B8A6); }

.kpi-val {
    font-size: 22px;
    font-weight: 800;
    color: #0F172A;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    margin-top: 2px;
}
.kpi-val.alert { color: var(--alert); }
.kpi-val.amber { color: var(--amber); }
.kpi-val.ok { color: var(--ok); }
.kpi-lbl {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 6px;
}
.kpi-sub {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 2px;
}
@media (max-width: 1150px) { .kpis-grid { grid-template-columns: repeat(4, 1fr); } }
@media (max-width: 768px) { .kpis-grid { grid-template-columns: repeat(2, 1fr); } }

/* Panels */
.panel {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.ph {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--hair);
    padding-bottom: 10px;
}
.pt {
    font-weight: 700;
    font-size: 14px;
    letter-spacing: -0.01em;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 8px;
}
.pn {
    color: var(--muted);
    font-size: 11.5px;
    background: #F1F5F9;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-low { background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.badge-med { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
.badge-high { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.badge-crit { background: #450A0A; color: #FEF2F2; border: 1px solid #7F1D1D; }

/* Narrative box */
.narr-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #0F172A;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 8px;
    font-size: 12.5px;
    line-height: 1.5;
}
.narr-box.alert {
    border-left-color: var(--alert);
    background: #FFF1F2;
    border-color: #FECDD3;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# State Management & Pipeline Orchestration
# -----------------------------------------------------------------------------

def initialize_session_state():
    defaults = {
        "dataset_key": None,
        "last_processed_file": None,
        "raw_df": None,
        "dataset_name": None,
        "column_mapping": None,
        "cleaned_bundle": None,
        "supervised_res": None,
        "unsupervised_res": None,
        "df_tx_risk": None,
        "G": None,
        "graph_metrics": None,
        "pattern_findings": None,
        "graph_features": None,
        "account_risk_df": None,
        "selected_account": None,
        "pipeline_completed": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()


def execute_pipeline(df_in: pd.DataFrame, file_name: str, mapping_override: Optional[Dict[str, str]] = None, fraud_val_override=None):
    """
    Executes the end-to-end automated pipeline:
      1. Ingestion & Column Mapping
      2. Automated Cleaning & Advanced AML Feature Engineering
      3. Random Forest ML (or Isolation Forest Fallback)
      4. Transaction-Level Risk Scoring (0 - 100)
      5. Graph Construction (NetworkX directed weighted graph)
      6. Graph Analytics & AML Pattern Typology Detection
      7. Account-Level Risk Synthesis (0 - 100)
    """
    import time
    t_start = time.time()
    with st.status(f"⚡ Processing '{file_name}' ({len(df_in):,} records)...", expanded=True) as status:
        # Step 1: Mapping
        t0 = time.time()
        st.write("Step 1/6: Validating transaction schema and column mappings...")
        if mapping_override:
            col_map = mapping_override
        else:
            col_map = fc.detect_column_mapping(df_in)
        
        valid, errs = fc.validate_column_mapping(df_in, col_map)
        if not valid:
            status.update(label="Pipeline validation error", state="error")
            for e in errs:
                st.error(e)
            return False
        st.write(f"✓ Step 1/6: Schema validated ({len(df_in):,} records) in {time.time()-t0:.2f}s")

        # Step 2: Cleaning & Engineering
        t0 = time.time()
        st.write("Step 2/6: Performing data cleaning and engineering behavioral / velocity features...")
        cleaned = fc.clean_and_engineer_features(df_in, col_map, fraud_values=fraud_val_override)
        st.write(f"✓ Step 2/6: Engineered {len(cleaned.X.columns)} AML features in {time.time()-t0:.2f}s")

        # Step 3: Machine Learning
        t0 = time.time()
        st.write("Step 3/6: Training model (Random Forest / Anomaly Fallback)...")
        is_supervised = cleaned.y is not None and int(cleaned.y.sum()) >= 5
        sup_model = None
        unsup_model = None
        scores_arr = None

        if is_supervised:
            sup_model = fc.train_supervised_random_forest(cleaned.X, cleaned.y, n_trees=50, n_splits=3)
            scores_arr = sup_model.proba
            model_info = f"Random Forest ({sup_model.n_trees} trees, AUC: {sup_model.roc_auc:.3f})"
        else:
            unsup_model = fc.train_unsupervised_isolation_forest(cleaned.X, n_trees=80)
            scores_arr = unsup_model.scores
            model_info = f"Isolation Forest Anomaly Detector ({unsup_model.n_trees} trees)"
        st.write(f"✓ Step 3/6: Trained {model_info} in {time.time()-t0:.2f}s")

        # Step 4: Transaction Risk Scoring
        t0 = time.time()
        st.write("Step 4/6: Computing transaction-level calibrated risk scores (0-100)...")
        df_tx = fc.compute_transaction_risk_scores(cleaned.df_raw_cleaned, scores_arr, is_supervised, col_map)
        st.write(f"✓ Step 4/6: Scored {len(df_tx):,} transactions in {time.time()-t0:.2f}s")

        # Step 5: Graph Construction & Topology Analytics
        t0 = time.time()
        st.write("Step 5/6: Building transaction network graph and detecting AML typologies...")
        snd_c = col_map.get("sender_account", "sender_account")
        rcv_c = col_map.get("receiver_account", "receiver_account")
        amt_c = col_map.get("amount", "amount")
        time_c = col_map.get("timestamp", "timestamp")
        tx_id_c = col_map.get("transaction_id", "transaction_id")

        G = ga.build_transaction_network(
            df_tx,
            sender_col="sender_account",
            receiver_col="receiver_account",
            amount_col="amount",
            timestamp_col="timestamp",
            tx_id_col="transaction_id",
            risk_score_col="risk_score"
        )
        metrics = ga.compute_network_metrics(G)
        patterns = ga.detect_all_patterns(G, df_tx, col_map)
        g_feats = ga.extract_account_graph_features(G, metrics, patterns)
        st.write(f"✓ Step 5/6: Graph built ({G.number_of_nodes()} accounts) & {len(patterns)} typologies detected in {time.time()-t0:.2f}s")

        # Step 6: Account-Level Risk Aggregation
        t0 = time.time()
        st.write("Step 6/6: Synthesizing account risk register integrating graph analytics...")
        acc_risk = fc.compute_account_risk_scores(df_tx, g_feats)
        st.write(f"✓ Step 6/6: Account risk register synthesized for {len(acc_risk):,} accounts in {time.time()-t0:.2f}s")

        # Compute unique dataset signature incorporating feature schema
        col_sig = "_".join(list(cleaned.X.columns))
        ds_sig = hashlib.md5(f"{file_name}_{len(df_in)}_{col_sig}".encode()).hexdigest()[:12]

        # Update Session State
        st.session_state.update({
            "dataset_key": ds_sig,
            "raw_df": df_in,
            "dataset_name": file_name,
            "column_mapping": col_map,
            "cleaned_bundle": cleaned,
            "supervised_res": sup_model,
            "unsupervised_res": unsup_model,
            "df_tx_risk": df_tx,
            "G": G,
            "graph_metrics": metrics,
            "pattern_findings": patterns,
            "graph_features": g_feats,
            "account_risk_df": acc_risk,
            "pipeline_completed": True
        })

        if not st.session_state.get("selected_account") and not acc_risk.empty:
            st.session_state["selected_account"] = acc_risk.index[0]

        total_elapsed = time.time() - t_start
        status.update(label=f"✓ Pipeline execution completed successfully in {total_elapsed:.1f}s!", state="complete", expanded=False)
        return True


# -----------------------------------------------------------------------------
# Auto-Initialize Benchmark Demo Dataset on Startup
# -----------------------------------------------------------------------------
# Use a fast benchmark demo (400 normal records) for the first-load only
if not st.session_state.get("pipeline_completed"):
    init_demo_df = generate_aml_demo_dataset(n_normal=400, seed=42, include_labels=True)
    st.session_state["last_processed_file"] = "demo_benchmark_init"
    execute_pipeline(init_demo_df, "Synthetic AML Benchmark (Laundering Rings Active)")


# -----------------------------------------------------------------------------
# Top Banner & Global Overview
# -----------------------------------------------------------------------------

is_sup_mode = st.session_state.get("supervised_res") is not None
mode_tag = "SUPERVISED // RANDOM FOREST" if is_sup_mode else "UNSUPERVISED // ISOLATION FOREST"
mode_col = "#10B981" if is_sup_mode else "#F59E0B"

st.markdown(f"""
<div class="brand-strip">
    <div class="brand-header">
        <div>
            <div class="brand-title">
                <span style="font-size:24px;">🛡️</span>
                <span>FINANCIAL CRIME & AML NETWORK INTELLIGENCE PLATFORM</span>
            </div>
            <div class="brand-sub">
                Enterprise Autonomous Threat Detection &bull; Multi-Layer Directed Network Mining &bull; Explainable AI Risk Scoring
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div class="live-pill"><span class="pulse-dot"></span> LIVE ENGINE</div>
            <div style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); padding:4px 10px; border-radius:20px; font-size:11px; font-weight:700; color:{mode_col}; font-family:'JetBrains Mono', monospace;">
                {mode_tag}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-expand sidebar if the user's browser had it collapsed from a previous session
components.html(
    """
    <script>
    function ensureSidebarExpanded() {
        try {
            const doc = window.parent.document;
            const collapsedBtn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button');
            if (collapsedBtn) {
                collapsedBtn.click();
            }
        } catch(e) {}
    }
    setTimeout(ensureSidebarExpanded, 250);
    setTimeout(ensureSidebarExpanded, 800);
    </script>
    """,
    height=0,
    width=0
)

# Top Quick-Action Control Bar (Works whether sidebar is open or closed)
active_ds_name = st.session_state.get("dataset_name", "Synthetic AML Benchmark (Laundering Rings Active)")
top_col1, top_col2, top_col3, top_col4 = st.columns([2.2, 1, 1, 2])

top_col1.markdown(f"""
<div style="padding: 6px 0; font-size: 13px; color: #0F172A; display:flex; align-items:center; gap:6px;">
    <span style="font-size:14px;">📁</span> <b>Source:</b> <span style="color:#64748B; font-weight:500;">{active_ds_name}</span>
</div>
""", unsafe_allow_html=True)

if top_col2.button("⚡ Labeled Demo", key="top_btn_demo_l", use_container_width=True, help="Load Supervised Demo with Known Laundering Rings"):
    st.session_state["last_processed_file"] = "demo_labeled"
    d_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=True)
    execute_pipeline(d_df, "Synthetic AML Benchmark (Supervised)")
    st.rerun()

if top_col3.button("🔮 Unlabeled Demo", key="top_btn_demo_u", use_container_width=True, help="Load Unlabeled Demo for Isolation Forest Anomaly Detection"):
    st.session_state["last_processed_file"] = "demo_unlabeled"
    d_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=False)
    execute_pipeline(d_df, "Synthetic AML Benchmark (Unsupervised)")
    st.rerun()

# Display active investigation target in top bar (No competing selectbox)
acc_df_top = st.session_state.get("account_risk_df")
if acc_df_top is not None and not acc_df_top.empty:
    cur_sel_top = st.session_state.get("selected_account", acc_df_top.index[0])
    if cur_sel_top not in acc_df_top.index:
        cur_sel_top = acc_df_top.index[0]
        st.session_state["selected_account"] = cur_sel_top
    r_top = acc_df_top.loc[cur_sel_top]
    if isinstance(r_top, pd.DataFrame):
        r_top = r_top.iloc[0]
    top_score = float(r_top.get("risk_score", 0.0))
    top_cat = str(r_top.get("risk_category", "Low"))
    badge_style = f"badge-{top_cat.lower()[:4]}"
    top_col4.markdown(f"""
    <div style="padding: 6px 0; font-size: 13px; text-align:right;">
        <span style="color:#64748B;">Target:</span> <code style="font-size:12.5px; font-weight:700; color:#0F172A;">{cur_sel_top}</code> 
        <span class="badge {badge_style}">{top_score:.0f} &bull; {top_cat}</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Executive KPI Metrics
# -----------------------------------------------------------------------------

df_tx = st.session_state["df_tx_risk"]
acc_risk = st.session_state["account_risk_df"]
patterns = st.session_state["pattern_findings"]
G = st.session_state["G"]

tot_acc = len(acc_risk) if acc_risk is not None else 0
tot_tx = len(df_tx) if df_tx is not None else 0
crit_acc = int((acc_risk["risk_category"] == "Critical").sum()) if acc_risk is not None else 0
high_acc = int((acc_risk["risk_category"] == "High").sum()) if acc_risk is not None else 0
susp_tx = int((df_tx["risk_score"] >= 60.0).sum()) if df_tx is not None else 0
susp_val = float(df_tx[df_tx["risk_score"] >= 60.0]["amount"].sum()) if df_tx is not None else 0.0
tot_patterns = len(patterns) if patterns is not None else 0

st.markdown(f"""
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-bottom:14px;">
    <div class="kpi-card accent-indigo">
        <div class="kpi-val">{tot_acc:,}</div>
        <div class="kpi-lbl">Entities</div>
        <div class="kpi-sub">Active Accounts</div>
    </div>
    <div class="kpi-card accent-indigo">
        <div class="kpi-val">{tot_tx:,}</div>
        <div class="kpi-lbl">Transactions</div>
        <div class="kpi-sub">Audited Ledger</div>
    </div>
    <div class="kpi-card accent-rose">
        <div class="kpi-val alert">{crit_acc:,}</div>
        <div class="kpi-lbl">Critical Risk</div>
        <div class="kpi-sub">Score 81–100</div>
    </div>
    <div class="kpi-card accent-rose">
        <div class="kpi-val alert">{high_acc:,}</div>
        <div class="kpi-lbl">High Risk</div>
        <div class="kpi-sub">Score 61–80</div>
    </div>
</div>
<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-bottom:14px;">
    <div class="kpi-card accent-amber">
        <div class="kpi-val amber">{susp_tx:,}</div>
        <div class="kpi-lbl">Suspicious TXs</div>
        <div class="kpi-sub">{susp_tx / max(tot_tx, 1):.1%} Volume</div>
    </div>
    <div class="kpi-card accent-amber">
        <div class="kpi-val amber">${susp_val:,.0f}</div>
        <div class="kpi-lbl">Flagged Value</div>
        <div class="kpi-sub">At-Risk Exposure</div>
    </div>
    <div class="kpi-card accent-teal">
        <div class="kpi-val ok">{tot_patterns:,}</div>
        <div class="kpi-lbl">AML Alerts</div>
        <div class="kpi-sub">Topology Matches</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: File Uploader, Demo Loaders & Quick Navigator
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<h3 style='color:#12305E; margin-bottom:6px;'>Data Ingestion</h3>", unsafe_allow_html=True)
    sidebar_upload = st.file_uploader(
        "Upload Transaction Dataset",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "parquet"],
        key="sb_file_uploader",
        help="Upload CSV, TSV, Excel, or Parquet transaction records."
    )

    sb_col1, sb_col2 = st.columns(2)
    sb_load_demo_l = sb_col1.button("Demo (Labeled)", key="sb_demo_l", use_container_width=True)
    sb_load_demo_u = sb_col2.button("Demo (Unlabeled)", key="sb_demo_u", use_container_width=True)

    if sidebar_upload is not None:
        file_key = f"file_{sidebar_upload.name}_{sidebar_upload.size}"
        if st.session_state.get("last_processed_file") != file_key:
            try:
                raw_bytes = sidebar_upload.getvalue()
                df_loaded = fc.read_any(sidebar_upload.name, raw_bytes)
                st.session_state["last_processed_file"] = file_key
                execute_pipeline(df_loaded, sidebar_upload.name)
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif sb_load_demo_l:
        st.session_state["last_processed_file"] = "demo_labeled"
        demo_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=True)
        execute_pipeline(demo_df, "Synthetic AML Benchmark (Supervised)")
        st.rerun()

    elif sb_load_demo_u:
        st.session_state["last_processed_file"] = "demo_unlabeled"
        demo_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=False)
        execute_pipeline(demo_df, "Synthetic AML Benchmark (Unsupervised)")
        st.rerun()

    st.markdown("---")
    st.markdown("<h4 style='color:#12305E; margin-bottom:4px;'>Quick Account Navigator</h4>", unsafe_allow_html=True)
    
    if acc_risk is not None and not acc_risk.empty:
        acc_list = list(acc_risk.index)
        current_sel = st.session_state.get("selected_account", acc_list[0])
        if current_sel not in acc_list:
            current_sel = acc_list[0]
            st.session_state["selected_account"] = current_sel
            
        cur_row = acc_risk.loc[current_sel]
        if isinstance(cur_row, pd.DataFrame):
            cur_row = cur_row.iloc[0]
        cur_score = float(cur_row.get("risk_score", 0.0))
        cur_cat = str(cur_row.get("risk_category", "Low"))

        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #D5DBE5; border-radius:5px; padding:8px 10px; margin-bottom:8px;">
            <div style="font-size:11px; color:#5B6779; font-weight:600; text-transform:uppercase;">Investigating Target</div>
            <div style="font-size:14px; font-weight:700; color:#12305E;">{current_sel}</div>
            <div style="font-size:11.5px; margin-top:2px;">Risk: <b>{cur_score:.0f} / 100</b> ({cur_cat})</div>
        </div>
        """, unsafe_allow_html=True)

        def _on_sb_nav_change():
            st.session_state["selected_account"] = st.session_state["sb_target_selector"]

        if current_sel in acc_list:
            st.session_state["sb_target_selector"] = current_sel

        st.selectbox(
            "Change Target Account",
            acc_list,
            format_func=lambda a: f"{a}  [{float(acc_risk.loc[a, 'risk_score'].iloc[0] if isinstance(acc_risk.loc[a, 'risk_score'], pd.Series) else acc_risk.loc[a, 'risk_score']):.0f} - {str(acc_risk.loc[a, 'risk_category'].iloc[0] if isinstance(acc_risk.loc[a, 'risk_category'], pd.Series) else acc_risk.loc[a, 'risk_category'])}]",
            key="sb_target_selector",
            on_change=_on_sb_nav_change
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 11.5px; color: #5B6779; line-height: 1.4;'>
        <b>Audit & Compliance Disclaimers:</b><br>
        Risk scores indicate statistical and topological anomaly likelihood; they do not constitute definitive legal proof of illicit activity.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Navigation Tabs: All 10 Dedicated Sections
# -----------------------------------------------------------------------------

tabs = st.tabs([
    "📥 Ingestion",
    "🧹 Sanitization",
    "🤖 ML Models",
    "⚡ TX Risk",
    "👥 Accounts",
    "🕸️ Network Graph",
    "🔍 Typologies",
    "🎯 Investigation",
    "🔬 Explainability",
    "📑 Audit Dossier"
])


# =============================================================================
# TAB 1: Dataset Upload
# =============================================================================
with tabs[0]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">1. Transaction Dataset Ingestion & Column Mapping</div>', unsafe_allow_html=True)

    # Main page upload area
    up_left, up_right = st.columns([1, 1])
    
    with up_left:
        st.markdown('<div class="panel"><div class="ph"><span class="pt">Upload New Transaction Dataset</span></div>', unsafe_allow_html=True)
        tab_uploader = st.file_uploader(
            "Drop your transaction file here (CSV, TSV, TXT, Excel, Parquet)",
            type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "parquet"],
            key="main_tab_uploader"
        )
        
        st.markdown("<div style='font-size:12px; color:#64748B; margin: 10px 0 6px; font-weight:700;'>⚡ Built-in Demo Datasets (1-Click):</div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button("📂 Load Demo (Labeled Rings)", key="main_demo_l", use_container_width=True):
            st.session_state["last_processed_file"] = "demo_labeled"
            demo_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=True)
            execute_pipeline(demo_df, "Synthetic AML Benchmark (Supervised)")
            st.rerun()
            
        if btn_c2.button("📂 Load Demo (Unlabeled)", key="main_demo_u", use_container_width=True):
            st.session_state["last_processed_file"] = "demo_unlabeled"
            demo_df = generate_aml_demo_dataset(n_normal=800, seed=42, include_labels=False)
            execute_pipeline(demo_df, "Synthetic AML Benchmark (Unsupervised)")
            st.rerun()

        if tab_uploader is not None:
            file_key = f"file_{tab_uploader.name}_{tab_uploader.size}"
            if st.session_state.get("last_processed_file") != file_key:
                try:
                    raw_bytes = tab_uploader.getvalue()
                    df_loaded = fc.read_any(tab_uploader.name, raw_bytes)
                    st.session_state["last_processed_file"] = file_key
                    execute_pipeline(df_loaded, tab_uploader.name)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with up_right:
        st.markdown('<div class="panel"><div class="ph"><span class="pt">Active Dataset Profile & Validation</span></div>', unsafe_allow_html=True)
        cur_map = st.session_state.get("column_mapping", {})
        raw_df = st.session_state.get("raw_df", pd.DataFrame())
        
        st.markdown(f"""
        <div style="font-size:13px; line-height:1.7;">
            • <b>Active Source:</b> {st.session_state.get('dataset_name', 'N/A')}<br>
            • <b>Total Transaction Records:</b> {len(raw_df):,}<br>
            • <b>Total Schema Columns:</b> {raw_df.shape[1]}<br>
            • <b>Identified Unique Accounts:</b> {len(acc_risk):,}<br>
            • <b>Sender Field:</b> <code>{cur_map.get('sender_account')}</code><br>
            • <b>Receiver Field:</b> <code>{cur_map.get('receiver_account')}</code><br>
            • <b>Amount Field:</b> <code>{cur_map.get('amount')}</code><br>
            • <b>Timestamp Field:</b> <code>{cur_map.get('timestamp') or 'None (Synthesized)'}</code><br>
            • <b>Supervised Label:</b> <code>{cur_map.get('is_fraud') or 'None (Isolation Forest Engaged)'}</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<b>Raw Transaction Preview (First 8 Rows)</b>", unsafe_allow_html=True)
    if not raw_df.empty:
        st.dataframe(raw_df.head(8), use_container_width=True)

    with st.expander("Correct / Adjust Column Mappings Manually"):
        st.write("If automatic detection mapped a column incorrectly, choose the correct columns below and click **Apply & Re-run**:")
        all_cols = ["None"] + list(raw_df.columns)
        
        m_c1, m_c2, m_c3 = st.columns(3)
        new_snd = m_c1.selectbox("Sender Account", all_cols, index=all_cols.index(cur_map.get("sender_account", "None")) if cur_map.get("sender_account") in all_cols else 0)
        new_rcv = m_c2.selectbox("Receiver Account", all_cols, index=all_cols.index(cur_map.get("receiver_account", "None")) if cur_map.get("receiver_account") in all_cols else 0)
        new_amt = m_c3.selectbox("Amount Column", all_cols, index=all_cols.index(cur_map.get("amount", "None")) if cur_map.get("amount") in all_cols else 0)
        
        m_c4, m_c5, m_c6 = st.columns(3)
        new_time = m_c4.selectbox("Timestamp Column", all_cols, index=all_cols.index(cur_map.get("timestamp", "None")) if cur_map.get("timestamp") in all_cols else 0)
        new_txid = m_c5.selectbox("Transaction ID", all_cols, index=all_cols.index(cur_map.get("transaction_id", "None")) if cur_map.get("transaction_id") in all_cols else 0)
        new_lbl = m_c6.selectbox("Fraud / Laundering Label", all_cols, index=all_cols.index(cur_map.get("is_fraud", "None")) if cur_map.get("is_fraud") in all_cols else 0)

        if st.button("Apply Mappings & Re-run Pipeline", type="primary"):
            upd_map = {
                "sender_account": None if new_snd == "None" else new_snd,
                "receiver_account": None if new_rcv == "None" else new_rcv,
                "amount": None if new_amt == "None" else new_amt,
                "timestamp": None if new_time == "None" else new_time,
                "transaction_id": None if new_txid == "None" else new_txid,
                "is_fraud": None if new_lbl == "None" else new_lbl
            }
            execute_pipeline(raw_df, st.session_state.get("dataset_name", "Dataset"), mapping_override=upd_map)
            st.rerun()


# =============================================================================
# TAB 2: Data Cleaning and Preprocessing
# =============================================================================
with tabs[1]:
    bundle: fc.CleanedDataBundle = st.session_state["cleaned_bundle"]
    rep = bundle.report
    
    st.markdown('<div class="pt" style="margin-bottom:8px;">2. Automated Cleaning, Sanitization & Feature Engineering Audit</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicates Handled", f"{rep.duplicates_removed:,}")
    c2.metric("Invalid Amounts Purged", f"{rep.invalid_amounts_removed:,}")
    c3.metric("Imputed Null Values", f"{rep.missing_cells_imputed:,}")
    c4.metric("Engineered Features", f"{len(rep.engineered_features):,}")

    st.markdown('<div class="panel" style="margin-top:14px;"><div class="ph"><span class="pt">Data Sanitization & Preprocessing Log</span></div>', unsafe_allow_html=True)
    log_rows = "".join(f"<tr><td style='font-weight:600; width:240px; padding:6px 8px; border-bottom:1px solid #E4E9F0;'>{html.escape(s)}</td><td style='padding:6px 8px; border-bottom:1px solid #E4E9F0;'>{html.escape(d)}</td></tr>" for s, d in rep.log_messages)
    st.markdown(f"<table style='width:100%; border-collapse:collapse; font-size:13px;'>{log_rows}</table></div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="ph"><span class="pt">Engineered Numeric Feature Matrix (First 8 Records)</span></div>', unsafe_allow_html=True)
    st.dataframe(bundle.X.head(8), use_container_width=True)
    st.markdown(f"<div class='pn'>Total feature space: {bundle.X.shape[0]:,} rows × {bundle.X.shape[1]} numeric attributes.</div></div>", unsafe_allow_html=True)


# =============================================================================
# TAB 3: Machine Learning Results
# =============================================================================
with tabs[2]:
    sup = st.session_state["supervised_res"]
    unsup = st.session_state["unsupervised_res"]

    if sup is not None:
        st.markdown('<div class="pt" style="margin-bottom:8px;">3. Supervised Random Forest Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:13px; color:#5B6779; margin-bottom:12px;">
            The model was trained and validated using <b>Stratified 5-Fold Cross-Validation</b>. Every training transaction received an honest out-of-fold prediction score generated exclusively by decision trees that never saw that record during training.
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("ROC-AUC", f"{sup.roc_auc:.4f}")
        k2.metric("PR-AUC (Avg Prec)", f"{sup.avg_precision:.4f}")
        k3.metric("Precision", f"{sup.precision:.1%}")
        k4.metric("Recall", f"{sup.recall:.1%}")
        k5.metric("F1-Score", f"{sup.f1:.4f}")

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            cm = sup.confusion
            st.markdown(f"""
            <div class="panel">
                <div class="ph"><span class="pt">Out-of-Fold Confusion Matrix</span></div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:8px;">
                    <div style="background:#EAF6F4; border:1px solid #1F9D8B; border-radius:4px; padding:12px; text-align:center;">
                        <span style="font-size:22px; font-weight:700; color:#1F9D8B;">{cm['tn']:,}</span><br>
                        <span style="font-size:12px; color:#5B6779;">True Negatives (Legitimate)</span>
                    </div>
                    <div style="background:#FFF5F5; border:1px solid #D7263D; border-radius:4px; padding:12px; text-align:center;">
                        <span style="font-size:22px; font-weight:700; color:#D7263D;">{cm['fp']:,}</span><br>
                        <span style="font-size:12px; color:#5B6779;">False Positives (False Alarms)</span>
                    </div>
                    <div style="background:#FFF5F5; border:1px solid #D7263D; border-radius:4px; padding:12px; text-align:center;">
                        <span style="font-size:22px; font-weight:700; color:#D7263D;">{cm['fn']:,}</span><br>
                        <span style="font-size:12px; color:#5B6779;">False Negatives (Missed)</span>
                    </div>
                    <div style="background:#EAF6F4; border:1px solid #1F9D8B; border-radius:4px; padding:12px; text-align:center;">
                        <span style="font-size:22px; font-weight:700; color:#1F9D8B;">{cm['tp']:,}</span><br>
                        <span style="font-size:12px; color:#5B6779;">True Positives (Caught)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Top Model Feature Importances (Gini Impurity)</span></div>', unsafe_allow_html=True)
            imp_df = sup.importances.head(10).reset_index()
            imp_df.columns = ["Feature", "Importance"]
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker=dict(color="#12305E")
            ))
            fig_imp.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                xaxis=dict(showgrid=True, gridcolor="#E4E9F0"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="pt" style="margin-bottom:8px;">3. Unsupervised Isolation Forest Model Results</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:13px; color:#5B6779; margin-bottom:12px;">
            Because this dataset contains no supervised fraud labels, the platform trained an <b>Isolation Forest ensemble (150 trees)</b>. 
            Anomalous transactions are isolated closer to the root of randomized decision trees; shorter path lengths signify multidimensional rarity. 
            Raw isolation scores are calibrated into percentile risk scores from 0.0 (baseline normal) to 100.0 (extreme outlier).
        </div>
        """, unsafe_allow_html=True)

        bundle_X = st.session_state["cleaned_bundle"].X
        u_tot = len(df_tx)
        u_crit = int((df_tx["risk_category"] == "Critical").sum())
        u_high = int((df_tx["risk_category"] == "High").sum())
        u_med = int((df_tx["risk_category"] == "Medium").sum())
        u_low = int((df_tx["risk_category"] == "Low").sum())
        u_mean_score = float(df_tx["risk_score"].mean())

        # 5 KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Isolation Trees", f"{unsup.n_trees}")
        k2.metric("Total Scored", f"{u_tot:,}")
        k3.metric("Critical Outliers", f"{u_crit:,}", f"{u_crit/max(u_tot,1):.1%}")
        k4.metric("High Outliers", f"{u_high:,}", f"{u_high/max(u_tot,1):.1%}")
        k5.metric("Avg Anomaly Score", f"{u_mean_score:.1f} / 100")

        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Anomaly Score Distribution & Severity Tiers</span></div>', unsafe_allow_html=True)
            fig_dist = exp.plot_anomaly_score_distribution(df_tx["risk_score"].values)
            fig_dist.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=30))
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-around; font-size:12px; margin-top:4px;">
                <span><b style="color:#1F9D8B;">● Low (0–30):</b> {u_low:,} ({u_low/max(u_tot,1):.1%})</span>
                <span><b style="color:#E69F00;">● Medium (31–60):</b> {u_med:,} ({u_med/max(u_tot,1):.1%})</span>
                <span><b style="color:#DC2626;">● High (61–80):</b> {u_high:,} ({u_high/max(u_tot,1):.1%})</span>
                <span><b style="color:#7F1D1D;">● Critical (81–100):</b> {u_crit:,} ({u_crit/max(u_tot,1):.1%})</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col_u2:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Feature Sensitivity (Correlation with Anomaly Score)</span></div>', unsafe_allow_html=True)
            # Calculate correlation between numeric features and the resulting anomaly score
            feat_corrs = bundle_X.apply(lambda col: abs(float(col.corr(df_tx["risk_score"])))).fillna(0.0).sort_values(ascending=False).head(10).reset_index()
            feat_corrs.columns = ["Feature", "Correlation"]
            fig_corr = go.Figure(go.Bar(
                x=feat_corrs["Correlation"],
                y=feat_corrs["Feature"],
                orientation="h",
                marker=dict(color="#1F9D8B")
            ))
            fig_corr.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                xaxis=dict(showgrid=True, gridcolor="#E4E9F0", title="Correlation with Anomaly Magnitude"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="ph"><span class="pt">Top 10 Most Anomalous Outlier Transactions</span><span class="pn">Requires immediate AML compliance review</span></div>', unsafe_allow_html=True)
        top_anomalies = df_tx.sort_values("risk_score", ascending=False).head(10)[["transaction_id", "timestamp", "sender_account", "receiver_account", "amount", "risk_score", "risk_category", "predicted_status"]]
        try:
            cfg_top = {
                "risk_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0.0, max_value=100.0, format="%.1f"),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f")
            }
            st.dataframe(top_anomalies, column_config=cfg_top, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(top_anomalies, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 4: Transaction-Level Risk Scoring
# =============================================================================
with tabs[3]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">4. Transaction-Level Risk Scoring (0 – 100)</div>', unsafe_allow_html=True)
    
    # Filter controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    tier_choice = f_col1.multiselect(
        "Filter by Risk Tier",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"],
        key="tx_tier_filt"
    )
    min_amount = f_col2.number_input("Minimum Transaction Amount ($)", min_value=0.0, value=0.0, step=500.0, key="tx_min_amt")
    acc_filter = f_col3.text_input("Search Account ID or TX ID", placeholder="e.g. ACC_CYCLE_A or TX_100020", key="tx_search_q")

    filtered_tx = df_tx.copy()
    if tier_choice:
        filtered_tx = filtered_tx[filtered_tx["risk_category"].isin(tier_choice)]
    if min_amount > 0:
        filtered_tx = filtered_tx[filtered_tx["amount"] >= min_amount]
    if acc_filter:
        q = acc_filter.strip().lower()
        filtered_tx = filtered_tx[
            filtered_tx["sender_account"].str.lower().str.contains(q) |
            filtered_tx["receiver_account"].str.lower().str.contains(q) |
            filtered_tx["transaction_id"].str.lower().str.contains(q)
        ]

    st.markdown(f"<div class='pn' style='margin-bottom:8px;'>Displaying <b>{len(filtered_tx):,}</b> matching transactions (sorted by Risk Score):</div>", unsafe_allow_html=True)
    
    try:
        cfg = {
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=100.0, format="%.1f"),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f")
        }
        st.dataframe(filtered_tx.sort_values("risk_score", ascending=False).head(1000), column_config=cfg, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(filtered_tx.sort_values("risk_score", ascending=False).head(1000), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5: Account-Level Risk Register
# =============================================================================
with tabs[4]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">5. Account Risk Register & Entity Ranking</div>', unsafe_allow_html=True)
    
    ar_col1, ar_col2, ar_col3 = st.columns([1, 1, 2])
    cat_filt = ar_col1.multiselect("Risk Category", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium"], key="reg_cat_filt")
    min_tx_cnt = ar_col2.number_input("Min Transactions", min_value=1, value=1, step=1, key="reg_min_tx")
    acc_search = ar_col3.text_input("Search Account Register", placeholder="e.g. ACC_HUB_COLLECTOR", key="reg_acc_search")

    filtered_acc = acc_risk.copy()
    if cat_filt:
        filtered_acc = filtered_acc[filtered_acc["risk_category"].isin(cat_filt)]
    if min_tx_cnt > 1:
        filtered_acc = filtered_acc[filtered_acc["total_transactions"] >= min_tx_cnt]
    if acc_search:
        filtered_acc = filtered_acc[filtered_acc.index.str.lower().str.contains(acc_search.strip().lower())]

    disp_reg = filtered_acc.reset_index().rename(columns={"index": "account_id"})
    cols_to_drop = [c for c in ["total_tx_count"] if c in disp_reg.columns]
    disp_reg = disp_reg.drop(columns=cols_to_drop)

    st.markdown(f"<div class='pn' style='margin-bottom:8px;'>Showing <b>{len(disp_reg):,}</b> ranked accounts. <i>Click any row to load into Investigation Workbench:</i></div>", unsafe_allow_html=True)
    
    try:
        cfg_acc = {
            "account_id": st.column_config.TextColumn("Account ID", width="medium"),
            "overall_risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=100.0, format="%.1f"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=100.0, format="%.1f"),
            "risk_category": st.column_config.TextColumn("Risk Tier"),
            "total_received": st.column_config.NumberColumn("Total Inflow ($)", format="$%.2f"),
            "total_sent": st.column_config.NumberColumn("Total Outflow ($)", format="$%.2f"),
            "net_flow": st.column_config.NumberColumn("Net Flow ($)", format="$%.2f"),
            "total_transactions": st.column_config.NumberColumn("Transactions"),
            "suspicious_tx_count": st.column_config.NumberColumn("Suspicious TXs"),
            "pattern_count": st.column_config.NumberColumn("Patterns Flagged")
        }
        ev = st.dataframe(
            disp_reg,
            column_config=cfg_acc,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="acc_reg_interactive_table"
        )
        if ev and hasattr(ev, "selection") and ev.selection and ev.selection.rows:
            sel_idx = ev.selection.rows[0]
            if sel_idx < len(disp_reg):
                clicked_acc = str(disp_reg.iloc[sel_idx]["account_id"])
                if st.session_state.get("selected_account") != clicked_acc:
                    st.session_state["selected_account"] = clicked_acc
                    st.rerun()
    except Exception:
        st.dataframe(filtered_acc, use_container_width=True)

    if not filtered_acc.empty:
        reg_c1, reg_c2 = st.columns([3, 1])

        def _fmt_acc_reg(a):
            try:
                row = filtered_acc.loc[a]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return f"{a}  — Risk {float(row['risk_score']):.0f} ({row['risk_category']})"
            except Exception:
                return str(a)

        chosen_reg_acc = reg_c1.selectbox(
            "Quick Select Account to Investigate",
            list(filtered_acc.index),
            index=list(filtered_acc.index).index(st.session_state.get("selected_account", filtered_acc.index[0])) if st.session_state.get("selected_account") in filtered_acc.index else 0,
            format_func=_fmt_acc_reg,
            key="reg_investigate_selector",
            label_visibility="collapsed"
        )
        if reg_c2.button("🔍 Investigate Account in Tab 8", key="btn_open_tab8_from_reg", use_container_width=True):
            st.session_state["selected_account"] = chosen_reg_acc
            st.rerun()

    st.markdown("""
    <div class="note" style="margin-top:8px;">
        💡 <b>Investigator Action:</b> Select any account in Tab 8 (Account Investigation) or sidebar to open its 360° investigation dossier.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 6: Transaction Network Graph (Interactive Plotly)
# =============================================================================
with tabs[5]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">6. Interactive Financial Crime Transaction Network Graph</div>', unsafe_allow_html=True)
    
    g_col1, g_col2, g_col3 = st.columns([1, 1, 2])
    g_tier = g_col1.selectbox("Filter Network by Risk Tier", ["All Tiers", "Critical", "High", "Medium", "Low"], index=0, key="net_tier_filt")
    max_nodes_disp = g_col2.slider("Maximum Nodes Displayed", 30, 250, 120, 10, key="net_max_nodes")
    focus_acc = g_col3.text_input("Highlight & Focus Account", value=st.session_state.get("selected_account", ""), key="net_focus_acc")

    filter_cat = None if g_tier == "All Tiers" else g_tier
    focus_target = focus_acc.strip() if focus_acc and focus_acc.strip() in G else None

    @st.cache_data(show_spinner=False)
    def _render_network_fig(_ds_key, max_nodes, filt_cat, focus_t):
        return ga.create_network_plotly_figure(
            G,
            account_risk_df=acc_risk,
            max_nodes=max_nodes,
            selected_account=focus_t,
            filter_category=filt_cat
        )

    with st.spinner("Rendering interactive transaction network..."):
        fig_net = _render_network_fig(
            st.session_state.get("dataset_key", "default"),
            max_nodes_disp,
            filter_cat,
            focus_target
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("""
    <div style="display:flex; gap:16px; align-items:center; font-size:12px; color:#5B6779; margin-top:4px;">
        <span><b>Legend:</b></span>
        <span><span style="color:#7F1D1D; font-size:15px;">●</span> Critical Risk</span>
        <span><span style="color:#DC2626; font-size:15px;">●</span> High Risk</span>
        <span><span style="color:#E69F00; font-size:15px;">●</span> Medium Risk</span>
        <span><span style="color:#1F9D8B; font-size:15px;">●</span> Low Risk</span>
        <span><span style="color:#12305E; font-size:15px;">●</span> Investigated / Focused Node</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 7: Graph Analytics & AML Typology Detection
# =============================================================================
with tabs[6]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">7. Graph Analytics & Suspicious Typology Detection</div>', unsafe_allow_html=True)
    
    pat_summary_df = pd.DataFrame([{
        "Pattern Typology": p.pattern_type,
        "Severity": p.severity,
        "Primary Account": p.primary_account,
        "Total Volume ($)": p.total_amount,
        "Description": p.description,
        "Underlying Evidence": p.evidence_summary
    } for p in patterns])

    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown(f"""
        <div class="panel">
            <div class="ph"><span class="pt">Detected Typologies Summary</span><span class="pn">{len(patterns)} alerts</span></div>
            <div style="font-size:13px; line-height:1.7;">
                • <b>Fan-In Smurfing Aggregations:</b> {sum(1 for p in patterns if 'Fan-In' in p.pattern_type)}<br>
                • <b>Fan-Out Rapid Disbursements:</b> {sum(1 for p in patterns if 'Fan-Out' in p.pattern_type)}<br>
                • <b>Circular Cycles (Round-Tripping):</b> {sum(1 for p in patterns if 'Circular' in p.pattern_type)}<br>
                • <b>Layering Chains (Pass-Through Shells):</b> {sum(1 for p in patterns if 'Layering' in p.pattern_type)}<br>
                • <b>Rapid Fund Movement (Pass-Through):</b> {sum(1 for p in patterns if 'Rapid' in p.pattern_type)}<br>
                • <b>Structuring Just Below $10k CTR:</b> {sum(1 for p in patterns if 'Structuring' in p.pattern_type)}<br>
                • <b>Mule Account Indicators:</b> {sum(1 for p in patterns if 'Mule' in p.pattern_type)}<br>
                • <b>High-Centrality Brokerage Hubs:</b> {sum(1 for p in patterns if 'Hub' in p.pattern_type)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown(f"""
        <div class="panel">
            <div class="ph"><span class="pt">Network Topology Summary</span><span class="pn">Graph Centrality</span></div>
            <div style="font-size:13px; line-height:1.7;">
                • <b>Total Graph Nodes (Entities):</b> {G.number_of_nodes():,}<br>
                • <b>Total Directed Edges (Flows):</b> {G.number_of_edges():,}<br>
                • <b>Network Density:</b> {nx.density(G):.5f}<br>
                • <b>Weakly Connected Components:</b> {nx.number_weakly_connected_components(G):,}<br>
                • <b>Max Degree Node:</b> {max(dict(G.degree()).items(), key=lambda t: t[1])[0]} ({max(dict(G.degree()).values())} links)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<b>Detailed Suspicious Topology Findings Register</b>", unsafe_allow_html=True)
    if pat_summary_df.empty:
        st.info("No suspicious structural typologies flagged.")
    else:
        st.dataframe(pat_summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 8: Interactive Account Investigation Dashboard
# =============================================================================
with tabs[7]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">8. Interactive Account Investigation Workbench</div>', unsafe_allow_html=True)

    # Account Selector inside tab
    acc_choices = list(acc_risk.index)
    cur_sel_acc = str(st.session_state.get("selected_account", acc_choices[0]))
    if cur_sel_acc not in acc_choices:
        cur_sel_acc = acc_choices[0]
        st.session_state["selected_account"] = cur_sel_acc
    cur_idx = acc_choices.index(cur_sel_acc)
    
    def _on_tab8_acc_change():
        st.session_state["selected_account"] = st.session_state["tab8_acc_selector"]

    if cur_sel_acc in acc_choices:
        st.session_state["tab8_acc_selector"] = cur_sel_acc

    sel_target_acc = st.selectbox(
        "Select Account to Investigate",
        acc_choices,
        format_func=lambda a: f"{a}  [Risk: {float(acc_risk.loc[a, 'risk_score'].iloc[0] if isinstance(acc_risk.loc[a, 'risk_score'], pd.Series) else acc_risk.loc[a, 'risk_score']):.0f} - {str(acc_risk.loc[a, 'risk_category'].iloc[0] if isinstance(acc_risk.loc[a, 'risk_category'], pd.Series) else acc_risk.loc[a, 'risk_category'])}]",
        key="tab8_acc_selector",
        on_change=_on_tab8_acc_change
    )

    target_acc = str(st.session_state.get("selected_account", sel_target_acc))
    # Guard: if selected_account drifted out of the index (e.g., after dataset reload), reset it
    if target_acc not in acc_risk.index:
        target_acc = acc_risk.index[0]
        st.session_state["selected_account"] = target_acc
    acc_row = acc_risk.loc[target_acc]
    if isinstance(acc_row, pd.DataFrame):
        acc_row = acc_row.iloc[0]
    r_score = float(acc_row.get("risk_score", 0.0))
    r_cat = str(acc_row.get("risk_category", "Low"))

    # KPI Row for Account
    k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
    k_col1.metric("Risk Score", f"{r_score:.1f} / 100", r_cat)
    k_col2.metric("Total Inflow", f"${acc_row.get('total_received', 0.0):,.2f}")
    k_col3.metric("Total Outflow", f"${acc_row.get('total_sent', 0.0):,.2f}")
    k_col4.metric("Net Retention", f"${acc_row.get('net_flow', 0.0):,.2f}")
    k_col5.metric("Transactions", f"{acc_row.get('total_transactions', 0):,}", f"{acc_row.get('suspicious_tx_count', 0)} Flagged")

    # Layout: Left column Ego Network, Right column Evidence & Typology explanations
    inv_left, inv_right = st.columns([1, 1])
    
    with inv_left:
        st.markdown('<div class="panel"><div class="ph"><span class="pt">Account Ego-Network (1-Hop Topology)</span></div>', unsafe_allow_html=True)
        fig_ego = ga.create_ego_network_plotly(G, target_acc, account_risk_df=acc_risk, hops=1)
        st.plotly_chart(fig_ego, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with inv_right:
        st.markdown('<div class="panel"><div class="ph"><span class="pt">Suspicious Activity & Typology Evidence</span></div>', unsafe_allow_html=True)
        
        acc_patterns = [p for p in patterns if str(target_acc) in [str(x) for x in getattr(p, "involved_accounts", [])]]
        if not acc_patterns:
            st.markdown("<p style='font-size:13px; color:#5B6779;'>No specific network topology patterns detected for this account.</p>", unsafe_allow_html=True)
        else:
            for p in acc_patterns:
                alert_cls = "alert" if p.severity in ("Critical", "High") else ""
                st.markdown(f"""
                <div class="narr-box {alert_cls}">
                    <b style="color:#12305E;">{p.pattern_type}</b> ({p.severity} Severity)<br>
                    {p.description}<br>
                    <span style="color:#5B6779; font-size:11.5px;"><b>Evidence:</b> {p.evidence_summary}</span>
                </div>
                """, unsafe_allow_html=True)

        if target_acc in st.session_state["graph_metrics"].index:
            gm = st.session_state["graph_metrics"].loc[target_acc]
            if isinstance(gm, pd.DataFrame):
                gm = gm.iloc[0]
            st.markdown(f"""
            <div style="font-size:12.5px; line-height:1.6; color:#101A2E; margin-top:8px;">
                <b>Graph Topology Metrics:</b><br>
                • In-Degree (Senders): {int(gm['in_degree'])} | Out-Degree (Receivers): {int(gm['out_degree'])}<br>
                • Betweenness Centrality: {float(gm['betweenness_centrality']):.5f}<br>
                • PageRank Score: {float(gm['pagerank']):.5f} | Community Cluster ID: #{int(gm['community_id'])}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Account Transaction Ledger (Inflows & Outflows)
    st.markdown('<div class="pt" style="margin-top:10px; margin-bottom:8px;">Account Transaction Ledger</div>', unsafe_allow_html=True)
    acc_txs = df_tx[(df_tx["sender_account"].astype(str) == str(target_acc)) | (df_tx["receiver_account"].astype(str) == str(target_acc))].copy()
    acc_txs["direction"] = np.where(acc_txs["receiver_account"].astype(str) == str(target_acc), "INFLOW", "OUTFLOW")
    acc_txs["counterparty"] = np.where(acc_txs["receiver_account"].astype(str) == str(target_acc), acc_txs["sender_account"], acc_txs["receiver_account"])

    acc_txs = acc_txs.sort_values("risk_score", ascending=False)
    
    try:
        cfg_atx = {
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0.0, max_value=100.0, format="%.1f"),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f")
        }
        st.dataframe(
            acc_txs[["transaction_id", "timestamp", "direction", "counterparty", "amount", "risk_score", "risk_category", "predicted_status"]],
            column_config=cfg_atx,
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        st.dataframe(acc_txs, use_container_width=True)


# =============================================================================
# TAB 9: SHAP Explainability
# =============================================================================
with tabs[8]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">9. SHAP Explainability & Risk Attribution Studio</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px; color:#5B6779; margin-bottom:12px;">
        SHAP (SHapley Additive exPlanations) attributes the Random Forest model's predictions to specific transactional and topological features.
        Positive contributions (Red) increase the predicted laundering risk; negative contributions (Teal) reduce it towards normal baselines.
    </div>
    """, unsafe_allow_html=True)

    sup          = st.session_state["supervised_res"]
    unsup        = st.session_state["unsupervised_res"]
    cleaned_bundle = st.session_state["cleaned_bundle"]
    _ds_key      = st.session_state.get("dataset_key", "default")

    # ── mode banner ──────────────────────────────────────────────────────────
    if sup is not None:
        st.markdown("""
        <div style="background:#EAF6F4; border-left:3px solid #1F9D8B; padding:8px 14px;
                    border-radius:0 4px 4px 0; font-size:12.5px; margin-bottom:12px;">
            <b>Supervised Mode — Random Forest</b>: SHAP TreeExplainer attributes exact
            probability shifts to individual features across each transaction.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#FEF3C7; border-left:3px solid #E69F00; padding:8px 14px;
                    border-radius:0 4px 4px 0; font-size:12.5px; margin-bottom:12px;">
            <b>Unsupervised Mode — Isolation Forest</b>: No fraud labels were detected.
            The platform shows anomaly score distributions and per-feature deviation
            contributions from the trained Isolation Forest model.
            <br><i>Compliance notice: anomaly scores indicate statistical rarity, not confirmed illicit activity.</i>
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # SUPERVISED PATH  ── Random Forest + SHAP
    # =========================================================================
    if sup is not None:
        @st.cache_resource(show_spinner=False)
        def _build_explainer(_model, _X_bg, _key):
            return exp.AMLModelExplainer(_model, _X_bg)

        @st.cache_data(show_spinner="Computing global feature importance…")
        def _global_importance(_key, _cols_tuple):
            _expl = _build_explainer(sup.fitted_model, cleaned_bundle.X, _key)
            sample = cleaned_bundle.X.sample(min(25, len(cleaned_bundle.X)), random_state=42)
            return _expl.explain_global(sample)

        cols_key = f"{_ds_key}_{len(cleaned_bundle.X.columns)}"
        explainer   = _build_explainer(sup.fitted_model, cleaned_bundle.X, cols_key)
        global_imp  = _global_importance(cols_key, tuple(cleaned_bundle.X.columns))

        sh_col1, sh_col2 = st.columns([1, 1])
        with sh_col1:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Global Feature Importance (Mean Absolute SHAP)</span></div>', unsafe_allow_html=True)
            st.plotly_chart(exp.plot_shap_summary(global_imp, top_n=10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with sh_col2:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Individual Transaction Waterfall Attribution</span></div>', unsafe_allow_html=True)

            shap_target_acc = str(st.session_state.get("selected_account", ""))
            acc_tx_indices  = df_tx[
                (df_tx["sender_account"].astype(str)  == shap_target_acc) |
                (df_tx["receiver_account"].astype(str) == shap_target_acc)
            ].index.tolist()
            if not acc_tx_indices:
                acc_tx_indices = list(range(min(50, len(cleaned_bundle.X))))

            sample_tx_idx = st.selectbox(
                "Select Transaction to Explain", acc_tx_indices,
                format_func=lambda idx: (
                    f"TX #{idx} — {df_tx.loc[idx, 'transaction_id']} "
                    f"(${df_tx.loc[idx, 'amount']:,.0f}  Score: {df_tx.loc[idx, 'risk_score']:.0f})"
                ), key="shap_tx_sel"
            )
            inst_exp = explainer.explain_instance(cleaned_bundle.X.iloc[sample_tx_idx])
            st.plotly_chart(exp.plot_waterfall(inst_exp, top_display=7), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="ph"><span class="pt">Plain-English AML Investigator Narratives</span></div>', unsafe_allow_html=True)
        for narr in exp.generate_investigator_narrative(inst_exp, top_k=5):
            st.markdown(f"<div class='narr-box'>{narr}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # UNSUPERVISED PATH  ── Isolation Forest Anomaly Explainability
    # =========================================================================
    else:
        @st.cache_resource(show_spinner=False)
        def _build_iso_explainer(_model, _X_bg, _key):
            return exp.AMLIsolationExplainer(_model, _X_bg)

        @st.cache_data(show_spinner="Computing anomaly feature contributions…")
        def _iso_global_imp(_key, _cols_tuple):
            _expl  = _build_iso_explainer(unsup.fitted_model, cleaned_bundle.X, _key)
            sample = cleaned_bundle.X.sample(min(30, len(cleaned_bundle.X)), random_state=42)
            return _expl.explain_global(sample)

        iso_cols_key = f"{_ds_key}_{len(cleaned_bundle.X.columns)}"
        iso_explainer  = _build_iso_explainer(unsup.fitted_model, cleaned_bundle.X, iso_cols_key)
        iso_global_imp = _iso_global_imp(iso_cols_key, tuple(cleaned_bundle.X.columns))

        # ── Row 1: Score distribution  +  Global feature importance ─────────
        u_col1, u_col2 = st.columns([1, 1])

        with u_col1:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Anomaly Score Distribution</span><span class="pn">All transactions</span></div>', unsafe_allow_html=True)
            # df_tx.risk_score is 0-100 scale already
            fig_dist = exp.plot_anomaly_score_distribution(df_tx["risk_score"].values)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Summary stats
            n_total  = len(df_tx)
            n_crit   = int((df_tx["risk_category"] == "Critical").sum())
            n_high   = int((df_tx["risk_category"] == "High").sum())
            n_med    = int((df_tx["risk_category"] == "Medium").sum())
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-top:4px;">
                <div style="background:#F8FAFC; border-radius:4px; padding:8px; text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#12305E;">{n_total:,}</div>
                    <div style="font-size:10px; color:#5B6779; text-transform:uppercase;">Total TXs</div>
                </div>
                <div style="background:#FFF5F5; border-radius:4px; padding:8px; text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#8B0000;">{n_crit:,}</div>
                    <div style="font-size:10px; color:#5B6779; text-transform:uppercase;">Critical</div>
                </div>
                <div style="background:#FEF3C7; border-radius:4px; padding:8px; text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#B45309;">{n_high:,}</div>
                    <div style="font-size:10px; color:#5B6779; text-transform:uppercase;">High</div>
                </div>
                <div style="background:#EAF6F4; border-radius:4px; padding:8px; text-align:center;">
                    <div style="font-size:18px; font-weight:700; color:#1F9D8B;">{n_med:,}</div>
                    <div style="font-size:10px; color:#5B6779; text-transform:uppercase;">Medium</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with u_col2:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Feature Anomaly Contributions (Global)</span><span class="pn">Most anomaly-driving features</span></div>', unsafe_allow_html=True)
            st.plotly_chart(exp.plot_anomaly_global_importance(iso_global_imp, top_n=12), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Per-transaction anomaly waterfall ─────────────────────────
        st.markdown('<div class="pt" style="margin:12px 0 6px;">Per-Transaction Anomaly Explanation</div>', unsafe_allow_html=True)
        u2_col1, u2_col2 = st.columns([1, 1])

        with u2_col1:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Select a Transaction to Explain</span></div>', unsafe_allow_html=True)

            iso_target_acc = str(st.session_state.get("selected_account", ""))
            iso_tx_idx = df_tx[
                (df_tx["sender_account"].astype(str)  == iso_target_acc) |
                (df_tx["receiver_account"].astype(str) == iso_target_acc)
            ].index.tolist()
            if not iso_tx_idx:
                iso_tx_idx = df_tx.nlargest(50, "risk_score").index.tolist()

            iso_sel_idx = st.selectbox(
                "Transaction", iso_tx_idx,
                format_func=lambda idx: (
                    f"TX #{idx} — {df_tx.loc[idx, 'transaction_id']} "
                    f"(${df_tx.loc[idx, 'amount']:,.0f}  Anomaly: {df_tx.loc[idx, 'risk_score']:.0f})"
                ), key="iso_tx_sel"
            )
            iso_inst = iso_explainer.explain_instance(cleaned_bundle.X.iloc[iso_sel_idx])

            # Mini KPI for selected transaction
            tx_row = df_tx.loc[iso_sel_idx]
            cat_color = "#8B0000" if tx_row["risk_category"] in ("Critical","High") else "#E69F00"
            st.markdown(f"""
            <div style="margin-top:10px; display:grid; grid-template-columns:repeat(3,1fr); gap:8px;">
                <div style="background:#F8FAFC; border-radius:5px; padding:10px; border-top:3px solid {cat_color};">
                    <div style="font-size:22px; font-weight:700; color:{cat_color};">{tx_row['risk_score']:.0f}</div>
                    <div style="font-size:11px; color:#5B6779; text-transform:uppercase;">Anomaly Score</div>
                    <div style="font-size:11px; font-weight:600; color:{cat_color};">{tx_row['risk_category']}</div>
                </div>
                <div style="background:#F8FAFC; border-radius:5px; padding:10px;">
                    <div style="font-size:18px; font-weight:700; color:#12305E;">${float(tx_row['amount']):,.0f}</div>
                    <div style="font-size:11px; color:#5B6779; text-transform:uppercase;">Amount</div>
                </div>
                <div style="background:#F8FAFC; border-radius:5px; padding:10px;">
                    <div style="font-size:14px; font-weight:700; color:#12305E;">{tx_row['predicted_status']}</div>
                    <div style="font-size:11px; color:#5B6779; text-transform:uppercase;">Status</div>
                </div>
            </div>
            <div style="margin-top:8px; font-size:12px; color:#5B6779;">
                <b>Method:</b> {iso_inst['method']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with u2_col2:
            st.markdown('<div class="panel"><div class="ph"><span class="pt">Anomaly Score Waterfall</span><span class="pn">Feature-level attribution</span></div>', unsafe_allow_html=True)
            st.plotly_chart(exp.plot_anomaly_waterfall(iso_inst, top_display=7), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 3: Investigator narratives ───────────────────────────────────
        st.markdown('<div class="panel"><div class="ph"><span class="pt">Anomaly Investigation Narratives</span><span class="pn">Plain-English compliance findings</span></div>', unsafe_allow_html=True)
        for narr in exp.generate_anomaly_narrative(iso_inst, top_k=5):
            alert_cls = "alert" if iso_inst["predicted_proba"] > 0.65 else ""
            st.markdown(f"<div class='narr-box {alert_cls}'>{narr}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 10: Reports and Downloads
# =============================================================================
with tabs[9]:
    st.markdown('<div class="pt" style="margin-bottom:8px;">10. Compliance Reports & Audit Package Downloads</div>', unsafe_allow_html=True)
    
    # Always fetch from session state so Tab 10 is independent of Tab 9 execution order
    _t10_bundle = st.session_state["cleaned_bundle"]
    rep_col1, rep_col2 = st.columns([1, 1])

    with rep_col1:
        target_acc = str(st.session_state.get("selected_account", acc_risk.index[0]))
        if target_acc not in acc_risk.index:
            target_acc = acc_risk.index[0]
        acc_summary_row = acc_risk.loc[target_acc]
        if isinstance(acc_summary_row, pd.DataFrame):
            acc_summary_row = acc_summary_row.iloc[0]
        acc_summary = acc_summary_row.to_dict()

        _r_score   = float(acc_summary.get("risk_score", 0.0))
        _r_cat     = str(acc_summary.get("risk_category", "Low"))
        _inflow    = float(acc_summary.get("total_received", 0.0))
        _outflow   = float(acc_summary.get("total_sent", 0.0))
        _net       = float(acc_summary.get("net_flow", 0.0))
        _tx_tot    = int(acc_summary.get("total_transactions", 0))
        _tx_susp   = int(acc_summary.get("suspicious_tx_count", 0))
        _pat_count = int(acc_summary.get("pattern_count", 0))
        _badge_col = "#8B0000" if _r_cat in ("Critical", "High") else ("#E69F00" if _r_cat == "Medium" else "#1F9D8B")

        st.markdown(f"""
        <div class="panel">
            <div class="ph"><span class="pt">Investigation Dossier for Current Account</span></div>
            <p style="font-size:13px; color:#101A2E; line-height:1.6; margin-bottom:10px;">
                Formal compliance dossier for account <b>{target_acc}</b>. Includes risk metrics,
                topological position, suspicious pattern evidence, SHAP drivers, and transaction ledger.
            </p>
            <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:8px; margin-bottom:12px;">
                <div style="background:#F8FAFC; border:1px solid #D5DBE5; border-radius:5px; padding:10px 12px; border-top:3px solid {_badge_col};">
                    <div style="font-size:20px; font-weight:700; color:{_badge_col};">{_r_score:.0f}</div>
                    <div style="font-size:11px; color:#5B6779; font-weight:600; text-transform:uppercase; margin-top:2px;">Risk Score</div>
                    <div style="font-size:11px; color:{_badge_col}; font-weight:700;">{_r_cat}</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D5DBE5; border-radius:5px; padding:10px 12px;">
                    <div style="font-size:17px; font-weight:700; color:#12305E;">${_inflow:,.0f}</div>
                    <div style="font-size:11px; color:#5B6779; font-weight:600; text-transform:uppercase; margin-top:2px;">Total Inflow</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D5DBE5; border-radius:5px; padding:10px 12px;">
                    <div style="font-size:17px; font-weight:700; color:#12305E;">${_outflow:,.0f}</div>
                    <div style="font-size:11px; color:#5B6779; font-weight:600; text-transform:uppercase; margin-top:2px;">Total Outflow</div>
                </div>
                <div style="background:#F8FAFC; border:1px solid #D5DBE5; border-radius:5px; padding:10px 12px;">
                    <div style="font-size:17px; font-weight:700; color:#D7263D;">{_tx_susp}</div>
                    <div style="font-size:11px; color:#5B6779; font-weight:600; text-transform:uppercase; margin-top:2px;">Flagged TXs</div>
                    <div style="font-size:11px; color:#5B6779;">of {_tx_tot} total · {_pat_count} patterns</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        @st.cache_data(show_spinner=False)
        def _get_account_dossiers(_target_acc, _ds_key):
            _summary_row = acc_risk.loc[_target_acc]
            if isinstance(_summary_row, pd.DataFrame):
                _summary_row = _summary_row.iloc[0]
            _summary = _summary_row.to_dict()

            _gm_row = st.session_state["graph_metrics"].loc[_target_acc] if _target_acc in st.session_state["graph_metrics"].index else {}
            if isinstance(_gm_row, pd.DataFrame):
                _gm_row = _gm_row.iloc[0]
            _gm = _gm_row.to_dict() if hasattr(_gm_row, "to_dict") else {}

            _pats = [p for p in patterns if _target_acc in [str(x) for x in getattr(p, "involved_accounts", [])]]
            _txs = df_tx[(df_tx["sender_account"].astype(str) == _target_acc) | (df_tx["receiver_account"].astype(str) == _target_acc)]

            _narrs = []
            if st.session_state["supervised_res"] is not None and not _txs.empty:
                _first_idx = _txs.index[0]
                _bundle = st.session_state["cleaned_bundle"]
                _expl = exp.AMLModelExplainer(st.session_state["supervised_res"].fitted_model, _bundle.X)
                _inst = _expl.explain_instance(_bundle.X.iloc[_first_idx])
                _narrs = exp.generate_investigator_narrative(_inst, top_k=5)

            _html = rg.generate_account_investigation_html(_target_acc, _summary, _gm, _pats, _txs, _narrs)
            _pdf = rg.generate_account_investigation_pdf(_target_acc, _summary, _gm, _pats, _txs, _narrs)
            return _html, _pdf

        @st.cache_data(show_spinner=False)
        def _get_csv_bundle(_ds_key):
            return rg.export_bundle_to_csv_dict(
                df_cleaned=_t10_bundle.df_raw_cleaned,
                df_tx_risk=df_tx,
                df_account_risk=acc_risk,
                df_metrics=st.session_state["graph_metrics"],
                pattern_findings=patterns
            )

        html_dossier, pdf_bytes = _get_account_dossiers(target_acc, st.session_state.get("dataset_key", "default"))
        csv_bundle = _get_csv_bundle(st.session_state.get("dataset_key", "default"))

        st.download_button(
            label="Download Formal PDF Dossier",
            data=pdf_bytes,
            file_name=f"AML_Investigation_Dossier_{target_acc}.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        st.download_button(
            label="Download Standalone HTML Dossier (Printable)",
            data=html_dossier.encode("utf-8"),
            file_name=f"AML_Investigation_Dossier_{target_acc}.html",
            mime="text/html"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with rep_col2:
        st.markdown("""
        <div class="panel">
            <div class="ph"><span class="pt">Audit Trail CSV Export Bundle</span></div>
            <p style="font-size:13px; color:#101A2E; line-height:1.6;">
                Export comprehensive machine-readable CSV audit files for compliance databases, regulatory reporting, or core banking ingestion.
            </p>
        """, unsafe_allow_html=True)

        st.download_button(
            label="Download Cleaned Transactions (CSV)",
            data=csv_bundle["cleaned_transactions.csv"].encode("utf-8"),
            file_name="cleaned_transactions.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Transaction Risk Register (CSV)",
            data=csv_bundle["transaction_risk_register.csv"].encode("utf-8"),
            file_name="transaction_risk_register.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Account Risk Register (CSV)",
            data=csv_bundle["account_risk_register.csv"].encode("utf-8"),
            file_name="account_risk_register.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Graph Centrality Metrics (CSV)",
            data=csv_bundle["graph_analytics_metrics.csv"].encode("utf-8"),
            file_name="graph_analytics_metrics.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Suspicious Typologies Log (CSV)",
            data=csv_bundle["suspicious_patterns_findings.csv"].encode("utf-8"),
            file_name="suspicious_patterns_findings.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
