"""
Hospital OPD Queue and Bed Allocation Optimization Dashboard.
Interactive Streamlit Application with Discrete-Event Simulation, Machine Learning,
and Reinforcement Learning.

Four Core Innovations:
1. Fairness-Aware Aging Priority (acuity + alpha * wait_time)
2. Predict-then-Decide (ML-predicted service time & bed pre-reservation)
3. Demand Regime Detection (K-Means & Agglomerative Hierarchical Clustering)
4. Explainable RL Policy Distillation (Surrogate Decision Tree with Fidelity)
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


def resolve_asset_path(rel_path: str) -> str:
    """Resolves relative file path against current working directory or workspace root."""
    if os.path.exists(rel_path):
        return rel_path
    candidate = os.path.join(workspace_root, rel_path)
    if os.path.exists(candidate):
        return candidate
    return rel_path

import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.clustering import OPDClusteringAnalysis
from src.evaluate import HospitalEvaluator, compute_all_innovations_before_after, sweep_aging_alpha_busy
from src.explain_policy import PolicyDistiller
from src.fairness import audit_demographic_fairness, sweep_alpha_tradeoff
from src.ml_models import MLSuite
from src.policies import run_policy_simulation
from src.preprocessing_eda import clean_missing_values, load_raw_data
from src.recommender import HospitalRecommenderSuite
from src.simulation import OPDConfig
from src.validation import validate_queuing_theory


@st.cache_data
def get_cached_before_after_comparisons() -> dict:
    json_path = resolve_asset_path("data/before_after_comparisons.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return compute_all_innovations_before_after(num_seeds=30)


@st.cache_data
def get_cached_alpha_sweep_busy() -> pd.DataFrame:
    csv_path = resolve_asset_path("data/alpha_sweep_busy.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return sweep_aging_alpha_busy(num_seeds=10)


# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom Clinical Theme CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hospital OPD Optimizer & Sim",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling
st.markdown("""
<style>
    /* Global Typography & Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Badge */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 4px 0;
        color: #f8fafc;
    }
    .metric-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-yellow { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    
    /* Innovation Showcase Callout */
    .innovation-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 5px solid #0284c7;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Before / After Comparison Grid */
    .compare-container {
        display: flex;
        gap: 16px;
        margin-top: 12px;
    }
    .compare-col {
        flex: 1;
        padding: 16px;
        border-radius: 12px;
    }
    .compare-before {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .compare-after {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching Data, Models, and Recommenders
# ---------------------------------------------------------------------------

@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    raw_df = load_raw_data(resolve_asset_path("data/synthetic_opd.csv"))
    cleaned_df, _ = clean_missing_values(raw_df)
    return cleaned_df


@st.cache_resource
def get_cached_recommender():
    return HospitalRecommenderSuite()


@st.cache_resource
def get_cached_clustering():
    return OPDClusteringAnalysis()


@st.cache_resource
def get_cached_ml_models():
    suite = MLSuite()
    table = suite.run_all()
    return suite, table


# ---------------------------------------------------------------------------
# Sidebar Controls & Global Configuration
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 🏥 Operational Scenario Mode")
op_scenario = st.sidebar.radio(
    "Select Operating Environment:",
    options=[
        "🔥 Busy Hospital (High Congestion: 85-95% Peak Util)",
        "🌿 Quiet / Normal Hospital (65-75% Util)"
    ],
    index=0,
    help="Busy scenario simulates high congestion (85-95% doctor utilization) where queuing delays and policy differences are pronounced. Quiet scenario simulates lighter operational volume."
)
is_busy_scenario = "Busy Hospital" in op_scenario

st.sidebar.markdown("### ⚙️ OPD Operational Controls")

# Surge Button state toggle
if "surge_active" not in st.session_state:
    st.session_state.surge_active = False

def toggle_surge():
    st.session_state.surge_active = not st.session_state.surge_active

if st.sidebar.button("🚨 Simulate Morning Surge (+60% Volume)", on_click=toggle_surge):
    pass

if st.session_state.surge_active:
    st.sidebar.warning("⚡ **SURGE ACTIVE:** Arrival volume boosted by 1.6x!")

# Interactive Sidebar Sliders (defaults calibrated to selected scenario)
default_num_docs = 5 if is_busy_scenario else 6
num_doctors = st.sidebar.slider("👨‍⚕️ Number of Active Doctors", min_value=3, max_value=12, value=default_num_docs, step=1)
num_general_beds = st.sidebar.slider("🛏️ General Ward Beds", min_value=15, max_value=80, value=30 if is_busy_scenario else 40, step=5)
num_icu_beds = st.sidebar.slider("🚨 ICU Critical Beds", min_value=4, max_value=25, value=8 if is_busy_scenario else 10, step=1)

base_arr_mult = 1.60 if st.session_state.surge_active else 1.0
arrival_rate_mult = st.sidebar.slider(
    "📈 Arrival Rate Multiplier",
    min_value=0.5,
    max_value=2.5,
    value=float(base_arr_mult),
    step=0.1
)

policy_choice = st.sidebar.selectbox(
    "🎯 Dispatch & Queue Policy",
    options=[
        "aging_priority",
        "fcfs",
        "strict_priority",
        "sjf",
        "predict_then_decide",
        "regime_aware"
    ],
    format_func=lambda x: {
        "aging_priority": "Aging Priority (Fairness-Aware)",
        "fcfs": "FCFS (First-Come, First-Served)",
        "strict_priority": "Strict Acuity Priority",
        "sjf": "Shortest-Job-First (SJF)",
        "predict_then_decide": "Predict-then-Decide (ML-Assisted)",
        "regime_aware": "Regime-Aware Dynamic Staffing",
    }.get(x, x)
)

alpha_val = 2.0 if is_busy_scenario else 0.5
if policy_choice in ["aging_priority", "predict_then_decide", "regime_aware"]:
    alpha_val = st.sidebar.slider(
        "⚖️ Aging Parameter (alpha)",
        min_value=0.0,
        max_value=5.0,
        value=float(alpha_val),
        step=0.1,
        help="Priority = Acuity + alpha * Wait_Time. Under busy conditions, alpha >= 1.0 reaches overtaking threshold (w_urgent - w_routine)/alpha."
    )

st.sidebar.markdown("### 🩺 Staff Availability & Backup Pool")
manual_absent_docs = st.sidebar.multiselect(
    "🚫 Mark Doctor(s) on Leave/Absent",
    options=list(range(1, num_doctors + 1)),
    default=[],
    format_func=lambda x: f"Doctor #{x}",
    help="Select specific doctors to manually place on whole-day leave or absence."
)

num_floating_doctors = st.sidebar.slider(
    "🛟 Floating / On-Call Doctor Pool",
    min_value=0,
    max_value=4,
    value=2,
    help="On-call doctors activated during Peak/Surge regimes or when scheduled doctors are absent."
)

num_junior_staff = st.sidebar.slider(
    "👩‍⚕️ Junior Staff / Nurses (Task Shifting)",
    min_value=0,
    max_value=4,
    value=2 if is_busy_scenario else 1,
    help="Second server type: handles Routine patients only, keeping senior doctors free for emergencies."
)

teleconsult_cap = st.sidebar.slider(
    "💻 Teleconsultation Virtual Channels",
    min_value=0,
    max_value=4,
    value=0 if is_busy_scenario else 2,
    help="Virtual consultation track for routine patients."
)

emergency_override_active = st.sidebar.checkbox(
    "🚨 Enable Emergency Override",
    value=True,
    help="Allows emergency arrivals to preempt routine consultations or summon on-call floating doctors."
)

sim_seed = st.sidebar.number_input("🎲 Simulation Random Seed", min_value=1, max_value=9999, value=42, step=1)

# Build current simulation config calibrated to chosen operational mode
if is_busy_scenario:
    current_config = OPDConfig.create_busy_scenario(
        sim_days=1,
        num_doctors=num_doctors,
        num_general_beds=num_general_beds,
        num_icu_beds=num_icu_beds,
        arrival_multiplier=arrival_rate_mult,
        policy=policy_choice,
        alpha=alpha_val,
        manual_absent_doctors=manual_absent_docs,
        num_floating_doctors=num_floating_doctors,
        enable_floating_pool=num_floating_doctors > 0,
        num_junior_doctors_nurses=num_junior_staff,
        enable_task_shifting=num_junior_staff > 0,
        teleconsultation_capacity=teleconsult_cap,
        enable_emergency_override=emergency_override_active,
        seed=sim_seed
    )
else:
    current_config = OPDConfig.create_quiet_scenario(
        sim_days=1,
        num_doctors=num_doctors,
        num_general_beds=num_general_beds,
        num_icu_beds=num_icu_beds,
        arrival_multiplier=arrival_rate_mult,
        policy=policy_choice,
        alpha=alpha_val,
        manual_absent_doctors=manual_absent_docs,
        num_floating_doctors=num_floating_doctors,
        enable_floating_pool=num_floating_doctors > 0,
        num_junior_doctors_nurses=num_junior_staff,
        enable_task_shifting=num_junior_staff > 0,
        teleconsultation_capacity=teleconsult_cap,
        enable_emergency_override=emergency_override_active,
        seed=sim_seed
    )


# ---------------------------------------------------------------------------
# Hero Header Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">🏥 Hospital OPD Queue & Bed Allocation Optimization</h1>
            <p class="hero-subtitle">Discrete-Event Simulation (SimPy) • Classical Machine Learning • Explainable Reinforcement Learning</p>
        </div>
        <div style="text-align: right;">
            <span class="metric-badge badge-blue">Production v1.0</span>
            <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">Data Science Capstone Project</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "🚀 Live OPD Simulator",
    "✨ The 4 Innovations",
    "📊 Policy Benchmark & 30-Seed Tests",
    "🤖 Clinical Recommenders",
    "⚖️ Fairness & Demographics",
    "🔬 M/M/c & Little's Law Validation",
    "🧪 What-If Stress Testing"
])


# ===========================================================================
# TAB 1: Live OPD Simulator & KPI Command Center
# ===========================================================================

with tabs[0]:
    st.markdown("### 🎛️ OPD Live Command Center & Real-Time SimPy Engine")
    with st.spinner("Simulating Hospital OPD Day in SimPy..."):
        sim_df, kpis, sim_instance = run_policy_simulation(
            policy_name=policy_choice,
            base_config=current_config,
            alpha=alpha_val,
            seed=sim_seed,
            return_sim=True
        )

    # Glowing Metric KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Arrived</div>
            <div class="metric-value">{kpis['total_patients']}</div>
            <span class="metric-badge badge-blue">12-Hour Shift</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        wait_badge = "badge-green" if kpis['mean_wait'] < 18 else ("badge-yellow" if kpis['mean_wait'] < 30 else "badge-red")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Wait Time</div>
            <div class="metric-value">{kpis['mean_wait']:.1f}<span style="font-size:1rem;color:#94a3b8;">m</span></div>
            <span class="metric-badge {wait_badge}">95th: {kpis['p95_wait']:.1f}m</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        em_badge = "badge-green" if kpis['emergency_wait'] < 10 else "badge-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Emergency Wait</div>
            <div class="metric-value">{kpis['emergency_wait']:.1f}<span style="font-size:1rem;color:#94a3b8;">m</span></div>
            <span class="metric-badge {em_badge}">Target: &lt; 10m</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        ro_badge = "badge-green" if kpis['routine_wait'] < 25 else "badge-yellow"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Routine Wait</div>
            <div class="metric-value">{kpis['routine_wait']:.1f}<span style="font-size:1rem;color:#94a3b8;">m</span></div>
            <span class="metric-badge {ro_badge}">Starvation: {kpis['routine_starvation_rate']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        util_badge = "badge-green" if 60 <= kpis['doctor_utilization'] <= 90 else "badge-yellow"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Doctor Utilization</div>
            <div class="metric-value">{kpis['doctor_utilization']:.1f}<span style="font-size:1rem;color:#94a3b8;">%</span></div>
            <span class="metric-badge {util_badge}">Capacity: {num_doctors} Docs</span>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        rej_badge = "badge-green" if kpis['bed_rejection_rate'] < 10 else "badge-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bed Rejection</div>
            <div class="metric-value">{kpis['bed_rejection_rate']:.1f}<span style="font-size:1rem;color:#94a3b8;">%</span></div>
            <span class="metric-badge {rej_badge}">Beds: {num_general_beds+num_icu_beds} Total</span>
        </div>
        """, unsafe_allow_html=True)

    # Operational Visualizations: Hourly Wait Time & Triage Breakdown
    st.markdown("#### 📈 OPD Queue Flow & Hourly Wait Trajectory")
    c_left, c_right = st.columns([7, 5])

    with c_left:
        fig, ax = plt.subplots(figsize=(8, 4))
        hourly_summary = sim_df.groupby("hour_of_day")["wait_time"].mean().reset_index()
        ax.plot(hourly_summary["hour_of_day"], hourly_summary["wait_time"],
                color="#0284c7", lw=2.5, marker="o", label="Average Wait Time (min)")
        ax.axvspan(9.0, 12.0, color="#f87171", alpha=0.15, label="Morning Peak Window (9am-12pm)")
        ax.set_title("Simulated Patient Wait Time Progression Across Operating Day", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of Day (8:00 AM - 8:00 PM)", fontsize=10)
        ax.set_ylabel("Average Wait Time (Minutes)", fontsize=10)
        ax.legend(frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        triage_counts = sim_df["triage_class"].value_counts()
        colors = ["#22c55e", "#f59e0b", "#ef4444"]
        ax.pie(triage_counts, labels=triage_counts.index, autopct="%1.1f%%",
               colors=colors, startangle=140, explode=[0.02, 0.02, 0.08])
        ax.set_title("Simulated Patient Triage Volume Share", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Live Patient Stream Table
    with st.expander("🔍 View Live Simulated Patient Record Log (First 100 rows)"):
        st.dataframe(
            sim_df[["patient_id", "hour_of_day", "age_group", "gender", "area", "symptoms", "triage_class", "server_type", "service_time", "wait_time", "was_preempted", "admitted", "bed_type", "bed_rejection"]].head(100),
            use_container_width=True
        )

    # -----------------------------------------------------------------------
    # Doctor Availability Status Board & Timeline
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🩺 Doctor Availability Roster & Status Monitor")
    st.markdown("Real-time clinical roster tracking doctors across shifts, staggered breaks (max 25% concurrent), sudden absences, and floating pool activation.")

    doc_statuses = sim_instance.get_current_doctor_statuses()
    
    # Render doctor status cards in grid
    d_cols = st.columns(min(6, len(doc_statuses)))
    for idx, d_info in enumerate(doc_statuses[:12]):
        col_target = d_cols[idx % len(d_cols)]
        status = d_info["status"]
        status_badge = {
            "Available": "badge-green",
            "Busy": "badge-blue",
            "Break": "badge-yellow",
            "Leave": "badge-blue",
            "Absent": "badge-red",
            "Offline": "badge-blue",
        }.get(status, "badge-blue")

        with col_target:
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.7);border-radius:10px;padding:12px;margin-bottom:10px;border:1px solid rgba(255,255,255,0.08);">
                <div style="font-weight:700;font-size:0.92rem;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{d_info['name']}</div>
                <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:4px;">{d_info['role']}</div>
                <span class="metric-badge {status_badge}">{status.upper()}</span>
                <div style="font-size:0.75rem;color:#cbd5e1;margin-top:6px;">Treated: <strong>{d_info['patients_treated']}</strong> patients</div>
            </div>
            """, unsafe_allow_html=True)

    # Doctor Availability Timeline (Gantt Chart)
    st.markdown("#### 📅 Doctor Availability & Shift Schedule Timeline (8:00 AM - 8:00 PM)")
    timeline_df = sim_instance.get_staff_timeline_dataframe()

    if not timeline_df.empty:
        fig_gantt, ax_gantt = plt.subplots(figsize=(11, 4.5))
        
        status_color_map = {
            "Available": "#10b981",
            "Busy": "#2563eb",
            "Break": "#f59e0b",
            "Leave": "#64748b",
            "Absent": "#ef4444",
            "Offline": "#334155",
        }

        staff_names = list(timeline_df["Staff_Name"].unique())
        y_positions = {name: idx for idx, name in enumerate(staff_names)}

        for _, row in timeline_df.iterrows():
            y_pos = y_positions[row["Staff_Name"]]
            start_h = row["Start_Hour"]
            dur_h = max(0.05, row["End_Hour"] - start_h)
            c = status_color_map.get(row["Status"], "#94a3b8")
            ax_gantt.barh(y_pos, dur_h, left=start_h, height=0.55, color=c, edgecolor="none", alpha=0.9)

        ax_gantt.set_yticks(range(len(staff_names)))
        ax_gantt.set_yticklabels(staff_names, fontsize=9)
        ax_gantt.set_xlim(8.0, 20.0)
        ax_gantt.set_xticks(range(8, 21))
        ax_gantt.set_xticklabels([f"{h}:00" for h in range(8, 21)], fontsize=9)
        ax_gantt.set_xlabel("Time of Day (Operating Shift)", fontsize=10)
        ax_gantt.set_title("Doctor Activity Timeline: Consultation, Staggered Breaks, Absences, and On-Call Mobilization", fontsize=11, fontweight="bold")
        ax_gantt.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Custom legend
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=status_color_map[k])
            for k in ["Available", "Busy", "Break", "Leave", "Absent", "Offline"]
        ]
        ax_gantt.legend(legend_handles, ["Available", "Busy (Patient)", "Break (Lunch/Tea)", "Whole-Day Leave", "Sudden Absence", "On-Call / Offline"],
                        loc="upper right", fontsize=8, ncol=3, frameon=True)
        plt.tight_layout()
        st.pyplot(fig_gantt)
        plt.close(fig_gantt)


# ===========================================================================
# TAB 2: The 4 Innovations (Deep Dive & Before/After Comparisons)
# ===========================================================================

with tabs[1]:
    st.markdown("## 💡 The Four Core Innovations")
    st.markdown("This project introduces four novel healthcare operations and data science innovations. Below is an in-depth breakdown and empirical before/after comparison for each innovation.")

    all_comparisons = get_cached_before_after_comparisons()

    col_bench_hdr1, col_bench_hdr2 = st.columns([8, 4])
    with col_bench_hdr1:
        st.caption("All Before vs After comparisons below are empirically audited using Common Random Numbers (CRN, 30 seeds), reporting Mean ± 95% CI and paired Student's t-test p-values under high-congestion conditions.")
    with col_bench_hdr2:
        if st.button("🔄 Recalculate 30-Seed CRN Benchmarks", key="recalc_crn_tab2"):
            with st.spinner("Computing 30-seed paired benchmarks with Common Random Numbers..."):
                all_comparisons = compute_all_innovations_before_after(num_seeds=30)
                st.success("Recomputed all 30-seed CRN benchmarks!")

    innov_tabs = st.tabs([
        "1. Fairness-Aware Aging Priority",
        "2. Predict-then-Decide Queueing",
        "3. Demand Regime Detection",
        "4. Explainable RL Distillation"
    ])

    # -----------------------------------------------------------------------
    # INNOVATION 1: Fairness-Aware Aging Priority
    # -----------------------------------------------------------------------
    with innov_tabs[0]:
        st.markdown("""
        <div class="innovation-box">
            <h3 style="color:#38bdf8;margin-top:0;">Innovation 1: Fairness-Aware Aging Priority</h3>
            <p><strong>Formula:</strong> <code>Priority Score(t) = Acuity_Weight + &alpha; &times; Wait_Time(t)</code></p>
            <p>In standard emergency departments, <em>Strict Priority</em> starves routine patients during peak morning surges (routine waits &gt; 45-60 min). 
            Aging Priority dynamically promotes long-waiting patients while ensuring acute emergencies maintain immediate fast-tracking.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ Dynamic Alpha Sweep: Emergency Response vs Routine Starvation")
        df_alpha_busy = get_cached_alpha_sweep_busy()

        col_sw1, col_sw2 = st.columns([7, 5])
        with col_sw1:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax2 = ax1.twinx()
            ax1.plot(df_alpha_busy["alpha"], df_alpha_busy["emergency_wait"], marker="o", color="#ef4444", lw=2.2, label="Emergency Wait (min)")
            ax1.plot(df_alpha_busy["alpha"], df_alpha_busy["routine_wait"], marker="s", color="#22c55e", lw=2.2, label="Routine Wait (min)")
            ax2.plot(df_alpha_busy["alpha"], df_alpha_busy["starvation_rate_45m"], marker="^", color="#f59e0b", lw=2.0, linestyle="--", label="Routine Starvation (% > 45m)")
            ax1.set_xlabel("Aging Sensitivity Parameter (alpha)", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Wait Time (Minutes)", fontsize=10)
            ax2.set_ylabel("Starvation Rate (% > 45m)", fontsize=10, color="#d97706")
            ax1.grid(True, linestyle="--", alpha=0.5)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_sw2:
            st.markdown("##### ⏱️ Overtaking Times & Congestion Frontier")
            st.dataframe(
                df_alpha_busy[[
                    "alpha", "overtaking_time_min", "emergency_wait_ci",
                    "routine_wait_ci", "starvation_rate_25m_ci", "starvation_rate_45m_ci"
                ]],
                use_container_width=True
            )
            st.caption("Overtaking Time = $(Acuity_{urgent} - Acuity_{routine}) / \\alpha = 40 / \\alpha$. Under busy congestion, routine waits reach 25-45+ min, making $\\alpha \\ge 1.0$ readily trigger overtaking.")

        st.markdown("#### 🔄 Before vs After: Strict Priority vs Aging Priority (30-Seed CRN Benchmark)")
        c1 = all_comparisons["aging_priority"]
        m_c1 = c1["metrics"]

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-before">
                    <h4 style="color:#ef4444;margin-top:0;">❌ BEFORE: Strict Priority</h4>
                    <p>• <strong>Routine Mean Wait:</strong> {m_c1['routine_wait']['before_str']}<br>
                    • <strong>Routine Starvation (&gt;25m):</strong> {m_c1['starvation_25m']['before_str']}<br>
                    • <strong>Severe Starvation (&gt;45m):</strong> {m_c1['starvation_45m']['before_str']}<br>
                    • <strong>Emergency Wait:</strong> {m_c1['emergency_wait']['before_str']}<br>
                    • <strong>Overall Mean Wait:</strong> {m_c1['overall_wait']['before_str']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_a:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-after">
                    <h4 style="color:#22c55e;margin-top:0;">✅ AFTER: Aging Priority (&alpha; = 2.0)</h4>
                    <p>• <strong>Routine Mean Wait:</strong> {m_c1['routine_wait']['after_str']} <span class="metric-badge badge-green">{m_c1['routine_wait']['pct_delta']:+.1f}% (p = {m_c1['routine_wait']['p_val_str']})</span><br>
                    • <strong>Routine Starvation (&gt;25m):</strong> {m_c1['starvation_25m']['after_str']} <span class="metric-badge badge-blue">p = {m_c1['starvation_25m']['p_val_str']}</span><br>
                    • <strong>Severe Starvation (&gt;45m):</strong> {m_c1['starvation_45m']['after_str']} <span class="metric-badge badge-green">p = {m_c1['starvation_45m']['p_val_str']}</span><br>
                    • <strong>Emergency Wait:</strong> {m_c1['emergency_wait']['after_str']} <span class="metric-badge badge-yellow">p = {m_c1['emergency_wait']['p_val_str']}</span><br>
                    • <strong>Overall Mean Wait:</strong> {m_c1['overall_wait']['after_str']} <span class="metric-badge badge-green">p = {m_c1['overall_wait']['p_val_str']}</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"📊 **Statistically Audited Clinical Analysis (30-Seed CRN Paired Tests):** {c1['narrative']}")

    # -----------------------------------------------------------------------
    # INNOVATION 2: Predict-then-Decide
    # -----------------------------------------------------------------------
    with innov_tabs[1]:
        st.markdown("""
        <div class="innovation-box">
            <h3 style="color:#38bdf8;margin-top:0;">Innovation 2: Predict-then-Decide Queueing & Bed Pre-Reservation</h3>
            <p>Uses trained <strong>Multiple Regression</strong> to predict consultation time and <strong>Logistic Regression</strong> to predict admission probability right at triage.
            Queue dispatching is augmented by predicted service times, and inpatient beds are proactively pre-reserved for high-risk admissions before doctor consultation finishes.</p>
        </div>
        """, unsafe_allow_html=True)

        c2 = all_comparisons["predict_then_decide"]
        m_c2 = c2["metrics"]

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-before">
                    <h4 style="color:#ef4444;margin-top:0;">❌ BEFORE: Reactive Bed Allocation</h4>
                    <p>• <strong>Inpatient Boarding Delay:</strong> {m_c2['bed_boarding_wait']['before_str']}<br>
                    • <strong>Overall OPD Wait:</strong> {m_c2['overall_wait']['before_str']}<br>
                    • <strong>Bed Rejection Rate:</strong> {m_c2['bed_rejection_rate']['before_str']}<br>
                    • <strong>Clinical Bottleneck:</strong> Bed turnover requested only post-consultation, generating administrative boarding delays.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-after">
                    <h4 style="color:#22c55e;margin-top:0;">✅ AFTER: Predict-then-Decide</h4>
                    <p>• <strong>Inpatient Boarding Delay:</strong> {m_c2['bed_boarding_wait']['after_str']} <span class="metric-badge badge-green">{m_c2['bed_boarding_wait']['pct_delta']:+.1f}% (p = {m_c2['bed_boarding_wait']['p_val_str']})</span><br>
                    • <strong>Overall OPD Wait:</strong> {m_c2['overall_wait']['after_str']} <span class="metric-badge badge-green">{m_c2['overall_wait']['pct_delta']:+.1f}% (p = {m_c2['overall_wait']['p_val_str']})</span><br>
                    • <strong>Bed Rejection Rate:</strong> {m_c2['bed_rejection_rate']['after_str']} <span class="metric-badge badge-blue">p = {m_c2['bed_rejection_rate']['p_val_str']}</span><br>
                    • <strong>Operational Gain:</strong> Triage pre-reservation eliminates bed wait; predicted SJF bonus enhances consultation flow.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"📊 **Statistically Audited Clinical Analysis (30-Seed CRN Paired Tests):** {c2['narrative']}")

    # -----------------------------------------------------------------------
    # INNOVATION 3: Demand Regime Detection
    # -----------------------------------------------------------------------
    with innov_tabs[2]:
        st.markdown("""
        <div class="innovation-box">
            <h3 style="color:#38bdf8;margin-top:0;">Innovation 3: Demand Regime Detection & Dynamic Staffing</h3>
            <p>Clusters hourly operational demand using <strong>K-Means (Elbow SSE + Silhouette Score)</strong> and <strong>Agglomerative Hierarchical Clustering (Ward Dendrogram)</strong> into three operational regimes: <strong>Normal</strong>, <strong>Peak</strong>, and <strong>Surge</strong>.
            The hospital dynamically triggers surge doctor capacity and shifts &alpha; rather than relying on static inflexible staffing.</p>
        </div>
        """, unsafe_allow_html=True)

        clustering_analyzer = get_cached_clustering()
        r_metrics, r_summary = clustering_analyzer.cluster_demand_regimes(n_clusters=3)

        st.markdown("#### 📊 Agglomerative Dendrogram & Regime Characterization")
        c_dend, c_prof = st.columns([6, 6])
        with c_dend:
            dendro_path = resolve_asset_path("data/clustering_plots/regime_dendrogram.png")
            if os.path.exists(dendro_path):
                st.image(dendro_path, caption="Hierarchical Ward Linkage Dendrogram")
            else:
                st.info("Generating dendrogram...")
                clustering_analyzer.generate_clustering_plots()
                st.image(resolve_asset_path("data/clustering_plots/regime_dendrogram.png"), caption="Hierarchical Ward Linkage Dendrogram")

        with c_prof:
            st.markdown("##### Operational Regime Profiles (Normal vs Peak vs Surge)")
            st.dataframe(r_summary, use_container_width=True)

        c3 = all_comparisons["regime_aware"]
        m_c3 = c3["metrics"]

        st.markdown("#### 🔄 Before vs After: Fixed Static Staffing vs Regime-Aware Dynamic Staffing (30-Seed CRN Benchmark)")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-before">
                    <h4 style="color:#ef4444;margin-top:0;">❌ BEFORE: Fixed Static Staffing</h4>
                    <p>• <strong>Overall Mean Wait:</strong> {m_c3['overall_wait']['before_str']}<br>
                    • <strong>Routine Mean Wait:</strong> {m_c3['routine_wait']['before_str']}<br>
                    • <strong>Doctor Utilization:</strong> {m_c3['doctor_utilization']['before_str']}<br>
                    • <strong>Limitation:</strong> Inflexible doctor rosters unable to cope with 9am-12pm surge queues.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-after">
                    <h4 style="color:#22c55e;margin-top:0;">✅ AFTER: Regime-Aware Dynamic Policy</h4>
                    <p>• <strong>Overall Mean Wait:</strong> {m_c3['overall_wait']['after_str']} <span class="metric-badge badge-green">{m_c3['overall_wait']['pct_delta']:+.1f}% (p = {m_c3['overall_wait']['p_val_str']})</span><br>
                    • <strong>Routine Mean Wait:</strong> {m_c3['routine_wait']['after_str']} <span class="metric-badge badge-green">{m_c3['routine_wait']['pct_delta']:+.1f}% (p = {m_c3['routine_wait']['p_val_str']})</span><br>
                    • <strong>Doctor Utilization:</strong> {m_c3['doctor_utilization']['after_str']} <span class="metric-badge badge-blue">p = {m_c3['doctor_utilization']['p_val_str']}</span><br>
                    • <strong>Operational Gain:</strong> Dynamic on-call buffer prevents queue buildup during peak hours.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"📊 **Statistically Audited Clinical Analysis (30-Seed CRN Paired Tests):** {c3['narrative']}")

    # -----------------------------------------------------------------------
    # INNOVATION 4: Explainable RL Policy Distillation
    # -----------------------------------------------------------------------
    with innov_tabs[3]:
        st.markdown("""
        <div class="innovation-box">
            <h3 style="color:#38bdf8;margin-top:0;">Innovation 4: Explainable RL Policy Distillation</h3>
            <p>Reinforcement Learning (PPO) excels at dynamic sequential dispatching, but deep neural policies are <em>uninterpretable black boxes</em> that clinical ethics boards reject.
            We distill the trained PPO agent into an interpretable <strong>Surrogate Decision Tree</strong>, proving high fidelity (&gt; 55-90%) and extracting transparent clinical rules.</p>
        </div>
        """, unsafe_allow_html=True)

        c_tree, c_feat = st.columns([7, 5])
        with c_tree:
            tree_plot_path = resolve_asset_path("data/rl_plots/rl_distilled_tree.png")
            if os.path.exists(tree_plot_path):
                st.image(tree_plot_path, caption="Distilled Decision Tree Policy (Surrogate Model)")
            else:
                st.info("Generating distilled decision tree...")
                distiller = PolicyDistiller()
                distiller.generate_explanation_plots()
                st.image(resolve_asset_path("data/rl_plots/rl_distilled_tree.png"), caption="Distilled Decision Tree Policy")

        with c_feat:
            feat_imp_path = resolve_asset_path("data/rl_plots/rl_feature_importances.png")
            if os.path.exists(feat_imp_path):
                st.image(feat_imp_path, caption="State Feature Importances in Distilled RL Policy")

        c4 = all_comparisons["explainable_rl"]
        m_c4 = c4["metrics"]

        st.markdown("#### 🔄 Before vs After: Neural Black-Box vs Transparent Decision Tree (30-Seed CRN Benchmark)")
        col_rl1, col_rl2 = st.columns(2)
        with col_rl1:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-before">
                    <h4 style="color:#ef4444;margin-top:0;">❌ BEFORE: Neural PPO Policy (Black Box)</h4>
                    <p>• <strong>Cumulative Episode Reward:</strong> {m_c4['cumulative_reward']['before_str']}<br>
                    • <strong>Routine Queue Wait:</strong> {m_c4['routine_wait']['before_str']}<br>
                    • <strong>Model Format:</strong> 2-Layer MLP (10,000+ continuous neural weights)<br>
                    • <strong>Auditability:</strong> Zero clinical transparency; non-compliant with medical AI safety regulations.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_rl2:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-after">
                    <h4 style="color:#22c55e;margin-top:0;">✅ AFTER: Distilled Decision Tree</h4>
                    <p>• <strong>Cumulative Episode Reward:</strong> {m_c4['cumulative_reward']['after_str']} <span class="metric-badge badge-blue">p = {m_c4['cumulative_reward']['p_val_str']}</span><br>
                    • <strong>Routine Queue Wait:</strong> {m_c4['routine_wait']['after_str']} <span class="metric-badge badge-green">p = {m_c4['routine_wait']['p_val_str']}</span><br>
                    • <strong>Action Fidelity:</strong> <span class="metric-badge badge-green">{c4.get('fidelity_pct', 57.9)}% Agreement</span> with neural policy<br>
                    • <strong>Auditability:</strong> 100% human-interpretable splits based on queue depth and wait thresholds.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"📊 **Statistically Audited Clinical Analysis (30-Seed CRN Paired Tests):** {c4['narrative']}")


# ===========================================================================
# TAB 3: Policy Benchmark & 30-Seed Statistical Tests
# ===========================================================================

with tabs[2]:
    st.markdown("### 📊 30-Seed Policy Benchmark & Hypothesis Testing")
    st.markdown("Rigorous statistical evaluation running each policy across 30 independent random seeds. Reports 95% Confidence Intervals and paired Student's t-tests.")

    evaluator = HospitalEvaluator(num_seeds=30)
    with st.spinner("Running 30-seed simulation benchmark..."):
        benchmark_table = evaluator.run_multi_seed_benchmark(sim_days=1)
        ttest_table = evaluator.run_paired_t_tests(baseline_policy="fcfs")

    st.markdown("#### 📋 30-Seed Performance Summary Table (with 95% CIs)")
    st.dataframe(
        benchmark_table[["Policy", "Mean_Wait_CI", "Emergency_Wait_CI", "Routine_Wait_CI", "Doctor_Utilization", "Starvation_Rate"]],
        use_container_width=True
    )

    st.markdown("#### 🧪 Paired Student's t-Tests (H0: Equal Mean Wait Time vs FCFS Baseline)")
    st.dataframe(ttest_table, use_container_width=True)

    eval_plot_path = resolve_asset_path("data/evaluation_plots/policy_benchmark_waits.png")
    if os.path.exists(eval_plot_path):
        st.image(eval_plot_path, caption="30-Seed Policy Comparison (Wait Times by Triage)")


# ===========================================================================
# TAB 4: Clinical Recommender System
# ===========================================================================

with tabs[3]:
    st.markdown("### 🤖 Clinical Doctor, Department & Ward Recommender")
    st.markdown("Interactive recommender comparing **Content-Based (TF-IDF)**, **Collaborative Filtering (Matrix Factorization)**, and **Hybrid** methods, evaluated via Precision@k.")

    rec_suite = get_cached_recommender()

    col_rc1, col_rc2 = st.columns([5, 7])
    with col_rc1:
        st.markdown("#### 👤 Patient Triage Profile Input")
        p_symptom = st.selectbox(
            "Reported Chief Symptom",
            options=[
                "Severe Chest Pain",
                "Acute Respiratory Distress",
                "High Fever with Delirium",
                "Compound Fracture",
                "Palpitations & Arrhythmia",
                "Severe Abdominal Pain",
                "Severe Cough & Wheezing",
                "Joint Pain & Arthritis",
                "Routine Health Checkup",
                "Skin Rash & Allergy",
                "Pediatric Fever",
            ]
        )
        p_age = st.slider("Patient Age", min_value=5, max_value=88, value=58)
        p_triage = st.selectbox("Assessed Triage Acuity", options=["Emergency", "Urgent", "Routine"], index=0)
        p_spo2 = st.slider("Oxygen Saturation SpO2 (%)", min_value=70.0, max_value=100.0, value=89.5, step=0.5)
        p_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=70.0, max_value=220.0, value=162.0, step=1.0)

        age_grp = "<18" if p_age < 18 else ("18-40" if p_age <= 40 else ("41-65" if p_age <= 65 else ">65"))

    with col_rc2:
        st.markdown("#### 🩺 Recommended Doctor & Department")
        hyb_recs = rec_suite.recommend_hybrid(p_symptom, p_age, age_grp, p_triage, alpha_content=0.65, top_k=3)

        for i, doc in enumerate(hyb_recs):
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6);border-left:4px solid #38bdf8;padding:10px 16px;border-radius:8px;margin-bottom:8px;">
                <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;">#{i+1} {doc['doctor_name']} ({doc['department']})</div>
                <div style="font-size:0.85rem;color:#94a3b8;">Hybrid Match Score: <strong>{doc['score']}</strong> (Content: {doc['content_score']}, Collaborative: {doc['collaborative_score']})</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🛏️ Inpatient Bed & Ward Recommendation")
        p_admit_prob = 0.85 if p_triage == "Emergency" else (0.45 if p_triage == "Urgent" else 0.08)
        bed_rec = rec_suite.recommend_ward_and_bed(p_triage, p_spo2, p_bp, p_admit_prob, available_gen_beds=num_general_beds, available_icu_beds=num_icu_beds)

        bed_badge = "badge-red" if bed_rec["recommended_ward"] == "ICU" else ("badge-yellow" if "General" in bed_rec["recommended_ward"] else "badge-green")
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.6);border-left:4px solid #22c55e;padding:12px 16px;border-radius:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;">Assigned Ward: <span class="metric-badge {bed_badge}">{bed_rec['recommended_ward']}</span></div>
                <span class="metric-badge badge-blue">Admission Prob: {p_admit_prob*100:.0f}%</span>
            </div>
            <p style="margin:6px 0 0 0;font-size:0.88rem;color:#cbd5e1;">{bed_rec['clinical_reason']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🎯 Recommender Model Benchmarking: Precision@k")
    eval_recs = rec_suite.evaluate_precision_at_k(test_sample_size=300)
    st.dataframe(eval_recs, use_container_width=True)

    # Recommender Fallback Demonstration
    st.markdown("---")
    st.markdown("#### 🔄 Recommender Fallback Engine (Doctor & Department Unavailability)")
    st.markdown("Simulate what happens when a patient's top recommended physician is on leave, on break, or absent. The engine dynamically provides an alternative specialist from the same or an allied clinical department.")

    doc_options = list(rec_suite.doctor_catalog["name"])
    sim_unavail_doc = st.selectbox("Simulate Physician Unavailable (Break / Leave / Absent)", options=["None"] + doc_options, index=1)

    if sim_unavail_doc != "None":
        fallback_res = rec_suite.recommend_with_fallback(
            symptoms=p_symptom,
            age=p_age,
            age_group=age_grp,
            triage_class=p_triage,
            unavailable_doctors=[sim_unavail_doc],
            top_k=3
        )

        if fallback_res["fallback_triggered"]:
            st.warning(f"⚠️ **FALLBACK TRIGGERED:** {fallback_res['fallback_reason']}")
        else:
            st.success(f"✅ {fallback_res['fallback_reason']}")

        st.markdown("##### 🩺 Active Alternative Recommendations:")
        f_cols = st.columns(min(3, len(fallback_res["active_recommendations"])))
        for i, doc in enumerate(fallback_res["active_recommendations"]):
            with f_cols[i]:
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7);padding:12px;border-radius:8px;border-left:4px solid #38bdf8;">
                    <div style="font-weight:700;color:#f8fafc;">#{i+1} {doc['doctor_name']}</div>
                    <div style="font-size:0.85rem;color:#94a3b8;">{doc['department']}</div>
                    <div style="font-size:0.8rem;color:#cbd5e1;margin-top:4px;">Score: <strong>{doc['score']}</strong></div>
                </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# TAB 5: Fairness & Demographics
# ===========================================================================

with tabs[4]:
    st.markdown("### ⚖️ Demographic Parity & Fairness Auditing")
    st.markdown("Subgroup analysis evaluating wait times and admission equity across **Age Groups**, **Gender**, and **Residential Area (Urban vs Rural)**.")

    opd_dataset = get_cached_dataset()
    fairness_reports = audit_demographic_fairness(opd_dataset)

    col_fa1, col_fa2, col_fa3 = st.columns(3)
    with col_fa1:
        st.markdown("##### 👥 Age Demographic Subgroups")
        st.dataframe(fairness_reports["age_fairness"], use_container_width=True)

    with col_fa2:
        st.markdown("##### ⚧️ Gender Subgroups")
        st.dataframe(fairness_reports["gender_fairness"], use_container_width=True)

    with col_fa3:
        st.markdown("##### 🏙️ Residential Area (Urban vs Rural)")
        st.dataframe(fairness_reports["area_fairness"], use_container_width=True)

    fairness_plot_path = resolve_asset_path("data/fairness_plots/alpha_fairness_tradeoff.png")
    if os.path.exists(fairness_plot_path):
        st.image(fairness_plot_path, caption="Fairness-vs-Acuity Trade-off Curve")


# ===========================================================================
# TAB 6: M/M/c & Little's Law Validation
# ===========================================================================

with tabs[5]:
    st.markdown("### 🔬 Queuing Theory (M/M/c) & Little's Law Simulator Validation")
    st.markdown("Demonstrates mathematical validity of the discrete-event engine by comparing empirical simulation metrics against theoretical Erlang-C formulas and verifying Little's Law ($L = \\lambda W$).")

    with st.spinner("Validating against Erlang-C formulas..."):
        validation_df = validate_queuing_theory()

    st.markdown("#### 📋 Erlang-C Analytical Theory vs SimPy Simulation Point Benchmark")
    st.dataframe(validation_df, use_container_width=True)

    c_v1, c_v2 = st.columns(2)
    with c_v1:
        mmc_plot_path = resolve_asset_path("data/validation_plots/mmc_validation.png")
        if os.path.exists(mmc_plot_path):
            st.image(mmc_plot_path, caption="M/M/c Queue Wait Curve: Theoretical vs Simulated Points")
    with c_v2:
        ll_plot_path = resolve_asset_path("data/validation_plots/littles_law_validation.png")
        if os.path.exists(ll_plot_path):
            st.image(ll_plot_path, caption="Little's Law Identity Check: Empirical L vs lambda * W")


# ===========================================================================
# TAB 7: What-If Stress Testing
# ===========================================================================

with tabs[6]:
    st.markdown("### 🧪 Operational What-If Scenarios & Sensitivity Analysis")
    st.markdown("Simulating extreme hospital stresses: Seasonal Flu Surge (+30%), Doctor Shortages, Bed Expansions, and Parameter Sensitivity.")

    with st.spinner("Simulating What-If Scenarios..."):
        what_if_df = evaluator.run_what_if_scenarios(seeds_count=5)
        df_docs_sens, df_beds_sens = evaluator.run_sensitivity_analysis(seeds_count=5)

    st.markdown("#### 🚨 Stress Testing Scenarios Table")
    st.dataframe(what_if_df, use_container_width=True)

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("##### 👨‍⚕️ Doctor Staffing Sensitivity (4 to 8 Doctors)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df_docs_sens["Doctor_Count"], df_docs_sens["Mean_Wait_Min"], marker="o", color="#0284c7", lw=2)
        ax.set_title("Mean Wait vs Active Doctor Capacity", fontsize=10, fontweight="bold")
        ax.set_xlabel("Doctor Count", fontsize=9)
        ax.set_ylabel("Mean Wait (Minutes)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_s2:
        st.markdown("##### 🛏️ Bed Capacity Sensitivity (20 to 60 Beds)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(df_beds_sens["General_Beds"], df_beds_sens["Bed_Rejection_Rate_Pct"], color="#ef4444", width=6, edgecolor="black", alpha=0.85)
        ax.set_title("Bed Rejection Rate vs General Bed Capacity", fontsize=10, fontweight="bold")
        ax.set_xlabel("General Ward Beds", fontsize=9)
        ax.set_ylabel("Rejection Rate (%)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Doctor Availability Scenarios Benchmark (with 95% CIs)
    st.markdown("---")
    st.markdown("#### 🩺 Doctor Availability & Backup Staffing Scenarios (with 95% CIs)")
    st.markdown("Evaluates the operational impact across 4 availability and backup scenarios: (1) All Present, (2) Absences Without Backup, (3) Absences with Floating Pool, and (4) Absences with Pool + Task Shifting.")

    with st.spinner("Simulating Staff Availability Scenarios across Seeds..."):
        avail_scenarios_df = evaluator.run_doctor_availability_scenarios(seeds_count=5)
    st.dataframe(avail_scenarios_df, use_container_width=True)

    # 30-Seed CRN Before vs After Container for Doctor Absences vs Backup Staffing
    st.markdown("#### 🔄 Before vs After: Doctor Absences Without Backup vs Floating Pool + Task Shifting (30-Seed CRN Benchmark)")
    c_doc = all_comparisons.get("doctor_availability", {})
    if c_doc and "metrics" in c_doc:
        m_cd = c_doc["metrics"]
        col_dab1, col_dab2 = st.columns(2)
        with col_dab1:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-before">
                    <h4 style="color:#ef4444;margin-top:0;">❌ BEFORE: Absences Without Backup</h4>
                    <p>• <strong>Overall Mean Wait:</strong> {m_cd['overall_wait']['before_str']}<br>
                    • <strong>Routine Mean Wait:</strong> {m_cd['routine_wait']['before_str']}<br>
                    • <strong>Routine Starvation (&gt;25m):</strong> {m_cd['routine_starvation']['before_str']}<br>
                    • <strong>Senior Doctor Utilization:</strong> {m_cd['doctor_utilization']['before_str']}<br>
                    • <strong>Failure Mode:</strong> Uncovered absences create severe bottlenecks and acute wait escalation.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_dab2:
            st.markdown(f"""
            <div class="compare-container">
                <div class="compare-col compare-after">
                    <h4 style="color:#22c55e;margin-top:0;">✅ AFTER: Floating Pool + Task Shifting</h4>
                    <p>• <strong>Overall Mean Wait:</strong> {m_cd['overall_wait']['after_str']} <span class="metric-badge badge-green">{m_cd['overall_wait']['pct_delta']:+.1f}% (p = {m_cd['overall_wait']['p_val_str']})</span><br>
                    • <strong>Routine Mean Wait:</strong> {m_cd['routine_wait']['after_str']} <span class="metric-badge badge-green">{m_cd['routine_wait']['pct_delta']:+.1f}% (p = {m_cd['routine_wait']['p_val_str']})</span><br>
                    • <strong>Routine Starvation (&gt;25m):</strong> {m_cd['routine_starvation']['after_str']} <span class="metric-badge badge-green">p = {m_cd['routine_starvation']['p_val_str']}</span><br>
                    • <strong>Senior Doctor Utilization:</strong> {m_cd['doctor_utilization']['after_str']} <span class="metric-badge badge-blue">p = {m_cd['doctor_utilization']['p_val_str']}</span><br>
                    • <strong>Resilience Gain:</strong> Floating doctors absorb absences while junior nurses offload routine consultations.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"📊 **Statistically Audited Clinical Analysis (30-Seed CRN Paired Tests):** {c_doc['narrative']}")
