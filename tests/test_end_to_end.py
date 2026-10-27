"""
End-to-End Test Suite for Hospital OPD Optimization System.
Tests all discrete-event simulation, machine learning, clustering,
recommender, reinforcement learning, and fairness modules.
"""

import math
import os
import sys

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import OPDConfig, HospitalOPDSimulation
from src.preprocessing_eda import load_raw_data, clean_missing_values, compute_descriptive_statistics, compute_correlation_matrices
from src.ml_models import MLSuite
from src.clustering import OPDClusteringAnalysis
from src.recommender import HospitalRecommenderSuite
from src.policies import run_policy_simulation
from src.rl_env import HospitalOPDEnv
from src.explain_policy import PolicyDistiller
from src.fairness import audit_demographic_fairness, compute_starvation_metrics, sweep_alpha_tradeoff
from src.validation import erlang_c_theoretical, simulate_stationary_mmc


def test_simulation_runs():
    """Verify SimPy OPD discrete-event simulation runs and outputs valid records and KPIs."""
    cfg = OPDConfig(sim_days=1, num_doctors=6, seed=42)
    sim = HospitalOPDSimulation(cfg)
    df = sim.run()
    kpis = sim.get_kpis()

    assert not df.empty, "Simulation output DataFrame should not be empty."
    assert len(df) > 100, f"Expected >100 patients for 1 day, got {len(df)}"
    assert 50.0 <= kpis["doctor_utilization"] <= 95.0, f"Doctor utilization {kpis['doctor_utilization']}% outside realistic range."
    assert kpis["mean_wait"] >= 0.0, "Mean wait cannot be negative."
    assert kpis["emergency_wait"] <= kpis["routine_wait"] + 5.0, "Emergency wait should generally be lower than routine wait under aging/priority."


def test_preprocessing_and_eda():
    """Verify data loading, imputation, descriptive statistics, and correlation calculations."""
    df_raw = load_raw_data("data/synthetic_opd.csv")
    assert not df_raw.empty
    assert df_raw.isnull().sum().sum() > 0, "Raw dataset should contain injected missing values."

    df_clean, report = clean_missing_values(df_raw)
    assert len(report) > 0
    assert df_clean["temperature"].isnull().sum() == 0, "Missing temperature should be imputed."
    assert df_clean["systolic_bp"].isnull().sum() == 0, "Missing systolic_bp should be imputed."
    assert df_clean["satisfaction_score"].isnull().sum() == 0, "Missing satisfaction_score should be imputed."

    stats_df = compute_descriptive_statistics(df_clean)
    assert not stats_df.empty
    assert "Mean" in stats_df.columns
    assert "Std_Dev" in stats_df.columns

    p_corr, s_corr = compute_correlation_matrices(df_clean)
    assert p_corr.shape == s_corr.shape
    assert p_corr.loc["service_time", "service_time"] == 1.0


def test_ml_models_suite():
    """Verify classical ML models: regression, logistic, decision tree, KNN, and evaluation table."""
    suite = MLSuite(seed=42)
    comparison_table = suite.run_all()

    assert not comparison_table.empty
    assert len(comparison_table) >= 6
    assert "Model" in comparison_table.columns
    assert "Primary Metric" in comparison_table.columns

    # Verify decision tree exists and has good accuracy (>80%)
    dt_res = suite.train_triage_decision_tree()
    assert dt_res["Accuracy"] > 0.80

    # Verify logistic regression has strong ROC-AUC (>0.75)
    log_res = suite.train_admission_classifier()
    assert log_res["ROC-AUC"] > 0.75


def test_clustering_modules():
    """Verify K-means patient clustering and agglomerative demand regime clustering."""
    analyzer = OPDClusteringAnalysis(seed=42)
    p_metrics, p_k, p_profiles = analyzer.analyze_patient_clusters()
    assert len(p_metrics) > 0
    assert p_k in [2, 3, 4, 5, 6]
    assert len(p_profiles) == p_k

    r_metrics, r_summary = analyzer.cluster_demand_regimes(n_clusters=3)
    assert len(r_metrics) > 0
    assert len(r_summary) == 3
    assert "Normal" in r_summary.index
    assert "Peak" in r_summary.index
    assert "Surge" in r_summary.index


def test_recommenders():
    """Verify Content-based, Collaborative, Hybrid, and Bed allocation recommenders."""
    suite = HospitalRecommenderSuite(seed=42)

    cb_recs = suite.recommend_content_based("Severe Chest Pain", age=55, triage_class="Emergency", top_k=3)
    assert len(cb_recs) == 3
    assert any(r["department"] in ["Cardiology", "Emergency Medicine"] for r in cb_recs)

    cf_recs = suite.recommend_collaborative("41-65", "Severe Chest Pain", top_k=3)
    assert len(cf_recs) == 3

    hyb_recs = suite.recommend_hybrid("Severe Chest Pain", age=55, age_group="41-65", triage_class="Emergency", top_k=3)
    assert len(hyb_recs) == 3

    bed_rec = suite.recommend_ward_and_bed("Emergency", oxygen_saturation=88.0, systolic_bp=165.0, predicted_admit_prob=0.85)
    assert bed_rec["recommended_ward"] == "ICU"

    eval_table = suite.evaluate_precision_at_k(test_sample_size=100)
    assert not eval_table.empty
    assert eval_table.loc[eval_table["Evaluation Metric"] == "Precision@3", "Hybrid Recommender"].values[0] >= 0.80


def test_policies():
    """Verify execution of all queue dispatching and capacity management policies."""
    policies = ["fcfs", "strict_priority", "sjf", "aging_priority", "predict_then_decide", "regime_aware"]
    for pol in policies:
        _, kpi = run_policy_simulation(pol, OPDConfig(sim_days=1, num_doctors=6), alpha=0.5, seed=42)
        assert kpi["total_patients"] > 0
        assert kpi["mean_wait"] >= 0.0


def test_rl_env_and_distillation():
    """Verify Gymnasium environment dynamics and explainable RL policy distillation."""
    env = HospitalOPDEnv(seed=42)
    obs, _ = env.reset()
    assert obs.shape == (10,)

    obs, reward, term, trunc, info = env.step(0)
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

    distiller = PolicyDistiller(seed=42)
    if os.path.exists(distiller.model_path):
        res = distiller.distill(num_episodes=5)
        assert res["fidelity"] > 0.50
        assert len(res["feature_importances"]) == 10


def test_fairness_and_alpha_sweep():
    """Verify demographic auditing and alpha parameter sweeping."""
    df_raw = load_raw_data("data/synthetic_opd.csv")
    df_clean, _ = clean_missing_values(df_raw)

    fairness_reports = audit_demographic_fairness(df_clean)
    assert "age_fairness" in fairness_reports
    assert "gender_fairness" in fairness_reports
    assert "area_fairness" in fairness_reports

    starve = compute_starvation_metrics(df_clean, threshold_minutes=45.0)
    assert "routine_starvation_rate" in starve

    sweep_df = sweep_alpha_tradeoff(alphas=[0.0, 0.5, 1.0], sim_days=1, seed=42)
    assert len(sweep_df) == 3
    assert "emergency_wait" in sweep_df.columns
    assert "routine_wait" in sweep_df.columns


def test_validation_theory():
    """Verify Erlang-C analytical formulas and Little's Law error (< 5%)."""
    # Test Erlang-C calculation
    theory = erlang_c_theoretical(lam=0.30, mu=0.08, c=5)
    assert 0.0 < theory["rho"] < 1.0
    assert theory["Wq"] > 0.0
    assert math.isclose(theory["L"], 0.30 * theory["W"], rel_tol=1e-3)

    # Test Little's Law in simulation
    sim_res = simulate_stationary_mmc(lam_per_hour=18.0, mean_service_minutes=12.0, num_doctors=5, sim_hours=40.0, seed=42)
    assert sim_res["littles_rel_error_pct"] < 5.0, f"Little's Law relative error {sim_res['littles_rel_error_pct']}% exceeded 5%."


def test_doctor_availability_and_staffing():
    """Verify doctor availability, shift schedules, staggered breaks (<25% concurrency), absences, and floating pool."""
    cfg = OPDConfig(
        sim_days=1,
        num_doctors=6,
        enable_breaks=True,
        max_break_concurrency_ratio=0.25,
        enable_absences=True,
        enable_leave=True,
        manual_absent_doctors=[2],
        num_floating_doctors=2,
        enable_floating_pool=True,
        num_junior_doctors_nurses=2,
        enable_task_shifting=True,
        teleconsultation_capacity=2,
        enable_emergency_override=True,
        seed=42
    )
    sim = HospitalOPDSimulation(cfg)
    df = sim.run()
    kpis = sim.get_kpis()

    assert not df.empty
    assert kpis["total_patients"] > 0

    # Verify doctor timeline logging
    timeline_df = sim.get_staff_timeline_dataframe()
    assert not timeline_df.empty
    assert "Staff_Name" in timeline_df.columns
    assert "Status" in timeline_df.columns

    # Verify manual leave was respected
    doc_statuses = sim.get_current_doctor_statuses()
    doc2_status = next(d["status"] for d in doc_statuses if d["doctor_id"] == 2)
    assert doc2_status == "Leave"

    # Verify break concurrency cap (never > 25% on break simultaneously)
    break_intervals = timeline_df[timeline_df["Status"] == "Break"]
    max_allowed_on_break = max(1, int(cfg.num_doctors * cfg.max_break_concurrency_ratio))
    for t_min in range(480, 1200, 5):
        h = t_min / 60.0
        active_breaks = len(break_intervals[(break_intervals["Start_Hour"] <= h) & (h < break_intervals["End_Hour"])])
        assert active_breaks <= max_allowed_on_break, f"Break concurrency {active_breaks} exceeded 25% cap ({max_allowed_on_break})"

    # Verify task shifting (some routine patients treated by junior staff or teleconsultation)
    server_types = df["server_type"].unique()
    assert any("Junior" in s or "Senior" in s for s in server_types)


def test_recommender_fallback():
    """Verify recommender fallback when preferred doctor is unavailable."""
    suite = HospitalRecommenderSuite(seed=42)
    res = suite.recommend_with_fallback(
        symptoms="Severe Chest Pain",
        age=58,
        age_group="41-65",
        triage_class="Emergency",
        unavailable_doctors=["Dr. Sarah Chen"],
        top_k=3
    )

    assert res["fallback_triggered"] is True
    assert "Dr. Sarah Chen" not in [d["doctor_name"] for d in res["active_recommendations"]]
    assert len(res["active_recommendations"]) > 0


def test_doctor_availability_scenarios_evaluation():
    """Verify multi-scenario evaluation in evaluate.py comparing availability backups."""
    from src.evaluate import HospitalEvaluator
    evaluator = HospitalEvaluator(num_seeds=3)
    scenarios_df = evaluator.run_doctor_availability_scenarios(seeds_count=3)

    assert not scenarios_df.empty
    assert len(scenarios_df) == 4
    assert "All Present" in scenarios_df["Scenario"].values
    assert "Absences Without Backup" in scenarios_df["Scenario"].values
    assert "Absences with Floating Pool" in scenarios_df["Scenario"].values
    assert "Absences with Pool + Task Shifting" in scenarios_df["Scenario"].values
    assert os.path.exists("data/availability_results.csv")
    assert os.path.getsize("data/availability_results.csv") > 0


def test_before_after_policies_differ_under_busy_scenario():
    """
    Requirement 6: Fails if before and after metrics are identical under the calibrated busy scenario.
    Tests all 4 operational policy comparisons:
    1. Strict Priority vs Aging Priority
    2. Reactive Bed Allocation vs Predict-then-Decide
    3. Fixed Static Staffing vs Regime-Aware Dynamic Staffing
    4. Doctor Absences Without Backup vs Floating Pool + Task Shifting
    """
    seed = 42

    # 1. Innovation 1: Strict Priority vs Aging Priority
    cfg_strict = OPDConfig.create_busy_scenario(seed=seed)
    cfg_aging = OPDConfig.create_busy_scenario(seed=seed)
    _, kpi_strict = run_policy_simulation("strict_priority", cfg_strict, seed=seed)
    _, kpi_aging = run_policy_simulation("aging_priority", cfg_aging, alpha=2.0, seed=seed)

    # Routine wait must differ by at least 1.0 minute and be lower under aging
    assert abs(kpi_aging["routine_wait"] - kpi_strict["routine_wait"]) >= 1.0, (
        f"Strict and Aging routine waits are identical or too close: "
        f"Strict={kpi_strict['routine_wait']:.2f}m, Aging={kpi_aging['routine_wait']:.2f}m"
    )
    assert kpi_aging["routine_wait"] < kpi_strict["routine_wait"], "Aging priority should reduce routine wait."
    assert kpi_strict["routine_starvation_rate_25m"] > 10.0, "Busy scenario should show measurable routine starvation."

    # 2. Innovation 2: Reactive Bed Allocation vs Predict-then-Decide
    cfg_ptd = OPDConfig.create_busy_scenario(seed=seed)
    _, kpi_ptd = run_policy_simulation("predict_then_decide", cfg_ptd, alpha=2.0, seed=seed)

    assert abs(kpi_ptd["mean_bed_wait"] - kpi_strict["mean_bed_wait"]) >= 3.0, (
        f"Reactive and Predict-then-Decide bed wait times are identical or too close: "
        f"Reactive={kpi_strict['mean_bed_wait']:.2f}m, PTD={kpi_ptd['mean_bed_wait']:.2f}m"
    )
    assert kpi_ptd["mean_bed_wait"] < kpi_strict["mean_bed_wait"], (
        "Predict-then-Decide pre-reservation must reduce inpatient bed boarding delay."
    )

    # 3. Innovation 3: Fixed Static Staffing vs Regime-Aware Dynamic Staffing
    cfg_fixed = OPDConfig.create_busy_scenario(seed=seed, enable_floating_pool=False)
    cfg_regime = OPDConfig.create_busy_scenario(seed=seed, enable_floating_pool=True, num_floating_doctors=2)
    _, kpi_fixed = run_policy_simulation("strict_priority", cfg_fixed, seed=seed)
    _, kpi_regime = run_policy_simulation("regime_aware", cfg_regime, alpha=1.0, seed=seed)

    assert abs(kpi_regime["mean_wait"] - kpi_fixed["mean_wait"]) >= 2.0, (
        f"Fixed and Regime-Aware mean waits are identical or too close: "
        f"Fixed={kpi_fixed['mean_wait']:.2f}m, Regime={kpi_regime['mean_wait']:.2f}m"
    )
    assert kpi_regime["mean_wait"] < kpi_fixed["mean_wait"], (
        "Regime-Aware dynamic staffing must reduce overall mean wait during peak surges."
    )

    # 4. Availability & Backup: Absences Without Backup vs Floating Pool + Task Shifting
    cfg_unbacked = OPDConfig.create_busy_scenario(
        seed=seed,
        enable_absences=True,
        enable_leave=True,
        enable_floating_pool=False,
        enable_task_shifting=False,
    )
    cfg_backed = OPDConfig.create_busy_scenario(
        seed=seed,
        enable_absences=True,
        enable_leave=True,
        enable_floating_pool=True,
        num_floating_doctors=2,
        enable_task_shifting=True,
        num_junior_doctors_nurses=2,
    )
    _, kpi_unbacked = run_policy_simulation("aging_priority", cfg_unbacked, alpha=1.0, seed=seed)
    _, kpi_backed = run_policy_simulation("aging_priority", cfg_backed, alpha=1.0, seed=seed)

    assert abs(kpi_backed["mean_wait"] - kpi_unbacked["mean_wait"]) >= 10.0, (
        f"Unbacked and backed mean waits are identical or too close: "
        f"Unbacked={kpi_unbacked['mean_wait']:.2f}m, Backed={kpi_backed['mean_wait']:.2f}m"
    )
    assert kpi_backed["mean_wait"] < kpi_unbacked["mean_wait"], (
        "Backup staffing must significantly prevent system backlog during absences."
    )

