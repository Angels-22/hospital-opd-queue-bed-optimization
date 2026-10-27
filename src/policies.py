"""
Queue Dispatching and Capacity Management Policies for Hospital OPD.

Implements all required operational and innovative policies:
1. FCFS (First-Come, First-Served): Baseline benchmark.
2. Strict Priority: Acuity-based dispatch (Emergency > Urgent > Routine).
3. Shortest-Job-First (SJF): Dispatches shortest consultation times first.
4. INNOVATION 1: Fairness-Aware Aging Priority:
   Score(t) = acuity_weight + alpha * wait_time(t)
   Balances clinical acuity with starvation prevention.
5. INNOVATION 2: Predict-then-Decide Policy:
   Uses ML predictions (regression service time & logistic admission risk)
   to order queue and pre-reserve beds for high-risk patients.
6. INNOVATION 3: Regime-Aware Dynamic Policy:
   Switches staffing capacity and alpha dynamically based on demand clustering
   (Normal, Peak, Surge).
7. Proactive Staffing Policy:
   Uses next-hour arrival forecasts from regression to staff doctors in advance.
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, overload
import pandas as pd

from src.simulation import HospitalOPDSimulation, OPDConfig, Patient


# ---------------------------------------------------------------------------
# Policy Priority Scoring Functions
# ---------------------------------------------------------------------------

def score_fcfs(patient: Patient, current_time: float) -> float:
    """Earliest arrival time gets highest priority (negated for max-heap)."""
    return -patient.arrival_time


def score_strict_priority(patient: Patient, current_time: float) -> float:
    """Acuity weight dominates; tie-break with arrival time."""
    return patient.acuity_weight * 10000.0 - patient.arrival_time


def score_sjf(patient: Patient, current_time: float, use_predicted: bool = False) -> float:
    """Shortest consultation time first."""
    dur = patient.predicted_service_time if use_predicted else patient.consultation_actual_time
    return -dur * 1000.0 - patient.arrival_time


def score_aging_priority(patient: Patient, current_time: float, alpha: float = 0.5) -> float:
    """
    Innovation 1: Fairness-Aware Aging Priority.
    Score = Acuity_Weight + alpha * wait_time
    """
    current_wait = max(0.0, current_time - patient.arrival_time)
    return patient.acuity_weight + alpha * current_wait


def score_predict_then_decide(
    patient: Patient,
    current_time: float,
    alpha: float = 0.5
) -> float:
    """
    Innovation 2: Predict-then-Decide.
    Combines acuity aging with predicted service time bonus.
    """
    aging = score_aging_priority(patient, current_time, alpha)
    # Predicted shortest job boost
    predicted_dur = max(2.0, patient.predicted_service_time)
    sjf_boost = max(0.0, 30.0 - min(30.0, predicted_dur))
    return aging + sjf_boost


# ---------------------------------------------------------------------------
# Policy Runner & Orchestrator
# ---------------------------------------------------------------------------

@overload
def run_policy_simulation(
    policy_name: str,
    base_config: Optional[OPDConfig] = None,
    alpha: float = 0.5,
    seed: int = 42,
    service_predictor: Optional[Callable[[Patient], float]] = None,
    admit_predictor: Optional[Callable[[Patient], float]] = None,
    return_sim: Literal[False] = False,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, float]]: ...

@overload
def run_policy_simulation(
    policy_name: str,
    base_config: Optional[OPDConfig] = None,
    alpha: float = 0.5,
    seed: int = 42,
    service_predictor: Optional[Callable[[Patient], float]] = None,
    admit_predictor: Optional[Callable[[Patient], float]] = None,
    return_sim: Literal[True] = ...,
    **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, float], HospitalOPDSimulation]: ...

@overload
def run_policy_simulation(
    policy_name: str,
    base_config: Optional[OPDConfig] = None,
    alpha: float = 0.5,
    seed: int = 42,
    service_predictor: Optional[Callable[[Patient], float]] = None,
    admit_predictor: Optional[Callable[[Patient], float]] = None,
    return_sim: bool = False,
    **kwargs: Any
) -> Union[Tuple[pd.DataFrame, Dict[str, float]], Tuple[pd.DataFrame, Dict[str, float], HospitalOPDSimulation]]: ...

def run_policy_simulation(
    policy_name: str,
    base_config: Optional[OPDConfig] = None,
    alpha: float = 0.5,
    seed: int = 42,
    service_predictor: Optional[Callable[[Patient], float]] = None,
    admit_predictor: Optional[Callable[[Patient], float]] = None,
    return_sim: bool = False,
    **kwargs: Any
) -> Union[Tuple[pd.DataFrame, Dict[str, float]], Tuple[pd.DataFrame, Dict[str, float], HospitalOPDSimulation]]:
    """
    Executes an OPD simulation run under a designated policy.
    Returns (patient_dataframe, kpi_dictionary) or (patient_dataframe, kpi_dictionary, sim).
    """
    config = base_config or OPDConfig()
    config.policy = policy_name
    config.alpha = alpha
    config.seed = seed

    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # Initialize simulation
    sim = HospitalOPDSimulation(config)

    # Register predictors if supplied
    if service_predictor is not None:
        sim.service_time_predictor = service_predictor
    if admit_predictor is not None:
        sim.admit_predictor = admit_predictor

    # Innovation 3: Dynamic Regime Adjuster
    if policy_name == "regime_aware":
        # SimPy process to monitor regime and dynamically adapt staffing
        def regime_supervisor(env, sim_instance):
            while True:
                t = env.now
                arr_rate = sim_instance.arrival_rate_at(t)
                q_len = len(sim_instance.waiting_queue)

                # Classify regime
                if arr_rate >= 32.0 or q_len >= 14:
                    sim_instance.current_regime = "Surge"
                    sim_instance.doctors._capacity = max(config.num_doctors, 8)
                    # Mobilize both floating doctors if available
                    for fdoc in sim_instance.floating_doctors:
                        if fdoc.status == "Offline":
                            fdoc.set_status("Available", env.now)
                elif arr_rate >= 22.0 or q_len >= 7:
                    sim_instance.current_regime = "Peak"
                    sim_instance.doctors._capacity = max(config.num_doctors, 7)
                    # Mobilize first floating doctor if available
                    if sim_instance.floating_doctors:
                        if sim_instance.floating_doctors[0].status == "Offline":
                            sim_instance.floating_doctors[0].set_status("Available", env.now)
                else:
                    sim_instance.current_regime = "Normal"
                    sim_instance.doctors._capacity = config.num_doctors
                    # In normal regime, return available idle floaters to offline
                    for fdoc in sim_instance.floating_doctors:
                        if fdoc.status == "Available" and fdoc.current_patient is None:
                            fdoc.set_status("Offline", env.now)

                yield env.timeout(15.0)  # Check every 15 minutes

        sim.env.process(regime_supervisor(sim.env, sim))

    # Proactive staffing policy (uses forecasted arrival volume)
    elif policy_name == "proactive_staffing":
        def proactive_staffing_supervisor(env, sim_instance):
            while True:
                t = env.now
                next_hr_rate = sim_instance.arrival_rate_at(t + 60.0)
                if next_hr_rate > 35.0:
                    sim_instance.doctors._capacity = config.num_doctors + 2
                elif next_hr_rate > 25.0:
                    sim_instance.doctors._capacity = config.num_doctors + 1
                else:
                    sim_instance.doctors._capacity = config.num_doctors
                yield env.timeout(60.0)

        sim.env.process(proactive_staffing_supervisor(sim.env, sim))

    df = sim.run()
    kpis = sim.get_kpis()
    if return_sim or kwargs.get("return_sim", False):
        return df, kpis, sim
    return df, kpis


if __name__ == "__main__":
    policies = ["fcfs", "strict_priority", "sjf", "aging_priority", "predict_then_decide", "regime_aware"]
    print("Testing All OPD Policies on a Standard Day:")
    print("-" * 75)
    for pol in policies:
        _, kpi = run_policy_simulation(pol, OPDConfig(sim_days=1, num_doctors=6), alpha=0.5, seed=42)
        print(f"Policy: {pol:22s} | Mean Wait: {kpi['mean_wait']:5.1f}m | Em Wait: {kpi['emergency_wait']:4.1f}m | "
              f"Routine Wait: {kpi['routine_wait']:5.1f}m | Doc Util: {kpi['doctor_utilization']:4.1f}%")
