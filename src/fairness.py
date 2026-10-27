"""
Fairness Auditing, Starvation Analysis, and Alpha Trade-off Sweeper.

INNOVATION 1 Module:
1. Subgroup Waiting Time Analysis across Demographics:
   - Age Groups (<18, 18-40, 41-65, >65)
   - Gender (Female, Male, Non-binary)
   - Area (Urban vs Rural)
   - Disparity Ratios and Statistical Tests
2. Routine Patient Starvation Analysis:
   - Quantifies starvation percentage (wait > 45m or > 60m) under FCFS,
     Strict Priority, and Aging Priority.
3. Alpha Parameter Sweep & Trade-off Frontier:
   - Sweeps alpha in [0.0, 2.0] for Aging Priority:
     Score(t) = Acuity_Weight + alpha * Wait_Time
   - Plots Emergency Wait vs Routine Wait vs Starvation Frontier.
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.policies import run_policy_simulation
from src.simulation import OPDConfig


def audit_demographic_fairness(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Computes waiting times, admission rates, and disparity metrics across
    demographic subgroups (Age Group, Gender, Area).
    """
    if df.empty:
        return {}

    # 1. By Age Group
    age_fairness = df.groupby("age_group").agg(
        patient_count=("patient_id", "count"),
        mean_wait_min=("wait_time", "mean"),
        median_wait_min=("wait_time", "median"),
        p95_wait_min=("wait_time", lambda s: float(np.percentile(s, 95))),
        admission_rate=("admitted", "mean")
    ).loc[["<18", "18-40", "41-65", ">65"]].round(2)

    # 2. By Gender
    gender_fairness = df.groupby("gender").agg(
        patient_count=("patient_id", "count"),
        mean_wait_min=("wait_time", "mean"),
        median_wait_min=("wait_time", "median"),
        p95_wait_min=("wait_time", lambda s: float(np.percentile(s, 95))),
        admission_rate=("admitted", "mean")
    ).round(2)

    # 3. By Area (Urban vs Rural)
    area_fairness = df.groupby("area").agg(
        patient_count=("patient_id", "count"),
        mean_wait_min=("wait_time", "mean"),
        median_wait_min=("wait_time", "median"),
        p95_wait_min=("wait_time", lambda s: float(np.percentile(s, 95))),
        admission_rate=("admitted", "mean")
    ).round(2)

    # Statistical two-sample t-test between Urban and Rural wait times
    urban_waits = df[df["area"] == "Urban"]["wait_time"].dropna()
    rural_waits = df[df["area"] == "Rural"]["wait_time"].dropna()
    t_stat, p_val = stats.ttest_ind(urban_waits, rural_waits, equal_var=False)

    area_fairness["t_test_p_value"] = round(float(p_val), 4)

    return {
        "age_fairness": age_fairness,
        "gender_fairness": gender_fairness,
        "area_fairness": area_fairness,
    }


def compute_starvation_metrics(df: pd.DataFrame, threshold_minutes: float = 45.0) -> Dict[str, float]:
    """
    Computes routine patient starvation metrics:
    - Percentage of routine patients waiting longer than threshold_minutes.
    - Max routine wait time.
    """
    routine_df = df[df["triage_class"] == "Routine"]
    if routine_df.empty:
        return {"routine_starvation_rate": 0.0, "max_routine_wait": 0.0}

    starved_count = (routine_df["wait_time"] > threshold_minutes).sum()
    starvation_rate = (starved_count / len(routine_df)) * 100.0
    max_wait = float(routine_df["wait_time"].max())

    return {
        "routine_starvation_rate": round(float(starvation_rate), 2),
        "max_routine_wait": round(max_wait, 2),
        "routine_patients_count": len(routine_df),
        "starved_patients_count": int(starved_count),
    }


def sweep_alpha_tradeoff(
    alphas: Optional[List[float]] = None,
    sim_days: int = 1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sweeps alpha parameter in Aging Priority:
    Score = Acuity_Weight + alpha * Wait_Time
    Quantifies the clinical trade-off between Emergency Wait and Routine Starvation.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    rows = []
    base_cfg = OPDConfig(sim_days=sim_days, num_doctors=6, seed=seed)

    for a in alphas:
        df, kpis = run_policy_simulation(
            policy_name="aging_priority",
            base_config=base_cfg,
            alpha=a,
            seed=seed
        )
        starve_info = compute_starvation_metrics(df, threshold_minutes=45.0)
        em_wait = kpis["emergency_wait"]
        ur_wait = kpis["urgent_wait"]
        ro_wait = kpis["routine_wait"]

        rows.append({
            "alpha": a,
            "overall_mean_wait": kpis["mean_wait"],
            "emergency_wait": em_wait,
            "urgent_wait": ur_wait,
            "routine_wait": ro_wait,
            "p95_wait": kpis["p95_wait"],
            "routine_starvation_rate": starve_info["routine_starvation_rate"],
            "max_routine_wait": starve_info["max_routine_wait"],
            "doctor_utilization": kpis["doctor_utilization"],
        })

    return pd.DataFrame(rows)


def generate_fairness_plots(
    sweep_df: Optional[pd.DataFrame] = None,
    save_dir: str = "data/fairness_plots"
) -> Dict[str, str]:
    """Generates and saves fairness and alpha trade-off diagnostic plots."""
    if sweep_df is None:
        sweep_df = sweep_alpha_tradeoff()

    os.makedirs(save_dir, exist_ok=True)
    paths = {}

    # 1. Alpha Trade-off Curve: Emergency Wait vs Routine Wait
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(sweep_df["alpha"], sweep_df["emergency_wait"], marker="o", color="#e74c3c", lw=2.2, label="Emergency Wait (min)")
    line2 = ax1.plot(sweep_df["alpha"], sweep_df["routine_wait"], marker="s", color="#2ecc71", lw=2.2, label="Routine Wait (min)")
    line3 = ax2.plot(sweep_df["alpha"], sweep_df["routine_starvation_rate"], marker="^", color="#f39c12", lw=2.0, linestyle="--", label="Routine Starvation Rate (%)")

    ax1.set_xlabel("Aging Priority Parameter (alpha)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Wait Time (Minutes)", fontsize=11, color="#2c3e50")
    ax2.set_ylabel("Routine Starvation Rate (% > 45 min)", fontsize=11, color="#d35400")

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    ax1.set_title("Fairness-Acuity Trade-off: Emergency Response vs Routine Starvation", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    p1 = os.path.join(save_dir, "alpha_fairness_tradeoff.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    paths["alpha_tradeoff"] = p1

    return paths


if __name__ == "__main__":
    print("Sweeping Alpha parameter in Fairness-Aware Aging Priority...")
    sweep_results = sweep_alpha_tradeoff()
    print("\nAlpha Sweep Results Table:\n")
    print(sweep_results.to_string(index=False))

    plots = generate_fairness_plots(sweep_results)
    print("\nGenerated Fairness Plots:", plots)
