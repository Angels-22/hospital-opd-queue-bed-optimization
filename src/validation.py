"""
Theoretical Queuing Model Validation and Little's Law Verification.

Validates the discrete-event simulator against:
1. Classical M/M/c Multi-Server Queuing Theory (Erlang-C Formula).
2. Little's Law (L = lambda * W).
Compares analytical theoretical predictions against empirical simulation results
across varying traffic intensities (rho in [0.50, 0.85]).
"""

from __future__ import annotations
import math
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
import simpy


# ---------------------------------------------------------------------------
# Analytical M/M/c Queuing Formulae (Erlang-C)
# ---------------------------------------------------------------------------

def erlang_c_theoretical(lam: float, mu: float, c: int) -> Dict[str, float]:
    """
    Computes exact steady-state theoretical metrics for an M/M/c queue:
    - lam: arrival rate (patients per minute)
    - mu: service rate per server (patients per minute)
    - c: number of parallel servers (doctors)
    Returns: {rho, P0, Pwait, Lq, Wq, W, L}
    """
    a = lam / mu  # Offered load (erlangs)
    rho = a / c   # Server utilization

    if rho >= 1.0:
        raise ValueError(f"M/M/c queue is unstable: traffic intensity rho={rho:.3f} >= 1.0")

    # Sum of terms for P0
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    c_term = ((a ** c) / math.factorial(c)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (sum_terms + c_term)

    # Erlang-C probability of waiting
    p_wait = c_term * p0

    # Expected queue length (Lq) and wait in queue (Wq)
    lq = (p_wait * rho) / (1.0 - rho)
    wq = lq / lam  # in minutes

    # Expected system time (W) and system count (L)
    w = wq + (1.0 / mu)
    l = lam * w

    return {
        "rho": round(rho, 4),
        "P0": round(p0, 4),
        "Pwait": round(p_wait, 4),
        "Lq": round(lq, 3),
        "Wq": round(wq, 3),
        "W": round(w, 3),
        "L": round(l, 3),
    }


# ---------------------------------------------------------------------------
# Simulator Validation Runner
# ---------------------------------------------------------------------------

def simulate_stationary_mmc(
    lam_per_hour: float,
    mean_service_minutes: float,
    num_doctors: int,
    sim_hours: float = 80.0,
    seed: int = 42
) -> Dict[str, float]:
    """
    Simulates a stationary M/M/c queue with Poisson arrivals and exponential service times.
    Collects empirical L, W, Wq, Lq, and verifies Little's Law.
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    doctors = simpy.Resource(env, capacity=num_doctors)

    lam_per_minute = lam_per_hour / 60.0

    waits = []
    service_times = []
    total_times = []

    # Queue length sampling
    queue_samples = []

    def queue_monitor():
        while True:
            # Current queue depth + in service
            q_len = len(doctors.queue)
            in_service = doctors.count
            queue_samples.append((q_len, in_service))
            yield env.timeout(1.0)  # sample every minute

    def arrival_process():
        while True:
            inter_arrival = rng.exponential(1.0 / lam_per_minute)
            yield env.timeout(inter_arrival)
            env.process(patient_process(env.now))

    def patient_process(arr_time: float):
        req = doctors.request()
        yield req
        wait_dur = env.now - arr_time
        serv_dur = rng.exponential(mean_service_minutes)
        waits.append(wait_dur)
        service_times.append(serv_dur)
        yield env.timeout(serv_dur)
        total_times.append(wait_dur + serv_dur)
        doctors.release(req)

    env.process(arrival_process())
    env.process(queue_monitor())
    env.run(until=sim_hours * 60.0)

    # Discard first 20% as warmup
    warmup_cutoff = int(len(waits) * 0.20)
    warmup_samples_cutoff = int(len(queue_samples) * 0.20)

    post_waits = waits[warmup_cutoff:]
    post_total = total_times[warmup_cutoff:]
    post_samples = queue_samples[warmup_samples_cutoff:]

    emp_wq = float(np.mean(post_waits))
    emp_w = float(np.mean(post_total))
    emp_lq = float(np.mean([s[0] for s in post_samples]))
    emp_l = float(np.mean([s[0] + s[1] for s in post_samples]))
    emp_rho = float(np.mean([s[1] / num_doctors for s in post_samples]))

    # Little's Law check: L_pred = lam * W
    littles_l_pred = lam_per_minute * emp_w
    littles_rel_error_pct = abs(emp_l - littles_l_pred) / max(0.01, emp_l) * 100.0

    return {
        "lam_per_hour": lam_per_hour,
        "emp_Wq": round(emp_wq, 3),
        "emp_W": round(emp_w, 3),
        "emp_Lq": round(emp_lq, 3),
        "emp_L": round(emp_l, 3),
        "emp_rho": round(emp_rho, 3),
        "littles_L_predicted": round(littles_l_pred, 3),
        "littles_rel_error_pct": round(littles_rel_error_pct, 2),
    }


def validate_queuing_theory(
    num_doctors: int = 5,
    mean_service_min: float = 12.0,
    arrival_rates_per_hour: Optional[List[float]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Runs systematic validation across multiple traffic intensities (rho = 0.60 to 0.94).
    Compares Erlang-C analytical theory with SimPy simulation outputs.
    """
    if arrival_rates_per_hour is None:
        arrival_rates_per_hour = [15.0, 18.0, 20.0, 22.0, 23.5]

    mu_per_minute = 1.0 / mean_service_min
    rows = []

    for arr_rate in arrival_rates_per_hour:
        lam_per_minute = arr_rate / 60.0
        theory = erlang_c_theoretical(lam_per_minute, mu_per_minute, num_doctors)
        sim_res = simulate_stationary_mmc(arr_rate, mean_service_min, num_doctors, sim_hours=80.0, seed=seed)

        wq_err_pct = abs(sim_res["emp_Wq"] - theory["Wq"]) / max(0.1, theory["Wq"]) * 100.0

        rows.append({
            "Arrival_Rate_hr": arr_rate,
            "Theoretical_Rho": theory["rho"],
            "Empirical_Rho": sim_res["emp_rho"],
            "Theory_Wq_min": theory["Wq"],
            "Sim_Wq_min": sim_res["emp_Wq"],
            "Wq_Relative_Error_Pct": round(wq_err_pct, 2),
            "Theory_L": theory["L"],
            "Sim_L": sim_res["emp_L"],
            "Littles_Law_Pred_L": sim_res["littles_L_predicted"],
            "Littles_Law_Error_Pct": sim_res["littles_rel_error_pct"],
        })

    return pd.DataFrame(rows)


def generate_validation_plots(
    val_df: Optional[pd.DataFrame] = None,
    save_dir: str = "data/validation_plots"
) -> Dict[str, str]:
    """Generates M/M/c theoretical vs simulated curves and Little's Law validation plots."""
    if val_df is None:
        val_df = validate_queuing_theory()

    os.makedirs(save_dir, exist_ok=True)
    paths = {}

    # 1. Theoretical vs Simulated Queue Wait (Wq) Curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(val_df["Theoretical_Rho"], val_df["Theory_Wq_min"],
            color="#2980b9", lw=2.5, label="M/M/c Erlang-C Theory ($W_q$)")
    ax.scatter(val_df["Empirical_Rho"], val_df["Sim_Wq_min"],
               color="#e74c3c", s=70, zorder=5, label="SimPy Discrete-Event Simulation")

    ax.set_title("Simulator Validation Against M/M/c Queuing Theory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Traffic Intensity / Server Utilization (rho)", fontsize=11)
    ax.set_ylabel("Average Queue Wait Time $W_q$ (Minutes)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    p1 = os.path.join(save_dir, "mmc_validation.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    paths["mmc_validation"] = p1

    # 2. Little's Law Verification Plot (L vs lambda * W)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_df["Theory_L"], val_df["Theory_L"], color="#7f8c8d", linestyle="--", label="Ideal Identity (L = lambda*W)")
    ax.scatter(val_df["Sim_L"], val_df["Littles_Law_Pred_L"],
               color="#27ae60", s=75, zorder=5, label="Simulated Observations")

    ax.set_title(r"Little's Law Validation: Empirical $L$ vs $\lambda W$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Empirical Average System Count ($L$)", fontsize=11)
    ax.set_ylabel(r"Predicted System Count ($\lambda \cdot W$)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    p2 = os.path.join(save_dir, "littles_law_validation.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    paths["littles_law_validation"] = p2

    return paths


if __name__ == "__main__":
    print("Validating SimPy Engine against M/M/c Queuing Theory and Little's Law...")
    validation_results = validate_queuing_theory()
    print("\n" + "=" * 90)
    print("QUEUING THEORY & LITTLE'S LAW VALIDATION BENCHMARK")
    print("=" * 90)
    print(validation_results.to_string(index=False))

    plots = generate_validation_plots(validation_results)
    print("\nGenerated Validation Diagnostic Plots:", plots)
