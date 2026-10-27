"""
Gymnasium Reinforcement Learning Environment for Hospital OPD Dispatching.

State Space (10 continuous dimensions):
- queue_emergency: count of emergency patients waiting
- queue_urgent: count of urgent patients waiting
- queue_routine: count of routine patients waiting
- wait_emergency: average waiting time of emergency patients (minutes)
- wait_urgent: average waiting time of urgent patients (minutes)
- wait_routine: average waiting time of routine patients (minutes)
- doctors_busy: count of busy doctors (0 to num_doctors)
- gen_beds_occupied: count of occupied general ward beds
- icu_beds_occupied: count of occupied ICU beds
- hour_of_day: current hour of day (8.0 to 20.0)

Action Space (Discrete 3):
- 0: Dispatch Emergency patient
- 1: Dispatch Urgent patient
- 2: Dispatch Routine patient
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HospitalOPDEnv(gym.Env):
    """Gymnasium Environment for Outpatient Department Queue Dispatching."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_doctors: int = 6,
        num_gen_beds: int = 40,
        num_icu_beds: int = 10,
        episode_length_steps: int = 200,
        seed: int = 42
    ):
        super().__init__()
        self.num_doctors = num_doctors
        self.num_gen_beds = num_gen_beds
        self.num_icu_beds = num_icu_beds
        self.episode_length_steps = episode_length_steps
        self.current_step = 0
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # 10 State variables
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 8.0], dtype=np.float32),
            high=np.array([50, 80, 150, 120, 180, 240, num_doctors, num_gen_beds, num_icu_beds, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # 3 Actions: Dispatch Emergency (0), Urgent (1), Routine (2)
        self.action_space = spaces.Discrete(3)

        self.state = np.zeros(10, dtype=np.float32)
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to initial morning state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        # Morning start (8:00 AM)
        self.state = np.array([
            self.rng.integers(0, 3),   # Emergency queue
            self.rng.integers(1, 6),   # Urgent queue
            self.rng.integers(2, 10),  # Routine queue
            float(self.rng.uniform(1.0, 5.0)),   # Em wait
            float(self.rng.uniform(3.0, 10.0)),  # Ur wait
            float(self.rng.uniform(5.0, 15.0)),  # Ro wait
            self.rng.integers(1, 4),             # Busy doctors
            self.rng.integers(5, 15),            # Occupied general beds
            self.rng.integers(1, 4),             # Occupied ICU beds
            8.0,                                 # 8:00 AM
        ], dtype=np.float32)

        return self.state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Simulates one dispatch step and updates hospital queues and beds."""
        self.current_step += 1
        q_em, q_ur, q_ro, w_em, w_ur, w_ro, busy_docs, gen_occ, icu_occ, hr = self.state

        # Advance time by ~3 minutes per step
        dt = 0.05  # ~3 minutes in hour fractions
        hr = min(20.0, hr + dt)

        # Simulate new arrivals (peaking between 9am - 12pm)
        is_peak = 9.0 <= hr <= 12.0
        lam = 2.0 if is_peak else 1.0

        new_em = int(self.rng.poisson(0.15 * lam))
        new_ur = int(self.rng.poisson(0.40 * lam))
        new_ro = int(self.rng.poisson(0.85 * lam))

        q_em += new_em
        q_ur += new_ur
        q_ro += new_ro

        # Dispatch action
        dispatched_acuity = None
        action_penalty = 0.0
        service_reward = 0.0

        if action == 0:  # Emergency
            if q_em > 0:
                q_em -= 1
                dispatched_acuity = "Emergency"
                w_em = max(0.0, w_em - 3.0)
                service_reward = 12.0
            else:
                action_penalty = -25.0  # Invalid action on empty emergency queue
        elif action == 1:  # Urgent
            if q_ur > 0:
                q_ur -= 1
                dispatched_acuity = "Urgent"
                w_ur = max(0.0, w_ur - 2.5)
                service_reward = 9.0
            else:
                action_penalty = -25.0  # Invalid action on empty urgent queue
        else:  # Routine
            if q_ro > 0:
                q_ro -= 1
                dispatched_acuity = "Routine"
                w_ro = max(0.0, w_ro - 2.0)
                service_reward = 7.0
            else:
                action_penalty = -25.0  # Invalid action on empty routine queue

        # Queue aging (waiting times increment for patients in queue)
        w_em = (w_em + 1.5) if q_em > 0 else 0.0
        w_ur = (w_ur + 1.2) if q_ur > 0 else 0.0
        w_ro = (w_ro + 1.0) if q_ro > 0 else 0.0

        # Doctor capacity dynamics
        busy_docs = np.clip(busy_docs + self.rng.choice([-1, 0, 1]), 1, self.num_doctors)

        # Bed occupancy dynamics
        if dispatched_acuity == "Emergency" and self.rng.random() < 0.40:
            icu_occ = min(self.num_icu_beds, icu_occ + 1)
        elif (dispatched_acuity in ["Emergency", "Urgent"]) and self.rng.random() < 0.25:
            gen_occ = min(self.num_gen_beds, gen_occ + 1)

        # Bed discharges
        if self.rng.random() < 0.15:
            gen_occ = max(0, gen_occ - 1)
        if self.rng.random() < 0.08:
            icu_occ = max(0, icu_occ - 1)

        # Balanced clinical reward function:
        wait_penalty = (w_em * 2.5) + (w_ur * 0.8) + (w_ro * 0.3)
        starvation_penalty = (w_ro * 0.5) if w_ro > 30.0 else 0.0

        reward = float(service_reward - wait_penalty - starvation_penalty + action_penalty)

        self.state = np.array([
            q_em, q_ur, q_ro,
            w_em, w_ur, w_ro,
            busy_docs, gen_occ, icu_occ, hr
        ], dtype=np.float32)

        terminated = bool(self.current_step >= self.episode_length_steps or hr >= 20.0)
        truncated = False

        info = {
            "dispatched_acuity": dispatched_acuity,
            "wait_penalty": wait_penalty,
            "starvation_penalty": starvation_penalty,
            "emergency_wait": w_em,
            "routine_wait": w_ro,
        }

        return self.state.copy(), reward, terminated, truncated, info


if __name__ == "__main__":
    env = HospitalOPDEnv()
    obs, _ = env.reset()
    print("HospitalOPDEnv initialized.")
    print("Initial Observation:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Action: {action} -> Reward: {reward:6.2f} | Info: {info}")
