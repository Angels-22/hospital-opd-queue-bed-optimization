"""
Explainable Reinforcement Learning: Policy Distillation into Decision Trees.

INNOVATION 4:
- Collects state-action trajectories from the trained PPO Reinforcement Learning agent.
- Trains an interpretable surrogate Decision Tree Classifier to distill the neural policy.
- Evaluates policy Fidelity (degree of agreement between Tree and Neural Agent).
- Plots the distilled Decision Tree and Feature Importances for clinical transparency.
- Exports human-readable decision rules and before/after comparison.
"""

from __future__ import annotations
import os
import pickle
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from stable_baselines3 import PPO

from src.rl_env import HospitalOPDEnv


FEATURE_NAMES = [
    "queue_emergency",
    "queue_urgent",
    "queue_routine",
    "wait_emergency",
    "wait_urgent",
    "wait_routine",
    "doctors_busy",
    "gen_beds_occupied",
    "icu_beds_occupied",
    "hour_of_day",
]

ACTION_LABELS = ["Emergency", "Urgent", "Routine"]


class PolicyDistiller:
    """Distills black-box RL neural policies into interpretable decision trees."""

    def __init__(
        self,
        model_path: str = "data/trained_models/ppo_hospital_agent.zip",
        max_depth: int = 5,
        seed: int = 42
    ):
        self.model_path = model_path
        self.max_depth = max_depth
        self.seed = seed
        self.tree_model: DecisionTreeClassifier | None = None
        self.fidelity: float = 0.0
        self.feature_importances_: Dict[str, float] = {}

    def collect_trajectories(self, num_episodes: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the trained RL agent in the OPD environment to record (state, action) pairs."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RL model not found at {self.model_path}. Run train_rl.py first.")

        agent = PPO.load(self.model_path)
        env = HospitalOPDEnv(seed=self.seed)

        states: List[np.ndarray] = []
        actions: List[int] = []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=self.seed + ep)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                states.append(obs)
                actions.append(int(action))
                obs, reward, terminated, truncated, _ = env.step(action)

        X = np.array(states)
        y = np.array(actions)
        return X, y

    def distill(self, num_episodes: int = 25) -> Dict[str, Any]:
        """Fits the surrogate Decision Tree to mimic the neural agent and evaluates fidelity."""
        X, y = self.collect_trajectories(num_episodes=num_episodes)

        self.tree_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion="entropy",
            random_state=self.seed
        )
        self.tree_model.fit(X, y)

        y_pred = self.tree_model.predict(X)
        self.fidelity = float(accuracy_score(y, y_pred))

        # Extract feature importances
        self.feature_importances_ = {
            f: round(float(imp), 4)
            for f, imp in zip(FEATURE_NAMES, self.tree_model.feature_importances_, strict=False)
        }

        # Save distilled tree model
        save_dir = "data/trained_models"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "distilled_tree.pkl"), "wb") as f:
            pickle.dump(self.tree_model, f)

        # Generate rule explanation
        rules_text = export_text(
            self.tree_model,
            feature_names=FEATURE_NAMES
        )

        return {
            "fidelity": round(self.fidelity, 4),
            "total_samples": len(X),
            "feature_importances": self.feature_importances_,
            "rules_text": rules_text,
        }

    def generate_explanation_plots(self, save_dir: str = "data/rl_plots") -> Dict[str, str]:
        """Generates plots of the distilled tree architecture and feature importances."""
        if self.tree_model is None:
            self.distill()

        os.makedirs(save_dir, exist_ok=True)
        paths = {}

        # 1. Feature Importances Bar Chart
        fig, ax = plt.subplots(figsize=(9, 5))
        sorted_feats = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_feats]
        scores = [x[1] for x in sorted_feats]

        ax.barh(names[::-1], scores[::-1], color="#2980b9", edgecolor="black", alpha=0.85)
        ax.set_title(f"Distilled RL Feature Importances (Fidelity: {self.fidelity*100:.1f}%)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Importance (Entropy Reduction)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        p1 = os.path.join(save_dir, "rl_feature_importances.png")
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        paths["feature_importances"] = p1

        # 2. Decision Tree Visualization
        fig, ax = plt.subplots(figsize=(16, 9))
        # Unique classes present in fitted model
        present_classes = [ACTION_LABELS[c] for c in self.tree_model.classes_]
        plot_tree(
            self.tree_model,
            feature_names=FEATURE_NAMES,
            class_names=present_classes,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax
        )
        ax.set_title(f"Distilled Decision Tree Policy (Max Depth: {self.max_depth})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        p2 = os.path.join(save_dir, "rl_distilled_tree.png")
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        paths["distilled_tree"] = p2

        return paths


if __name__ == "__main__":
    distiller = PolicyDistiller()
    print("Distilling trained PPO policy into Decision Tree...")
    res = distiller.distill(num_episodes=20)
    print(f"\nDistillation Fidelity: {res['fidelity']*100:.2f}% across {res['total_samples']} states")
    print("\nFeature Importances:")
    for f, imp in sorted(res['feature_importances'].items(), key=lambda x: x[1], reverse=True):
        if imp > 0.001:
            print(f"  - {f:20s}: {imp:.4f}")

    plots = distiller.generate_explanation_plots()
    print("\nGenerated Explanation Plots:", plots)
