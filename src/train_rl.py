"""
Reinforcement Learning Training Module for Hospital OPD Dispatching.

Trains a Proximal Policy Optimization (PPO) agent from stable-baselines3
on the Gymnasium HospitalOPDEnv.
Saves the trained policy to data/trained_models/ppo_hospital_agent.zip.
"""

from __future__ import annotations
import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from src.rl_env import HospitalOPDEnv


def train_rl_agent(
    total_timesteps: int = 20000,
    save_path: str = "data/trained_models/ppo_hospital_agent.zip",
    seed: int = 42
) -> PPO:
    """
    Trains a PPO reinforcement learning agent on the HospitalOPDEnv.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create vectorized environment
    env = make_vec_env(lambda: HospitalOPDEnv(seed=seed), n_envs=1)

    print(f"Initializing PPO agent for {total_timesteps} timesteps...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.08,
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)

    # Evaluate trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"PPO Training Complete. Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save model
    model.save(save_path)
    print(f"Trained PPO agent saved to {save_path}")

    return model


if __name__ == "__main__":
    train_rl_agent(total_timesteps=15000)
