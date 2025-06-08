#!/usr/bin/env python3
# filepath: /Users/antoine2/Projects/gym-adr/examples/train_sb3_agent.py
"""
Example script to train and evaluate a Stable Baselines 3 agent on the ADR environment.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

import gym_adr  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Stable Baselines 3 agent on the ADR environment"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "A2C", "DQN"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=20000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--n_debris", type=int, default=10, help="Number of debris in the environment"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--render", action="store_true", help="Render the environment after training"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create the environment
    env = gym.make("gym_adr/ADR-v0", render_mode="human" if args.render else None)

    # Wrap the environment with Monitor for logging
    env = Monitor(env)

    # Validate the environment
    check_env(env.unwrapped)

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create the model based on the specified algorithm
    if args.algo == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed)
    elif args.algo == "A2C":
        model = A2C("MultiInputPolicy", env, verbose=1, seed=args.seed)
    elif args.algo == "DQN":
        model = DQN(
            policy="MultiInputPolicy",
            env=env,
            #             learning_rate=0.0001,
            #             buffer_size=500000,
            #             batch_size=64,
            #             exploration_fraction = 0.2,
            #             stats_window_size=1000,
            #             train_freq=(1, "episode"),
            #             gradient_steps=4,
            verbose=1,
            seed=args.seed,
        )

    # Train the agent
    print(f"Training {args.algo} agent for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    # Save the model if requested
    if args.save_model:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{args.algo}_{args.n_debris}_debris")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Test the agent and render if requested
    if args.render:
        print("Testing the trained agent with rendering...")
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode finished with reward: {total_reward}")
        env.render()  # Render the episode

    env.close()


if __name__ == "__main__":
    main()
