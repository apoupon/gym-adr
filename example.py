import argparse

import gymnasium as gym
import wandb

import gym_adr  # noqa: F401

def main(use_wandb: bool = False):
    if use_wandb:
        wandb.login()
        run = wandb.init(project="gym-adr")
    else:
        run = None

    env = gym.make("gym_adr/ADR-v0", render_mode="human")
    observation, info = env.reset()

    for _ in range(1001):
        print("iteration ", _)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("action : ", action)
        print("observation : ", observation)

        if use_wandb:
            wandb.log(
                {
                    "removal_step": observation["step_and_debris"][0],
                    "number_debris_left": observation["step_and_debris"][1],
                    "current_removing_debris": observation["step_and_debris"][2],
                    "dv_left": observation["fuel_time_constraints"][0],
                    "dt_left": observation["fuel_time_constraints"][1],
                    "step_reward": reward,
                    "terminated": terminated,
                }
            )

        if terminated or truncated:
            print("Episode terminated ! Reset ongoing...")
            # env.render() if _ > 50 else None
            observation, info = env.reset()

    env.close()
    if use_wandb and run is not None:
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADR Gym environment with optional W&B logging.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable logging to Weights & Biases.")
    args = parser.parse_args()

    main(args.use_wandb)