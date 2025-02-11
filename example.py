import gymnasium as gym
import wandb

import gym_adr # noqa: F401

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(project="gym-adr")

    env = gym.make("gym_adr/ADR-v0", render_mode="human")
    observation, info = env.reset()

    for _ in range(1001):
        print("iteration ", _)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("action : ", action)
        print("observation : ", observation)

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
            env.render() if _ > 50 else None
            observation, info = env.reset()

    env.close()
