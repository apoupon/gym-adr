# import gymnasium as gym
import wandb
from gym_adr.envs.adr import ADREnv  # import gym_adr

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(project="gym-adr")

    env = ADREnv()  # gym.make("gym_adr/ADR-v0", render_mode="human")
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()
        print("action : ", action)
        print("observation : ", observation)

        wandb.log(
            {
                "removal_step": len(observation["step_and_debris"]),
                "current_removing_debris": observation["step_and_debris"][-1],
                "dv_left": observation["fuel_time_constraints"][0],
                "dt_left": observation["fuel_time_constraints"][1],
                "step_reward": reward,
                "terminated": terminated,
            }
        )

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
