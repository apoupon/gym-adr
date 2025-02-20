import gymnasium as gym
from gym_adr.envs.adr import ADREnv # import gym_adr

if __name__ == "__main__":
    env = ADREnv() # gym.make("gym_adr/ADR-v0", render_mode="human")
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        image = env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()