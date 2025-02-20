import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_adr  # noqa: F401

def test_env():
    env = gym.make('gym_adr')
    check_env(env.unwrapped)