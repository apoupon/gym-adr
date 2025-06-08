from gymnasium.envs.registration import register

import gym_adr  # noqa: F401

register(id="gym_adr/ADR-v0", nondeterministic=True, entry_point="gym_adr.adr:ADREnv")
