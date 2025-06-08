from gymnasium.envs.registration import register

register(id="gym_adr/ADR-v0", nondeterministic=True, entry_point="gym_adr.adr:ADREnv")
