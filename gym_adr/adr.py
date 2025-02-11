import copy
from typing import Optional, List, Union

import random
import numpy as np
import pandas as pd
import gymnasium as gym
from astropy import units as u
import multiprocessing

from gym_adr.space_physics.simulator import Simulator


DEBUG = True


def seed_everything(seed: Optional[int] = None):
    """
    Set the seed for random number generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


class ADREnv(gym.Env):
    """
    ## Description

    Active Debris Removal environment.

    The goal of the agent is to deorbit as many debris as possible given constraints on fuel
    and time. The agent is an Orbital Transfer Vehicle (OTV).

    ## Action Space

    The action space is discrete and consists of n values, n being the number of debris in
    orbit around the Earth.

    ## Observation Space

    The observation space is a (5+2n)-dimensional vector representing the state of the agent:
    - removal_step: refer to the number of debris deorbited by the OTV
    - number_debris_left: refer to the number of debris still in orbits around the Earth
    - current_removing_debris: refer to the current target debris
    - dv_left: refer to the current amount of fuel available to the OTV
    - dt_left: refer to the current amount of time available to the OTV
    - binary_flag_debris_1, ..., binary_flag_debris_n: refer to the state of
        debris (0 is in orbit, 1 is already deorbited)
    - priority_score_debris_1, ..., priority_score_debris_n: refer to the priority
        score of debris (1 is not prioritary, 10 is prioritary = with a high chance of collision)

    ## Rewards

    The reward is 1 when the OTV deorbit an non-prioritary debris, 10 when it deorbit a prioritary
    debris, 0 if it doesn't deorbit any debris (no more fuel/time or debris already deorbited).

    ## Success Criteria

    The environment is considered solved if at least 95% debris in orbit have been deorbited during
    the mission.

    ## Starting State

    The agent starts at the position of a random debris. This debris is considered deorbited for the
    rest of the episode.

    ## Episode Termination

    The episode terminates when the OTV run out of fuel (or time) or when it chose as target debris
    a debris that has already been deorbited.

    ## Arguments

    * `total_n_debris`: (int) The number of total debris in orbit around the Earth. Default is `10`.
    * `dv_max_per_mission`: (int) The total amount of fuel available at the start of the mission.
        Default is `5`.
    * `dt_max_per_mission`: (int) The initial duration of the mission. Default is `100`.
    * `random_first_debris`: (bool) The debris chosen to initialize the position of the OTV.
        Default is `True`.
    * `first_debris`: (int) If `random_first_debris` is set to `False`, the debris chosen to initialize
        the position of the OTV. Default is `None`.

    ## Version History

    * v0: Original version
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        total_n_debris: int = 10,
        dv_max_per_mission: float = 5.0,
        dt_max_per_mission: float = 100.0,
        random_first_debris: bool = True,
        first_debris: Optional[int] = 3,
    ):
        super().__init__()
        self.total_n_debris = total_n_debris
        self.dv_max_per_mission = dv_max_per_mission  # [km/s]
        self.dt_max_per_mission = dt_max_per_mission  # [day]
        self.random_first_debris = random_first_debris
        self.first_debris = first_debris
        self.last_action_was_valid = True

        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris - 1)
        self.simulator = Simulator(
            starting_index=self.first_debris, n_debris=self.total_n_debris
        )

        self._initialize_observation_space()
        self.action_space = gym.spaces.Discrete(self.total_n_debris)

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f"Invalid render mode: {render_mode}. Must be None or one of {self.metadata['render_modes']}."
        self.render_mode = render_mode

    def step(self, action: np.int64):
        # Don't do the propagation if the action terminates the episode by binary flags (case not handled by the simulator)
        if self.binary_flags[action] == 1:
            observation = self.get_obs()
            reward = 0
            terminated = True
            info = {}

            self.last_action_was_valid = False
            return observation, reward, terminated, False, info

        # Use the simulator to compute the maneuvre fuel and time and propagate
        cv, dt_min = self.simulator.simulate_action(action)

        terminated = self.is_terminated(action, cv, dt_min)
        if terminated:
            observation = self.get_obs()
            reward = 0
            info = {}
            return observation, reward, terminated, False, info

        self.deorbited_debris.append(action)

        reward = self._compute_reward(action, terminated)

        self.transition_function(action=action, cv=cv, dt_min=dt_min)

        # reset priority list after computing reward
        self.priority_scores = np.ones(self.total_n_debris, dtype=int)

        # Modify priority list if there is a priority debris (high risk of collision)
        priority_debris = self.get_priority()
        if priority_debris:
            self.priority_scores[priority_debris] = 10

        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed: int = None, options=None):
        super().reset(seed=seed)
        seed_everything(seed)

        self.last_action_was_valid = True
        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris - 1)
        self.simulator.__init__(
            starting_index=self.first_debris, n_debris=self.total_n_debris
        )
        self.deorbited_debris = [self.first_debris]

        # initialize state
        state = np.concatenate(
            [
                np.array(
                    [
                        1,  # we initialize the OTV at the position of a debris and consider this debris as deorbited
                        self.total_n_debris
                        - 1,  # -1 because we consider the first debris as already deorbited
                        self.first_debris,
                        self.dv_max_per_mission,
                        self.dt_max_per_mission,
                    ]
                ),
                np.zeros(self.total_n_debris, dtype=int),
                np.ones(self.total_n_debris, dtype=int),
            ]
        )
        state[5 + self.first_debris] = 1
        self._set_state(state)

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def _initialize_observation_space(self):
        self.observation_space = gym.spaces.Dict(
            {
                # [removal_step, number_debris_left, current_removing_debris]
                "step_and_debris": gym.spaces.Box(
                    low=0, high=self.total_n_debris, shape=(3,), dtype=np.int64
                ),
                # [dv_left, dt_left]
                "fuel_time_constraints": gym.spaces.Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([self.dv_max_per_mission, self.dt_max_per_mission]),
                    dtype=np.float64,
                ),
                # [binary_flag_debris1, binary_flag_debris2...]
                "binary_flags": gym.spaces.MultiBinary(self.total_n_debris),
                # [priority_score_debris1, priority_score_debris2...]
                "priority_scores": gym.spaces.Box(
                    low=0,
                    high=self.total_n_debris + 1,
                    shape=(self.total_n_debris,),
                    dtype=np.int64,
                ),
            }
        )

    def get_obs(self):
        return {
            "step_and_debris": np.array(
                [
                    self.removal_step,
                    self.number_debris_left,
                    self.current_removing_debris,
                ],
                dtype=np.int64,
            ),
            "fuel_time_constraints": np.array(
                [self.dv_left, self.dt_left], dtype=np.float64
            ),
            "binary_flags": np.array(self.binary_flags, dtype=np.int8),
            "priority_scores": np.array(self.priority_scores, dtype=np.int64),
        }

    def _set_state(self, state: List[Union[int, float]]):
        self.removal_step = state[0]
        self.number_debris_left = state[1]
        self.current_removing_debris = state[2]
        self.dv_left = state[3]
        self.dt_left = state[4]
        self.binary_flags = state[5 : 5 + self.total_n_debris]
        self.priority_scores = state[
            5 + self.total_n_debris : 6 + self.total_n_debris * 2
        ]

    def get_info(self):
        info = {}

        return info

    def _compute_reward(self, action: np.int64, terminated: bool):
        # Calculate reward using the priority list
        reward = self.priority_scores[action]

        # Set reward to 0 if the action is not legal
        if terminated:
            reward = 0

        return reward

    def is_terminated(self, action: np.int64, cv: u.Quantity, dt_min: u.Quantity):
        # input is state before transition
        next_debris_index = action

        # 1st check: do we have enough time to go to the next debris ?
        # 2nd check: do we have enough fuel to go to the next debris ?
        # 3rd check: is the next debris still in orbit or not anymore ?
        if (self.dt_left * u.day - dt_min) < 0:
            print("Time limit reached, no more time left to deorbit debris.")
        elif (self.dv_left * (u.km / u.s) - cv) < 0:
            print("Fuel limit reached, no more fuel left to deorbit debris.")
        elif self.binary_flags[next_debris_index] == 1:
            print("Next debris already deorbited, cannot deorbit it again.")

        if (
            (self.dt_left * u.day - dt_min) < 0
            or (self.dv_left * (u.km / u.s) - cv) < 0
            or self.binary_flags[next_debris_index] == 1
        ):
            return True

        return False

    def transition_function(self, action: np.int64, cv: u.Quantity, dt_min: u.Quantity):
        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= dt_min.to(u.day).value
        self.dv_left -= cv.to(u.km / u.s).value

        # Update current removing debris after computing CB
        self.current_removing_debris = action
        self.binary_flags[self.current_removing_debris] = 1

    def get_priority(self):
        """
        Returns a random debris index to set as priority
        Taken from the available debris that have not been removed yet
        """
        # Get the list of indices where the binary flag is 0
        available_debris = [i for i, flag in enumerate(self.binary_flags) if flag == 0]

        if available_debris and random.random() < 0.3:
            # Randomly select a debris from the available list
            return random.choice(available_debris)

        return None

    def render(self, step_sec: int = 40):
        """
        Render the previous episode.
        """
        if self.render_mode is None:
            print(
                "Render mode is not set. Please set render_mode to 'human' to visualize the simulation."
            )
            return
        elif self.render_mode != "human":
            print(
                f"Render mode {self.render_mode} is not supported. Please set render_mode to 'human'."
            )
            return

        if len(self.deorbited_debris) <= 1:
            print(
                "OTV didn't manage to deorbit any debris, therefore there is nothing to visualize. Try again."
            )
            return
        elif self.last_action_was_valid is False:
            print(
                "Last action was not valid (OTV tried to deorbit a debris already deorbited), therefore the visualization makes no sense. Try again."
            )
            return

        print("Rendering in progress...")
        df = pd.DataFrame([])
        first_debris = self.deorbited_debris[0]
        self.simulator.otv_orbit = copy.copy(
            self.simulator.debris_list[first_debris].poliastro_orbit
        )
        self.simulator.current_fuel = self.dv_max_per_mission
        for debris in self.deorbited_debris[1:]:
            transfer_frames = self.simulator.simulate_action(
                action=debris, render=True, step_sec=step_sec
            )
            df = pd.concat([df, transfer_frames], axis=0).reset_index(drop=True)

        render_process = start_render_engine_in_subprocess(
            df, ADREnv.metadata["render_fps"]
        )
        render_process.join()

    def close(self):
        pass


def run_render_engine(df, fps: int):
    from gym_adr.rendering.rendering import RenderEngine

    renderEngine = RenderEngine(df, fps)
    renderEngine.run()


def start_render_engine_in_subprocess(df, fps: int):
    process = multiprocessing.Process(target=run_render_engine, args=(df, fps))
    process.start()
    return process
