import gymnasium as gym
import numpy as np
import random
from typing import Optional

from astropy import units as u

from gym_adr.envs.space_physics.simulator import Simulator


DEBUG = True


class ADREnv(gym.Env):
    """
    ## Description
    
    Active Debris Removal environment.
    
    The goal of the agent is to ...
    
    ## Action Space
    
    
    ## Observation Space
    
    
    ## Rewards
    
    
    ## Success Criteria
    
    
    ## Starting State
    
    
    ## Episode Termination
    
    
    ## Arguments
    
    
    ## Reset Arguments
    
    
    ## Version History
    
    
    ## References
    
    
    """
    def __init__(self,
                 total_n_debris: int = 10,
                 dv_max_per_mission: int = 5,
                 dt_max_per_mission: int = 100,
                 dt_max_per_transfer: int = 30,
                 priority_is_on: bool = True,
                 time_based_action: bool = False,
                 random_first_debris: bool = True,
                 first_debris: Optional[int] = 3
                 ):
        super().__init__()

        # environment variables
        self.total_n_debris = total_n_debris
        self.dv_max_per_mission = dv_max_per_mission # [km/s]
        self.dt_max_per_mission = dt_max_per_mission # [day]
        self.dt_max_per_transfer = dt_max_per_transfer # [day]    
        self.priority_is_on = priority_is_on
        self.time_based_action = time_based_action
        self.random_first_debris = random_first_debris
        self.first_debris = first_debris

        ###########################
        self.fuel_uses_in_episode = [] # to log the fuel use, why is it used ?
        self.time_uses_in_episode = []
        ###########################

        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris-1)
        self.simulator = Simulator(starting_index=self.first_debris , n_debris=self.total_n_debris)

        self.observation_space = self._initialize_observation_space()
        self.action_space = gym.spaces.Discrete(self.total_n_debris) # debris to deorbit (dt_max_per_transfer is not in action space anymore)

        self.action_is_legal = False # check when it is used

    def step(self, action):
        if DEBUG:
            print("\n -----  ENV STEP -----")
            print('action: ', action)

        if self.state.binary_flags[action[0]] == 1:
            print('illegal binary flag') if DEBUG else None
            observation = 0 # self.state.to_list() # think about how state should be implemented, State class ? 
            reward = 0
            terminated = True
            truncated = False
            info = {} # check what to put in info
            return observation, reward, terminated, truncated, info

        # Use the simulator to compute the maneuvre fuel and time and propagate
        cv, dt_min = self.simulator.simulate_action(action)

        self.action_is_legal = self.is_legal(action, cv, dt_min)
        terminated = not self.action_is_legal
        truncated = False # check when True or False

        reward = self.compute_reward(action)

        # reset priority list after computing reward

        # update state somewhere 

        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random # check how np_random works
        super().reset(seed=seed)
        self._setup()

        # ########
        # it's here that we initialize our state (with values coming from the Iridium-33)
        # from now random only
        rs = np.random.RandomState(seed=seed) # si init des débris random
        state = np.array([rs.randint(0,100),
                          rs.randint(0,100)]) # to do
        # ########
        self._set_state(state)
    
        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def _initialize_observation_space(self):
        self.observation_space = gym.spaces.Dict(
                {
                    # [removal_step, number_debris_left, current_removing_debris]
                    "step_and_debris": gym.spaces.MultiDiscrete(
                        np.array([self.total_n_debris, self.total_n_debris, self.total_n_debris])
                    ),
                    # [dv_left, dt_left]
                    "fuel_time_constraints": gym.spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.dv_max_per_mission, self.dt_max_per_mission]),
                        dtype=np.float64,
                    ),
                    # [binary_flag_debris1, binary_flag_debris2...]
                    "binary_flag": gym.spaces.MultiBinary(
                        self.total_n_debris
                    ),
                    # [priority_score_debris1, priority_score_debris2...]
                    "priority_score": gym.spaces.MultiBinary(
                        self.total_n_debris,
                    ),
                }
            )

    def get_obs(self):
        return {
            'step_and_debris': np.array()
        }
        pass

    def _set_state(self, state):
        self.removal_step = state[0]
        self.number_debris_left = state[1]
        self.current_removing_debris = state[2]
        self.dv_left = state[3]
        self.dt_left = state[4]
        self.binary_flag = state[4:self.total_n_debris+4]
        self.priority_score = state[self.total_n_debris+4:2*self.total_n_debris+4]

        # Run physics to take effect
        # self.space.step(self.dt) # to do

    def get_info(self):
        info = {}
        if not self.action_is_legal:
            info['is_legal'] = False
        
        return info

    def compute_reward(self, action):
        # Calculate reward using the priority list
        reward = self.state.priority_list[action]

        # Set reward to 0 if the action is not legal
        if not self.action_is_legal:
            reward = 0
        
        return reward

    def _setup(self):
        pass

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        # do we use this function for rendering or deal with it somewhere else ?
        pass

    def close(self):
        pass
