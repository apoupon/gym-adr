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

        #######
        self.fuel_uses_in_episode = [] # to log the fuel use
        self.time_uses_in_episode = []
        #######

        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris-1)
        self.simulator = Simulator(starting_index=self.first_debris , n_debris=self.total_n_debris)

        self.observation_space = self._initialize_observation_space()
        self.action_space = gym.spaces.Discrete(self.total_n_debris)
        
        self.priority_list = np.ones(self.total_n_debris, dtype=int)
        
    def step(self, action):
        if DEBUG:
            print("\n -----  ENV STEP -----")

        # Don't do the propagation if the action terminates the episode by binary flags (case not handled by the simulator)
        if self.binary_flags[action] == 1:
            observation = self.get_obs()
            reward = 0
            terminated = True
            info = {}
            return observation, reward, terminated, False, info

        # Use the simulator to compute the maneuvre fuel and time and propagate
        cv, dt_min = self.simulator.simulate_action(action)

        terminated = self.is_terminated(action, cv, dt_min)

        reward = self.compute_reward(action, terminated)

        self.transition_function(action=action,
                                 cv=cv,
                                 dt_min=dt_min)

        # reset priority list after computing reward
        self.priority_list = np.ones(self.total_n_debris, dtype=int)

        # Modify priority list if there is a priority debris (high risk of collision)
        priority_debris = self.get_priority()
        if priority_debris and self.priority_is_on:
            self.priority_list[priority_debris] = 10

        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        print("\n -----  RESET ENV -----")

        super().reset(seed=seed)
        self._setup() # à voir quand on s'occupe du rendering

        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris-1)
        self.simulator.__init__(starting_index=self.first_debris , n_debris=self.total_n_debris)

        # initialize state
        state = np.concatenate([
            np.array([0, self.total_n_debris, self.first_debris, self.dv_max_per_mission, self.dt_max_per_mission]),
            np.zeros(self.total_n_debris, dtype=int),
            np.zeros(self.total_n_debris, dtype=int),
        ])
        state[5+self.first_debris] = 1
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
                    "binary_flags": gym.spaces.MultiBinary(
                        self.total_n_debris
                    ),
                    # [priority_score_debris1, priority_score_debris2...]
                    "priority_scores": gym.spaces.MultiBinary(
                        self.total_n_debris,
                    ),
                }
            )

    def get_obs(self):
        return {
            'step_and_debris': np.array([self.removal_step, self.number_debris_left, self.current_removing_debris]),
            'fuel_time_constraints': np.array([self.dv_left, self.dt_left]),
            'binary_flags': np.array(self.binary_flags),
            'priority_scores': np.array(self.priority_scores)
        }

    def _set_state(self, state):
        # check every value
        self.removal_step = state[0]
        self.number_debris_left = state[1]
        self.current_removing_debris = state[2]
        self.dv_left = state[3]
        self.dt_left = state[4]
        self.binary_flags = state[5:self.total_n_debris+5]
        self.priority_scores = state[self.total_n_debris+6:2*self.total_n_debris+6]

        # render first frame


    def get_info(self):
        info = {}
        
        # store dataframe in info ? (rendering)
        
        return info

    def compute_reward(self, action, terminated):
        # Calculate reward using the priority list
        reward = self.priority_list[action]

        # Set reward to 0 if the action is not legal
        if terminated:
            reward = 0
        
        return reward

    def is_terminated(self,
                      action,
                      cv,
                      dt_min):
        # input is state before transition
        next_debris_index = action

        # 1st check: do we have enough time to go to the next debris ?
        # 2nd check: do we have enough fuel to go to the next debris ?
        # 3rd check: is the next debris still in orbit or not anymore ?
        if (self.dt_left*u.day - dt_min) < 0 or \
            (self.dv_left * (u.km/u.s) - cv) < 0 or \
            self.binary_flags[next_debris_index] == 1:
            return True
        
        return False

    def transition_function(self,
                            action,
                            cv,
                            dt_min
                            ):
        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= dt_min.to(u.day).value
        self.dv_left -= cv.to(u.km/u.s).value

        # Update current removing debris after computing CB
        self.current_removing_debris = action
        self.binary_flags[self.current_removing_debris] = 1

    def get_priority(self):
        '''
            Returns a random debris index to set as priority
            Taken from the available debris that have not been removed yet
        '''
        priority_debris = None

        # Get the list of indices where the binary flag is 0
        available_debris = [i for i, flag in enumerate(self.binary_flags) if flag == 0]

        if random.random() < 0.3:
            # Randomly select a debris from the available list
            priority_debris = random.choice(available_debris)

        return priority_debris

    def _setup(self):
        pass

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        # do we use this function for rendering or deal with it somewhere else ?
        pass

    def close(self):
        pass
