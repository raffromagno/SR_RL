import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')
import gym
from gym import spaces


# from ML_SR.envs.world import World as sim
# from gym.envs.mujoco import MuJocoPyEnv # in case we want to build mujoco gym environment

class DroneSim(gym.Env):
    """

    """

    def __init__(self, world, goal, stateNoise=True, scale=0.1):
        """

        :param stateNoise: True/False. Adds noise on the true state
        """
        self.stateNoise = stateNoise
        self.sim = world
        self.goal = goal  # have it as a final goal
        self.dt = self.sim.ts

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)  # x,y,z
        obs_space_lower_bound = np.array([-np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        obs_space_upper_bound = np.array([np.inf, 5, np.inf, 5, np.inf, 5, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, ])
        self.observation_space = spaces.Box(low=obs_space_lower_bound, high=obs_space_upper_bound, shape=(12,), dtype=np.float64)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        self.scale = scale

    def reward(self):  # reward should be distance from next waypoint as goal might be the init position if desired traj is a circle
        x0 = self.sim.x0[[1, 3, 5]]
        dist2 = self.scale * np.sum(np.square(x0 - self.goal))
        return -dist2

    def reset(self,start):
        self.sim.reset_sim(start=start)
        return self.sim.x0

    def step(self, action):
        obs_ = self.sim.do_sim(action)
        hat_x0 = obs_[[1, 3, 5]]
        r = self.reward()
        done = not (np.abs(r) >= 0.02)
        return obs_, r, done, {}
