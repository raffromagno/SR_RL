import numpy as np
import gym
from gym import spaces
from ML_SR.envs.world import World as sim


# from gym.envs.mujoco import MuJocoPyEnv # in case we want to build mujoco gym environment

class DroneSim(gym.Env):
    """

    """

    def __init__(self, stateNoise=True):
        """

        :param stateNoise: True/False. Adds noise on the true state
        """
        self.stateNoise = stateNoise
        self.sim = sim
        self.dt = self.sim.ts

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

    def reset(self):
        self.sim.reset_sim()

    def step(self, action):
        obs_ = self.sim.do_sim(action)
        done = not (dist <= 0.2)
        return obs_, reward, done, {}
