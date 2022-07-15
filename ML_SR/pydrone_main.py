import numpy as np
import gym
from ML_SR.envs.droneSim import DroneSim

# TODO we need time limit wrapper probably

episodes = 10
timesteps = 5
env = DroneSim()
done = 0
W = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # state waypoints (1,1,1) is where we are at and we want to go to (2,1,1)
              [0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])

for epis in range(episodes):
    env.reset()
    for t in range(timesteps):  # while (not done):
        action = W[t][[1, 3, 5]] # x,y,z
        s, r, done = env.step(action)
        if done:
            break
