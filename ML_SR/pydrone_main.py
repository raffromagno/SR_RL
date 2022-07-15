import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import gym
from ML_SR.envs.droneSim import DroneSim
from ML_SR.envs.world import World


def plot3DTrajectory():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = zp
    xline = xp
    yline = yp
    ax.plot3D(xline, yline, zline, 'tab:blue', lw=2.5, label='Actual Drone Trajectory')
    # zdata = zSet[::100]
    # xdata = xSet[::100]
    # ydata = ySet[::100]
    # ax.scatter3D(xdata, ydata, zdata, c='tab:red', s = 50, label = 'Waypoints');
    plt.title('Drone 3D Trajectory', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    plt.legend(fontsize=18)
    plt.show()


# TODO we need time limit wrapper probably

W = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # state waypoints (1,1,1) is where we are at and we want to go to (2,1,1)
              [0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
world = World()  # <-- we need the parenthesis!!!
env = DroneSim(world, goal=W[-1][[1, 3, 5]])
env.reset()
episodes = W.shape[0]
timesteps = 5
done = 0

xx_tot = []
# Notes: We will change the env.goal to each of the waypoints. when the drone reaches the waypoint task is done
for epis in range(episodes):
    while not done:  # while (not done):
        action = W[epis][[1, 3, 5]]  # x,y,z
        env.goal = action
        s, r, done, info = env.step(action)
        print('dist:',r, 'done:', done)
        xx_tot.append(s)

print('DONE')
X = np.stack(xx_tot, axis=0)
xp = X[:, 1]
yp = X[:, 3]
zp = X[:, 5]
plot3DTrajectory()
