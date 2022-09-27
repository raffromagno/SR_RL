from itertools import count
from tracemalloc import start
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt
# import wandb

from envs.droneSim import DroneSim
from envs.world import World

device = 'cpu'

# WandB
# wandb.init(project="DeepRL", entity="sangela")

def plot3DTrajectory(xp,yp,zp):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = zp
    xline = xp
    yline = yp
    ax.plot3D(xline, yline, zline, 'tab:blue', lw=2.5, label='Actual Drone Trajectory')
    plt.title('Drone 3D Trajectory', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train_one_episode(env, model, optimizer, start=[1,1,1], goal=[-1,-1,-1], epochs=5, batch_size=32, render=False, scale=1.0, gamma=0.8):
    
    # make function to compute action distribution
    def get_policy(obs):
        model_output = model(obs.to(device))
        return Normal(model_output,torch.full_like(model_output,scale).to(device))
    
    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().tolist()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        # [in] obs: (batch,12)
        # [in] act: (batch, 3)
        obs = obs.to(device)
        act = act.to(device)
        weights = weights.to(device)
        logp = get_policy(obs).log_prob(act)
        new_weights = weights.repeat(3,1).T
        return -(logp*new_weights).mean()
    
    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset(start)       # first obs comes from starting distribution
        env.goal = goal
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            print("obs={}".format(obs[[1,3,5]]))
            obs, rew, done, info = env.step(act)
            print("act={} rew={} obs={}".format(act,rew,obs[[1,3,5]]))
            print("true obs={}".format(env.sim.x0[[1,3,5]]))

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done or len(batch_obs) >= batch_size:
                print("End Episode with done={}".format(done))
                
                # if episode is over, record info about episode
                # ep_ret, ep_len = sum(ep_rews), len(ep_rews)

                
                ep_ret = (torch.as_tensor(np.array(ep_rews)).to(device) * (gamma**(torch.arange(len(batch_obs))).to(device))).sum().item()
                ep_len = len(ep_rews)

                print("ep_ret={}".format(ep_ret))

                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # r = torch.as_tensor(ep_rews) * (gamma**(torch.arange(t)).to(device))
                # batch_weights += r
                

                # # reset episode-specific variables
                # obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if (len(batch_obs) >= batch_size):
                    break
        
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    
    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

## Main ###
if __name__ == '__main__':
    # Define env
    W = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # state waypoints (1,1,1) is where we are at and we want to go to (2,1,1)
              [0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
    world = World()  # <-- we need the parenthesis!!!
    env = DroneSim(world, goal=W[-1][[1, 3, 5]])    
    episodes = W.shape[0]

    # Model Parameters
    hidden_sizes=[32]
    lr=1e-2

    # make core of policy network
    model = mlp(sizes=[12]+hidden_sizes+[3]).to(device)   # state space (12 x 1), action space (3 x 1) 

    # make optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    xx_tot = []

    for epis in range(1):
    # for epis in range(episodes):
        # Randomize start and goal
        epis = 1
        init = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1.5, 1.5, 1.5])    # Random generate start and goal
        init[:-3] = W[epis]
        init[[1,3,5]] = [0,0,0]
        obs = env.reset(start=init[[1,3,5]])
        env.goal = init[-3:]

        x0 = np.zeros(12)
        x0[[1,3,5]] = init[[1,3,5]]

        
        print("Start:", init[[1, 3, 5]])
        print("End:", init[-3:])

        print("Episode: {}".format(epis))
        train_one_episode(env,model,optimizer,start=init[[1,3,5]],goal=init[-3:])

    print('DONE')
#     X = np.stack(xx_tot, axis=0)
#     xp = X[:, 1]
#     yp = X[:, 3]
#     zp = X[:, 5]
#     plot3DTrajectory(xp,yp,zp)