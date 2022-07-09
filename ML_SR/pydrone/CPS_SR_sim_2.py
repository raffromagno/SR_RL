# This script simulates the CPS SR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import scipy.io as sci


# TODO: For replication keep the seed of the random generator the same as in MATLAB
def qds_dt(x, U):
    """
    Quadrotor dynamics

    %% Body state vector

    % u body frame velocity along x-axis
    % v body frame velocity along y-axis
    % w body frame velocity along z-axis

    % p roll rate along x-axis body frame
    % q pitch rate along y-axis body frame
    % r yaw rate along z-axis body frame

    %% Inertial state vector

    % p_n inertial north position (x-axis)
    % p_e inertial east position (y-axis)
    % h   inertial altitude (- z-axis)

    % phi   roll angle respect to vehicle-2
    % theta pitch angle respect to vehicle-1
    % psi   yaw angle respect to vehicle

    %% Generic Quadrotor X PX4

    """
    pi = np.pi
    sin = np.sin
    cos = np.cos
    tan = np.tan

    M = 0.6  # mass (Kg)
    L = 0.2159 / 2  # arm length (m)

    g = 9.81  # acceleration due to gravity m/s^2

    m = 0.410  # Sphere Mass (Kg)
    R = 0.0503513  # Radius Sphere (m)

    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor +  propeller mass (Kg)

    Jx = (2 * m * R) / 5 + 2 * (L ** 2) * m_m
    Jy = (2 * m * R) / 5 + 2 * (L ** 2) * m_m
    Jz = (2 * m * R) / 5 + 4 * (L ** 2) * m_m

    J = np.array([[Jx, 0, 0],
                  [0, Jy, 0],
                  [0, 0, Jz]
                  ])
    phi = x[7]
    theta = x[9]
    psi = x[11]

    # Rotation Matrices
    Rp = np.array([[cos(theta) * cos(psi), sin(phi) * sin(theta) - cos(psi) * sin(psi), cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)],
                   [cos(theta) * sin(psi), sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), cos(phi) * sin(theta) * sin(psi) + sin(phi) * cos(psi)],
                   [sin(theta), -sin(phi) * cos(theta), cos(phi) * cos(theta)]
                   ])

    Rv = np.array([[1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                   [0, cos(phi), -sin(phi)],
                   [0, sin(phi) / cos(theta), cos(phi) / cos(theta)]
                   ])

    vect_p = np.linalg.solve(Rv, np.array([x[6], x[8], x[10]]))

    p = vect_p[0]
    q = vect_p[1]
    r = vect_p[2]

    vect_u = np.linalg.solve(Rp, np.array([x[0], x[2], x[4]]))

    u = vect_u[0]
    v = vect_u[1]
    w = vect_u[2]

    du = r * v - q * w - g * sin(theta)
    dv = p * w - r * u + g * cos(theta) * sin(phi)
    dw = q * u - p * v + g * cos(theta) * cos(phi) - (1 / M) * U[0]

    dp = ((Jy - Jz) / Jx) * q * r + (1 / Jx) * U[1]
    dq = ((Jz - Jx) / Jy) * p * r + (1 / Jy) * U[2]
    dr = ((Jx - Jy) / Jz) * p * q + (1 / Jz) * U[3]

    RpT = np.transpose(Rp)
    dpos = np.array([du, dv, dw])

    vect_x = RpT.dot(dpos)

    RvT = np.transpose(Rv)
    dang = np.array([dp, dq, dr])

    vect_phi = RvT.dot(dang)

    dx = np.zeros(12)
    dx[0] = vect_x[0]
    dx[1] = x[0]
    dx[2] = vect_x[1]
    dx[4] = vect_x[2]

    dx[3] = x[2]
    dx[5] = x[4]
    dx[6] = vect_phi[0]
    dx[7] = x[6]
    dx[8] = vect_phi[1]
    dx[9] = x[8]
    dx[10] = vect_phi[2]
    dx[11] = x[10]

    return dx


def rec_KF(x_k, P_k, y_k, Q, R, Ad, Bw, Cd, Dv, Bu, u):
    """
    This function implements one step of the discrete time Kalman Filter
    :param x_k: state estimation previous step
    :param P_k: state estimation covariance matrix
    :param y_k: actual output measurements
    :param Q, R: covariance matrices process and measurement noises
    :param Ad, Bw, Cd: generic signal model
    :param Dv: seems 1-dim
    :param Bu:
    :param u: seems 0
    :return:
        x_kk : current state estimation
        P_kk : current covariance matrix of the state estimation
    """
    n = Ad.shape[0]
    P = Ad @ P_k @ Ad.T + Bw @ Q @ Bw.T  # P(k+1|k)
    Kk = P @ Cd.T @ np.linalg.inv(Cd @ P @ Cd.T + Dv * R * Dv)  # K(k+1) # Dv seems to be a scalar
    x_kk = Ad @ x_k + Bu * u + Kk @ (y_k - Cd @ (Ad @ x_k + Bu * u))  # x(k+1|k+1)
    P_kk = (np.eye(n) - Kk @ Cd) @ P  # P(k+1|k+1)
    return x_kk, P_kk


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


# K = load('control.mat');
# K = cell2mat(struct2cell(K));
K = sci.loadmat('/Users/constantinos/Documents/Projects/SR_RL/ML_SR/control.mat')['K']  # K:4x12
ts = 0.002
tt = np.arange(0, 40, ts).tolist()  # 10 seconds to reach the initial point

tckp = 0.02  # time needed for a checkpoint
trb = 0.016  # time needed for a rollback

ttck = np.arange(0, tckp, ts).tolist()
ttrb = np.arange(0, trb, ts).tolist()

# Kalman Filter setup

M = 0.6
g = 9.81
ff = M * g

Ad = np.array([[1, 0],
               [ts, 1]])  # 2x2
Bd = np.array([[0, ts],
               [(1 / 2) * (ts ** 2), 0]])
A0 = np.zeros_like(Ad)
B0 = np.zeros_like(Bd)
At = np.asarray(np.bmat([[Ad, A0, A0],
                         [A0, Ad, A0],
                         [A0, A0, Ad]]))  # nice way consistent with matlab syntax: https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array

Bp = np.asarray(np.bmat([[Bd, B0, B0],
                         [B0, Bd, B0],
                         [B0, B0, Bd]]))
Bup = np.array([0, 0, 0, 0, 0, ts])
Bua = np.zeros_like(Bup)

Cp = np.array([[0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1]])

Ca = np.array([[1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0]])
# xdot x ydot y zdot z θdot θ φdot φ ψdot ψ, goal position x, y, z
x_sp = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, ])  # starting WAYPOINT (state in which the mission starts)
x0 = np.zeros_like(x_sp)  # init state of the drone

# measurement noise
R = 1e-5 * np.eye(3)  # %process noise variance
Dv = 3 * 1e-4

# process noise
sigma_a2 = 1e-4  # acceleration variance

Q = np.sqrt(sigma_a2) * np.array([[ts ** 4 / 4, ts ** 3 / 2],
                                  [ts ** 3 / 2, ts ** 2]])  # process noise convariance matrix

Qt = np.asarray(np.bmat([[Q, A0, A0],
                         [A0, Q, A0],
                         [A0, A0, Q]]))

# Initial Condition and Initial Covariance matrix
P_p = 0.001 * np.eye(6)  # %initial covariance matrix for the position (6 first coords)
hat_xp = np.zeros(6)  # initial guess of the initial position
P_a = 0.01 * P_p  # cov matrix for the angle (6 last coords)
hat_xa = np.zeros(6)  # initial guess of the initial angle

# Simulation: Quadrotor takeoff, from zero
# Now we run the simulation to get the drone at x_sp (starting point). And
# from there we will start later the full mission!
Ns = len(tt)
uu = []  # for saving the control
xx_tot = [x0]  # for saving states. Put the init state in
hat_xx_tot = [np.concatenate([hat_xp, hat_xa])]

# start loop from 1 as at t=0 we have stored the init states already
for j in range(1,Ns):  # to validate the dynamics with python use -K*(x0-x_sp), i.e the ground truth state and not the noisy estimation
    yp = Cp @ x0[:6] + np.sqrt(R) @ np.random.randn(3)  # position measurements. x0 is the ground truth
    ya = Ca @ x0[6:] + np.sqrt(R) @ np.random.randn(3)  # angular vel.measurements

    [hat_xp, P_p] = rec_KF(hat_xp, P_p, yp, Qt, R, At, Bp, Cp, Dv, Bup, 0)  # position and velocity estimation
    [hat_xa, P_a] = rec_KF(hat_xa, P_a, ya, Qt, R, At, Bp, Ca, Dv, Bup, 0)  # orientation and angular velocity estimation

    hat_x0 = np.concatenate([hat_xp, hat_xa])  # 12 element vector with the new state drone
    # Compute the control law. K:4x12, Force, τθ, τφ, τψ (torgues).
    # Basically this is a gain in the state variables. K contains some
    # safety conditions for the drone which if we substitute with a NN
    # there might not be satisfied these safety constraints. Instead we
    # want to use the NN to provide the waypoints for a smoother trajectory
    # by keeping the safety.
    cu = -K @ (hat_x0 - x_sp)  # translate the coords to your new estimate. This is not error it is coordinate translation
    cu[0] = cu[0] + ff  # we add feed forward, force of the gravity
    # Simulation of the real system
    x0 = x0 + qds_dt(x0, cu) * ts  # Euler method. this is the env.step (new ground truth state after applying the control action)
    # % Save the control input, true state, and state estimation
    uu.append(cu)
    xx_tot.append(x0)
    hat_xx_tot.append(hat_x0)

print('DONE')
X = np.stack(xx_tot, axis=0)
xp = X[:, 1]
yp = X[:, 3]
zp = X[:, 5]
plt.figure(figsize=(16, 9))
plt.plot(tt, xp, label='$x$', lw=2.5)  # 20001 points for xp, yp, zp but tt has 20K TODO: check
plt.plot(tt, yp, label='$y$', lw=2.5)
plt.plot(tt, zp, c='#e62e00', label='$z$', lw=2.5)
plt.xlabel('time in seconds', fontsize=18)
plt.ylabel('distance in $m$', fontsize=18)
plt.title('Translational coordinates', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.show()

plot3DTrajectory()
