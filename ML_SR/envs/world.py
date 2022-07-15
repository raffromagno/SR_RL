import numpy as np
import scipy.io as sci


class World():
    def __init__(self):
        self.ts = 0.002
        self.M = 0.6  # mass (Kg)
        self.g = 9.81  # acceleration due to gravity m/s^2
        self.ff = self.M * self.g
        self.K = sci.loadmat('/Users/constantinos/Documents/Projects/SR_RL/ML_SR/control.mat')['K']
        self.L = 0.2159 / 2  # arm length (m)
        self.m = 0.410  # Sphere Mass (Kg)
        self.Radius = 0.0503513  # Radius Sphere (m)

        self.m_prop = 0.00311  # propeller mass (Kg)
        self.m_m = 0.036 + self.m_prop  # motor +  propeller mass (Kg)

        self.Ad = np.array([[1, 0],
                            [self.ts, 1]])  # 2x2
        self.Bd = np.array([[0, self.ts],
                            [(1 / 2) * (self.ts ** 2), 0]])
        self.A0 = np.zeros_like(self.Ad)
        self.B0 = np.zeros_like(self.Bd)
        self.At = np.asarray(np.bmat([[self.Ad, self.A0, self.A0],
                                      [self.A0, self.Ad, self.A0],
                                      [self.A0, self.A0, self.Ad]]))
        # bmat: nice way consistent with matlab syntax: https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array

        self.Bp = np.asarray(np.bmat([[self.Bd, self.B0, self.B0],
                                      [self.B0, self.Bd, self.B0],
                                      [self.B0, self.B0, self.Bd]]))
        self.Bup = np.array([0, 0, 0, 0, 0, self.ts])
        self.Bua = np.zeros_like(self.Bup)

        self.Cp = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1]])

        self.Ca = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
        # xdot x ydot y zdot z θdot θ φdot φ ψdot ψ, goal position x, y, z
        self.x_sp = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])  # starting WAYPOINT (state in which the mission starts)
        self.x0 = np.zeros_like(self.x_sp)  # init state of the drone

        # measurement noise
        self.R = 1e-5 * np.eye(3)  # %process noise variance
        self.Dv = 3 * 1e-4

        # process noise
        self.sigma_a2 = 1e-4  # acceleration variance

        self.Q = np.sqrt(self.sigma_a2) * np.array([[self.ts ** 4 / 4, self.ts ** 3 / 2],
                                                    [self.ts ** 3 / 2, self.ts ** 2]])  # process noise convariance matrix

        self.Qt = np.asarray(np.bmat([[self.Q, self.A0, self.A0],
                                      [self.A0, self.Q, self.A0],
                                      [self.A0, self.A0, self.Q]]))

        # Initial Condition and Initial Covariance matrix
        self.P_p = 0.001 * np.eye(6)  # %initial covariance matrix for the position (6 first coords)
        self.hat_xp = np.zeros(6)  # initial guess of the initial position
        self.P_a = 0.01 * self.P_p  # cov matrix for the angle (6 last coords)
        self.hat_xa = np.zeros(6)  # initial guess of the initial angle
        self.hat_x0 = [np.concatenate([self.hat_xp, self.hat_xa])] # initial guess of the initial state

    def qds_dt(self, x, U):
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

        Jx = (2 * self.m * self.Radius) / 5 + 2 * (self.L ** 2) * self.m_m
        Jy = (2 * self.m * self.Radius) / 5 + 2 * (self.L ** 2) * self.m_m
        Jz = (2 * self.m * self.Radius) / 5 + 4 * (self.L ** 2) * self.m_m

        # J = np.array([[Jx, 0, 0],
        #               [0, Jy, 0],
        #               [0, 0, Jz]
        #               ])
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

        du = r * v - q * w - self.g * sin(theta)
        dv = p * w - r * u + self.g * cos(theta) * sin(phi)
        dw = q * u - p * v + self.g * cos(theta) * cos(phi) - (1 / self.M) * U[0]

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

    def rec_KF(self, x_k, P_k, y_k, Q, R, Ad, Bw, Cd, Dv, Bu, u):
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

    def reset_sim(self):
        # Loop for starting at a specific point or just return the init state
        return self.hat_x0

    def do_sim(self, action):
        x, y, z = action
        x_sp = np.array([0, x, 0, y, 0, z, 0, 0, 0, 0, 0, 0])
        cu = -K @ (hat_x0 - x_sp)
        self.x0 = self.x0 + self.qds_dt(self.x0, cu) * self.ts
