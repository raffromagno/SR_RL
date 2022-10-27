import numpy as np
import scipy.io as sci


class World(object):
    # static variables shared by all instances.
    ts = 0.002
    # ts = ts
    M = 0.6  # mass (Kg)
    g = 9.81  # acceleration due to gravity m/s^2
    ff = M * g
    K = sci.loadmat('/Users/sangela/Desktop/DeepRL/SR_RL/ML_SR/control.mat')['K']
    L = 0.2159 / 2  # arm length (m)
    m = 0.410  # Sphere Mass (Kg)
    Radius = 0.0503513  # Radius Sphere (m)

    m_prop = 0.00311  # propeller mass (Kg)
    m_m = 0.036 + m_prop  # motor +  propeller mass (Kg)

    Ad = np.array([[1, 0],
                   [ts, 1]])  # 2x2
    Bd = np.array([[0, ts],
                   [(1 / 2) * (ts ** 2), 0]])
    A0 = np.zeros_like(Ad)
    B0 = np.zeros_like(Bd)
    At = np.asarray(np.bmat([[Ad, A0, A0],
                             [A0, Ad, A0],
                             [A0, A0, Ad]]))
    # bmat: nice way consistent with matlab syntax: https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array

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

    # measurement noise
    R = 1e-5 * np.eye(3)  # %process noise variance
    Dv = 3 * 1e-4

    # process noise
    sigma_a2 = 1e-4  # acceleration variance

    def __init__(self):
        # dynamic variables that belong to an instance of the class.
        # xdot x ydot y zdot z θdot θ φdot φ ψdot ψ, goal position x, y, z
        self.init_x_sp = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])  # starting WAYPOINT (state in which the mission starts)
        self.init_x0 = np.zeros_like(self.init_x_sp)  # init state of the drone

        self.Q = np.sqrt(self.sigma_a2) * np.array([[self.ts ** 4 / 4, self.ts ** 3 / 2],
                                                    [self.ts ** 3 / 2, self.ts ** 2]])  # process noise convariance matrix

        self.Qt = np.asarray(np.bmat([[self.Q, self.A0, self.A0],
                                      [self.A0, self.Q, self.A0],
                                      [self.A0, self.A0, self.Q]]))

        # Initial Condition and Initial Covariance matrix
        self.P_p = 0.001 * np.eye(6)  # %initial covariance matrix for the position (6 first coords)
        self.init_hat_xp = np.zeros(6)  # initial guess of the initial position
        self.P_a = 0.01 * self.P_p  # cov matrix for the angle (6 last coords)
        self.init_hat_xa = np.zeros(6)  # initial guess of the initial angle
        self.init_hat_x0 = np.concatenate([self.init_hat_xp, self.init_hat_xa])  # initial guess of the initial state
        # Hack
        self.hat_xp = self.init_hat_xp
        self.hat_xa = self.init_hat_xa
        self.hat_x0 = self.init_hat_x0
        self.x_sp = self.init_x_sp
        self.x0 = self.init_x0
        # Trajectory Plot
        self.collect_traj = []

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

    def reset_sim(self, start=[1,1,1]):
        # TO DO: random init
        x0 = np.zeros(12)
        x0[[1,3,5]] = start
        # Loop for starting at a specific point or just return the init state
        self.hat_xp = self.init_hat_xp
        self.hat_xa = self.init_hat_xa
        self.hat_x0 = self.init_hat_x0
        self.x_sp = x0
        self.x0 = x0

    def do_sim(self, action):
        # for time in range(6000):  # 20000 is the original, and you get very close to the target points.
        for time in range(3000):  # 20000 is the original, and you get very close to the target points.
            yp = self.Cp @ self.x0[:6] + np.sqrt(self.R) @ np.random.randn(3)  # position and velocity + noise. x0 vector is the ground truth
            ya = self.Ca @ self.x0[6:] + np.sqrt(self.R) @ np.random.randn(3)  # angles and ang. velocities + noise.

            # self.hat_xp, self.P_p = self.rec_KF(self.hat_xp, self.P_p, yp, self.Qt, self.R, self.At, self.Bp, self.Cp, self.Dv, self.Bup, 0)  # position and velocity estimation
            # self.hat_xa, self.P_a = self.rec_KF(self.hat_xa, self.P_a, ya, self.Qt, self.R, self.At, self.Bp, self.Ca, self.Dv, self.Bup, 0)  # orientation and angular velocity estimation

            # self.hat_x0 = np.concatenate([self.hat_xp, self.hat_xa])  # 12 element vector with the new state drone
            x, y, z = action
            self.x_sp = np.array([0, x, 0, y, 0, z, 0, 0, 0, 0, 0, 0])
            # cu = -self.K @ (self.hat_x0 - self.x_sp)
            cu = -self.K @ (self.x0 - self.x_sp)
            cu[0] = cu[0] + 0.6*9.81
            
            ## TO DO: constraints on cu
            # cu[1,2,3]: +/-0.432
            # cu[0]: 0 - 16
            epsilon = 1e-15
            min_bound = np.array([0,-0.0432,-0.0432,-0.0432])+epsilon
            max_bound = np.array([16,0.0432,0.0432,0.0432])-epsilon

            cu = np.clip(cu,min_bound,max_bound)

            self.x0 = self.x0 + self.qds_dt(self.x0, cu) * self.ts
            self.cu.append(cu)
            self.theta.append(self.x0[9])
            self.collect_traj.append(self.x0)
        return self.x0 # return the estimated state
