import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import time

class QUADROTOR(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    def __init__(self):

        # crazyflie: physical parameters for the Crazyflie 2.0

        # This function creates a struct with the basic parameters for the
        # Crazyflie 2.0 quad rotor (without camera, but with about 5 vicon
        # markers)
        # Model assumptions based on physical measurements:
        # motor + mount + vicon marker = mass point of 3g
        # arm length of mass point: 0.046m from center
        # battery pack + main board are combined into cuboid (mass 18g) of
        # dimensions:
        # width  = 0.03m
        # depth  = 0.03m
        # height = 0.012m

        m = 0.030 # weight (in kg) with 5 vicon markers (each is about 0.25g)
        gravity = 9.81 #gravitational constant
        I = [[1.43e-5, 0, 0],
                  [0, 1.43e-5, 0],
                  [0, 0, 2.89e-5]] #inertial tensor in m^2 kg
        L = 0.046 # arm length in m

        self.m = m
        self.g = gravity
        self.I = I
        self.invI=np.linalg.inv(self.I)
        self.arm_length = L

        self.max_angle = 40 * math.pi/180 # you can specify the maximum commanded angle here
        self.max_F = 2.5 * m * self.g # left these untouched from the nano plus
        self.min_F = 0.05 * m * self.g # left these untouched from the nano plus

        # You can add any fields you want in self
        # for example you can add your controller gains by
        # self.k = 0, and they will be passed into controller.

        x_bound = 5
        y_bound = 5
        z_bound = 5
        xdot_bound = 5
        ydot_bound = 5
        zdot_bound = 5
        qW_bound = 5
        qX_bound = 5
        qY_bound = 5
        qZ_bound = 5
        p_bound = 5
        q_bound = 5
        r_bound = 5
        high_s = np.array([
            x_bound*1.5,
            y_bound*1.5,
            z_bound*1.5,
            xdot_bound,
            ydot_bound,
            zdot_bound,
            qW_bound,
            qX_bound,qY_bound,
            qZ_bound,
            p_bound,
            q_bound,
            r_bound,])
        high_a = np.array([
            self.max_F,
            1,
            1,
            1,
        ])
        low_a = np.array([
            self.min_F,
            -1,
            -1,
            -1,
        ])

        self.force_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.observation_space = spaces.Box(-high_s, high_s, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.desire_state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def quadEOM(self,t,s, F, M):


        # QUADEOM Solve quadrotor equation of motion
        # quadEOM calculate the derivative of the state vector
        # INPUTS:
        # t      - 1 x 1, time
        # s      - 13 x 1, state vector = [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
        # F      - 1 x 1, thrust output from controller (only used in simulation)
        # M      - 3 x 1, moments output from controller (only used in simulation)
        # self   -  output from init() and whatever parameters you want to pass in
        #  OUTPUTS:
        # sdot   - 13 x 1, derivative of state vector s


        self.A = [[0.25, 0, -0.5 / self.arm_length],
                  [0.25, 0.5 / self.arm_length, 0],
                  [0.25, 0, 0.5 / self.arm_length],
                  [0.25, -0.5 / self.arm_length, 0]]


        prop_thrusts = np.dot(self.A, [F, M[0], M[1]])

        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, self.max_F / 4), self.min_F / 4)
        B = [[1, 1, 1, 1],
             [0, self.arm_length, 0, -self.arm_length],
             [-self.arm_length, 0, self.arm_length, 0]]

        F = np.dot(B[0],prop_thrusts_clamped)

        M = np.reshape([np.dot(B[1:3],prop_thrusts_clamped)[0],np.dot(B[1:3],prop_thrusts_clamped)[1],M[2]],[3])

        # Assign states
        x = s[0]
        y = s[1]
        z = s[2]
        xdot = s[3]
        ydot = s[4]
        zdot = s[5]
        qW = s[6]
        qX = s[7]
        qY = s[8]
        qZ = s[9]
        p = s[10]
        q = s[11]
        r = s[12]
        quat = [qW,qX,qY,qZ]

        bRw = QuatToRot(quat)
        wRb = bRw.T

        # Acceleration
        accel = 1 / self.m * (np.dot(wRb, [[0],[0], [F]]) - [[0], [0], [self.m * self.g]])
        # Angular velocity
        K_quat = 2 #this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1 - (qW*qW + qX*qX+ qY*qY + qZ*qZ)
        qdot = np.dot(np.multiply( [[0., -p, -q, -r],
                                    [p,  0. ,-r,  q],
                                    [q,  r,  0., -p],
                                    [r, -q,  p,  0.]],-1/2),quat)+np.multiply(K_quat * quaterror,quat)


        # Angular acceleration
        omega = [p,q,r]
        pqrdot   = np.dot(self.invI,(M - np.cross(omega, np.dot(self.I,omega))))

        # Assemble sdot
        sdot = np.zeros([13])
        sdot[0]  = xdot
        sdot[1] = ydot
        sdot[2] = zdot
        sdot[3] = accel[0]
        sdot[4] = accel[1]
        sdot[5] = accel[2]
        sdot[6] = qdot[0]
        sdot[7] = qdot[1]
        sdot[8] = qdot[2]
        sdot[9] = qdot[3]
        sdot[10] = pqrdot[0]
        sdot[11] = pqrdot[1]
        sdot[12] = pqrdot[2]
        return sdot

    def step_old(self, F,M):
        time = np.linspace(0, 0.001, 2)
        s = self.state
        s_ = odeint(self.quadEOM, s, time, args=(F, M), tfirst=True)
        self.state = s_[1]

        return self.state

    def ode_step(self,F,M):
        # print(F,M)
        s_dot = self.quadEOM(0, self.state, F, M)
        # print(s_dot)
        s_ = self.state + 0.005 * s_dot
        self.state = s_
        return self.state

    def step(self,t):
        time = np.linspace(t,t+0.001,2)
        s= self.state
        s_ = odeint(self.equation,s, time,args=(traj,),tfirst=True)
        self.state=s_[1]

        return self.state

    def reset_traj(self,start, yaw):
        s = np.zeros([13])
        phi0 = 0.0
        theta0 = 0.0
        psi0 = yaw
        Rot0 = RPYtoRot_ZXY(phi0, theta0, psi0)
        Rot0=np.array(Rot0)
        Quat0 = RotToQuat(Rot0)
        s[0] = start[0]   # x
        s[1] = start[1]   # y
        s[2] = start[2]   # z
        s[3] = 0          # xdot
        s[4] = 0          # ydot
        s[5] = 0          # zdot
        s[6] = Quat0[0]   # qw
        s[7] = Quat0[1]   # qx
        s[8] = Quat0[2]   # qy
        s[9] = Quat0[3]  # qz
        s[10] = 0         # p
        s[11] = 0         # q
        s[12] = 0         # r
        self.state = s
        self.steps_beyond_done = None
        return self.state

    def reset(self):
        self.state = np.zeros([13])
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        self.state[1] = self.np_random.uniform(low=-2, high=2)
        self.state[3] = self.np_random.uniform(low=2, high=4)
        self.state[6] = 1
        return self.state

    def controller(self,desired_state,x):
        # CONTROLLER quadrotor controller
        # The current states are:
        # pos, vel, euler = [roll;pitch;yaw], qd{qn}.omega
        # The desired states are:
        # pos_des, vel_des, acc_des, yaw_des, yawdot_des
        # Using these current and desired states, you have to compute the desired controls% position controller params

        # position controller params
        Kp = [15,15,30]
        Kd = [12,12,10]

        # attitude controller params
        KpM = np.ones([3])*3000
        KdM = np.ones([3])*300

        # desired_state=[pos,vel,acc,yaw,yawdot]
        # x y z xdot ydot zdot qw qx qy qz p q r
        [pos, vel, euler, omega]=stateToQd(x)
        pos_des,vel_des,acc_des,yaw_des,yawdot_des=desired_state[0],desired_state[1],desired_state[2],desired_state[3],desired_state[4]


        acc_des = acc_des + Kd*(np.subtract(vel_des,vel)) + Kp*(np.subtract(pos_des,pos))
        #
        #  Desired roll, pitch and yaw
        phi_des = 1/self.g * (acc_des[0]*np.sin(yaw_des) - acc_des[1]*np.cos(yaw_des))
        theta_des = 1/self.g * (acc_des[0]*np.cos(yaw_des) + acc_des[1]*np.sin(yaw_des))
        psi_des = yaw_des
        #
        euler_des = [phi_des,theta_des,psi_des]
        pqr_des = [0,0,yawdot_des]

        # Thurst
        # qd{qn}.acc_des(3);
        F  = self.m*(self.g + acc_des[2])
        # Moment

        M_=np.multiply(KdM,(np.subtract(pqr_des,omega))) + np.multiply(KpM,(np.subtract(euler_des,euler)))
        M =  np.dot(self.I,M_)
        return F,M

    def equation(self,t,x,traj):
        desired_state = trajectory(traj,t)
        F, M = self.controller(desired_state,x)
        s_dot =self.quadEOM(t,x,F,M)
        return s_dot



def stateToQd(x):
    # Converts qd struct used in hardware to x vector used in simulation
    # x is 1 x 13 vector of state variables [pos vel quat omega]
    # qd is a struct including the fields pos, vel, euler, and omega

    # current state
    pos = x[0:3]
    vel = x[3:6]
    Rot = QuatToRot(x[6:10].T)
    [phi, theta, yaw] = RotToRPY_ZXY(Rot)

    euler = [phi,theta,yaw]
    omega = x[10:13]
    return [pos.tolist(),vel.tolist(),euler,omega.tolist()]
def QuatToRot(q):
    # QuatToRot Converts a Quaternion to Rotation matrix
    # normalize q

    q = q/np.sqrt(sum(np.multiply(q,q)))
    qahat=np.zeros([3,3])
    qahat[0, 1] = -q[3]
    qahat[0, 2] = q[2]
    qahat[1, 2] = -q[1]
    qahat[1, 0] = q[3]
    qahat[2, 0] = -q[2]
    qahat[2, 1] = q[1]
    R = np.eye(3) + 2*np.dot(qahat,qahat) + 2*np.dot(q[0],qahat)
    return R
def RPYtoRot_ZXY(phi,theta,psi):

    R = [[math.cos(psi) * math.cos(theta) - math.sin(phi) * math.sin(psi) * math.sin(theta),
          math.cos(theta) * math.sin(psi) + math.cos(psi) * math.sin(phi) * math.sin(theta),
         - math.cos(phi) * math.sin(theta)],
         [- math.cos(phi) * math.sin(psi),
          math.cos(phi) * math.cos(psi),
          math.sin(phi)],
         [math.cos(psi) * math.sin(theta) + math.cos(theta) * math.sin(phi) * math.sin(psi),
          math.sin(psi) * math.sin(theta) - math.cos(psi) * math.cos(theta) * math.sin(phi),
          math.cos(phi) * math.cos(theta)]]
    return R
def RotToRPY_ZXY(R):
    # RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
    # The rotation matrix in this function is world to body [bRw] you will
    # need to transpose the matrix if you have a body to world [wRb] such
    # that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and
    # [wP] is a point in the world frame
    # bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
    #           cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
    #          -cos(phi)*sin(theta)]
    #         [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
    #         [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
    #            sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
    #            cos(phi)*cos(theta)]

    phi = math.asin(R[1, 2])
    psi = math.atan2(-R[1, 0] / math.cos(phi), R[1, 1] / math.cos(phi))
    theta = math.atan2(-R[0, 2] / math.cos(phi), R[2, 2] / math.cos(phi))
    return [phi,theta,psi]
def RotToQuat(R):
    # ROTTOQUAT Converts a Rotation matrix into a Quaternion
    # takes in W_R_B rotation matrix
    tr = R[0,0] + R[1,1] + R[2,2]

    if (tr > 0):
      S = np.sqrt(tr+1.0) * 2 # S=4*qw
      qw = 0.25 * S
      qx = (R[2,1] - R[1,2]) / S
      qy = (R[0,2] - R[2,0]) / S
      qz = (R[1,0] - R[0,1]) / S

    elif ((R[0,0] > R[1,1]) and (R[0,0] > R[2,2])):
      S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2#  S=4*qx
      qw = (R[2,1] - R[1,2]) / S
      qx = 0.25 * S
      qy = (R[0,1] + R[1,0]) / S
      qz = (R[0,2] + R[2,0]) / S

    elif (R[1,1] > R[2,2]):
      S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2#  S=4*qy
      qw = (R[0,2] - R[2,0]) / S
      qx = (R[0,1] + R[1,0]) / S
      qy = 0.25 * S
      qz = (R[1,2] + R[2,1]) / S
    else:
      S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2#  S=4*qz
      qw = (R[1,0] - R[0,1]) / S
      qx = (R[0,2] + R[2,0]) / S
      qy = (R[1,2] + R[2,1]) / S
      qz = 0.25 * S

    q = [qw,qx,qy,qz]
    q = np.multiply(q,np.sign(qw))
    return q
# Output desired_state
def trajectory(name,t):
    if name == 'circle':
        time_tol = 12
        radius = 5
        dt = 0.0001
        def tj_from_line(start_pos, end_pos, time_ttl, t_c):
            v_max = (end_pos - start_pos) * 2 / time_ttl
            if t_c >= 0 and t_c < time_ttl / 2:
                vel = v_max * t_c / (time_ttl / 2)
                pos = start_pos + t_c * vel / 2
                acc = [0, 0, 0]
            else:
                vel = v_max * (time_ttl - t_c) / (time_ttl / 2)
                pos = end_pos - (time_ttl - t_c) * vel / 2
                acc = [0, 0, 0]
            return [pos, vel, acc[0], acc[1], acc[2]]

        def pos_from_angle(a):
            # pos = [radius*np.cos(a), radius*np.sin(a), 2.5*a/(2*np.pi)]
            pos = [np.multiply(radius, np.cos(a)), np.multiply(radius, np.sin(a)), np.multiply(2.5 / (2 * np.pi), a)]
            return pos

        def get_vel(t):
            angle1 = tj_from_line(0, 2 * np.pi, time_tol, t)[0]
            pos1 = pos_from_angle(angle1)
            angle2 = tj_from_line(0, 2 * np.pi, time_tol, t + dt)[0]
            pos2 = pos_from_angle(angle2)
            vel = (np.subtract(pos2, pos1)) / dt
            return vel

        if t > time_tol:
            pos = [radius, 0, 2.5]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
        else:
            angle = tj_from_line(0, 2 * np.pi, time_tol, t)[0]
            pos = pos_from_angle(angle)
            vel = get_vel(t).tolist()
            acc = ((get_vel(t + dt) - get_vel(t)) / dt).tolist()
        yaw = 0
        yawdot = 0

        desired_state = [pos, vel, acc, yaw, yawdot]
    elif name=='hover':
        time_tol = 1000
        length = 5
        if t <= 0:
            pos = [0, 0, 0]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
        else:
            pos = [1, 0, 0]
            vel = [0, 0, 0]
            acc = [0, 0, 0]

        yaw = 0
        yawdot = 0

        desired_state = [pos, vel, acc, yaw, yawdot]
    return desired_state

env=QUADROTOR()
traj='circle'
des_start = trajectory(traj,0)
des_stop = trajectory(traj,100)
state=env.reset_traj(des_start[0],0)
# print(state)
# state=env.reset()
# print(state)

t=0
cstep     = 0.001
fig = plt.figure()
ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
plot_x=[]
plot_y=[]
plot_z=[]
d_x=[]
d_y=[]
d_z=[]
t1 = time.time()
# x y z xdot ydot zdot qw qx qy qz p q r
# for step in range(1500):
#     desired_state = trajectory(traj, t)
#     s_=env.step(t,traj)
#     t=t+0.01
#     print(step,s_[0],s_[1],s_[2])
#     plot_x.append(s_[0])
#     plot_y.append(s_[1])
#     plot_z.append(s_[2])
#     d_x.append(desired_state[0][0])
#     d_y.append(desired_state[0][1])
#     d_z.append(desired_state[0][2])
#     # ax.scatter(plot_x, plot_y, plot_z, c='r')  # 绘制数据点,颜色是红色
#     # ax.set_zlabel('Z')  # 坐标轴
#     # ax.set_ylabel('Y')
#     # ax.set_xlabel('X')
#     # plt.draw()
#     # plt.pause(0.000001)
# ax.scatter(plot_x, plot_y, plot_z, c='r')  # 绘制数据点,颜色是红色
# ax.scatter(d_x, d_y, d_z, c='b')
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.draw()
# plt.pause(10000)
# plt.savefig('3D.jpg')
# plt.close()
M1=0
M2=0
M3=0
S0=0
S1=0
S2=0
S3=0
S4=0
S5=0
S6=0
S7=0
S8=0
S9=0
S10=0
S11=0
S12=0
S13=0
for step in range(15000):
    desired_state=trajectory(traj,t)
    F,M=env.controller(desired_state,env.state)
    print(M)
    # print(time.time() - t1)
    s_=env.ode_step(F,M)
    # print(time.time() - t1)
    t=t+0.005
    if abs(M[0])>M1:
        M1=abs(M[0])
    if abs(M[1])>M2:
        M2=abs(M[1])
    if abs(M[2])>M3:
        M3=abs(M[2])
    # print(M1,M2,M3)
    # #
    # if abs(s_[0])>S0:
    #     S0=abs(s_[0])
    # if abs(s_[1])>S1:
    #     S1=abs(s_[1])
    # if abs(s_[2])>S2:
    #     S2=abs(s_[2])
    # if abs(s_[3])>S3:
    #     S3=abs(s_[3])
    # if abs(s_[4])>S4:
    #     S4=abs(s_[4])
    # if abs(s_[5])>S5:
    #     S5=abs(s_[5])
    # if abs(s_[6])>S6:
    #     S6=abs(s_[6])
    # if abs(s_[7])>S7:
    #     S7=abs(s_[7])
    # if abs(s_[8])>S8:
    #     S8=abs(s_[8])
    # if abs(s_[9])>S9:
    #     S9=abs(s_[9])
    # if abs(s_[10])>S10:
    #     S10=abs(s_[10])
    # if abs(s_[11])>S11:
    #     S11=abs(s_[11])
    # if abs(s_[12])>S12:
    #     S12=abs(s_[12])
    # S=[S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12]
    # print(step,s_[0],s_[1],s_[2])
    # print(S)
    d_x.append(desired_state[0][0])
    d_y.append(desired_state[0][1])
    d_z.append(desired_state[0][2])
    plot_x.append(s_[0])
    plot_y.append(s_[1])
    plot_z.append(s_[2])
    # ax.scatter(plot_x, plot_y, plot_z, c='r')  # 绘制数据点,颜色是红色
    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.draw()
    # plt.pause(0.000001)
ax.scatter(plot_x, plot_y, plot_z, c='r')  # 绘制数据点,颜色是红色
ax.scatter(d_x, d_y, d_z, c='b')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.draw()
plt.pause(10000)
plt.savefig('3D.jpg')
plt.close()
