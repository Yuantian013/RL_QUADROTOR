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
from Q_ENV import QUADROTOR as QUADROTOR
# state range
#[5.010798538734121, 5.012434648416977, 2.5000005805731815, 3.878042657293132, 5.27397004687728, 0.4119249932803836, 1.0000402186578015, 0.17876637191165382, 0.2536754822865587, 0.029890570096486373, 3.1296342050612465, 0.43550437587386814, 0.8480594157867983]
# state change
#[0.01939021328646562, 0.0263698502343864, 0.002059624966401996, 0.027633574744554273, 0.018391469309218422, 0.0008386027898790727, 0.0002948957425122911, 0.007723181017609458, 0.0012656587839915967, 0.0014338519440371227, 2.629213119720032, 0.04027734049984105, 0.09852208242145807]

#####################  trajactory test ODE ####################
# fig = plt.figure()
# ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
# plot_x=[]
# plot_y=[]
# plot_z=[]
# d_x=[]
# d_y=[]
# d_z=[]
# env=QUADROTOR()
# traj='circle'
# des_start = env.trajectory(traj,0)
# state=env.reset_traj(des_start[0],0)
# t=0
# for step in range(15000):
#     desired_state = env.trajectory(traj, t)
#     F,M=env.controller(desired_state,env.state)
#     s_,_,_=env.ode_step(F,M)
#     t=t+0.001
#     # print(step,s_[0],s_[1],s_[2])
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

#####################  trajactory test DISCRETE ####################
fig = plt.figure()
ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
plot_x=[]
plot_y=[]
plot_z=[]
d_x=[]
d_y=[]
d_z=[]
env=QUADROTOR()
traj='circle'
des_start = env.trajectory(traj,0)
state=env.reset_traj(des_start[0],0)
t=0
for step in range(3000):
    desired_state = env.trajectory(traj, t)
    print(desired_state[2])
    F,M=env.controller(desired_state,env.state)
    s_,_,_=env.step(F,M)
    t=t+0.005
    # print(step,s_[0],s_[1],s_[2])
    plot_x.append(s_[0])
    plot_y.append(s_[1])
    plot_z.append(s_[2])
    d_x.append(desired_state[0][0])
    d_y.append(desired_state[0][1])
    d_z.append(desired_state[0][2])
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


#####################  parameters scale analyse ####################
# Fmax=0
# Fmin=10
# M1=0
# M2=0
# M3=0
# S0=0
# S1=0
# S2=0
# S3=0
# S4=0
# S5=0
# S6=0
# S7=0
# S8=0
# S9=0
# S10=0
# S11=0
# S12=0
# S13=0
# env=QUADROTOR()
# traj='circle'
# des_start = env.trajectory(traj,0)
# state=env.reset_traj(des_start[0],0)
# t=0
# def stateToQd(x):
#     # Converts qd struct used in hardware to x vector used in simulation
#     # x is 1 x 13 vector of state variables [pos vel quat omega]
#     # qd is a struct including the fields pos, vel, euler, and omega
#
#     # current state
#     pos = x[0:3]
#     vel = x[3:6]
#     Rot = QuatToRot(x[6:10].T)
#     [phi, theta, yaw] = RotToRPY_ZXY(Rot)
#
#     euler = [phi,theta,yaw]
#     omega = x[10:13]
#     return [pos.tolist(),vel.tolist(),euler,omega.tolist()]
# def QuatToRot(q):
#     # QuatToRot Converts a Quaternion to Rotation matrix
#     # normalize q
#
#     q = q/np.sqrt(sum(np.multiply(q,q)))
#     qahat=np.zeros([3,3])
#     qahat[0, 1] = -q[3]
#     qahat[0, 2] = q[2]
#     qahat[1, 2] = -q[1]
#     qahat[1, 0] = q[3]
#     qahat[2, 0] = -q[2]
#     qahat[2, 1] = q[1]
#     R = np.eye(3) + 2*np.dot(qahat,qahat) + 2*np.dot(q[0],qahat)
#     return R
# def RPYtoRot_ZXY(phi,theta,psi):
#
#     R = [[math.cos(psi) * math.cos(theta) - math.sin(phi) * math.sin(psi) * math.sin(theta),
#           math.cos(theta) * math.sin(psi) + math.cos(psi) * math.sin(phi) * math.sin(theta),
#          - math.cos(phi) * math.sin(theta)],
#          [- math.cos(phi) * math.sin(psi),
#           math.cos(phi) * math.cos(psi),
#           math.sin(phi)],
#          [math.cos(psi) * math.sin(theta) + math.cos(theta) * math.sin(phi) * math.sin(psi),
#           math.sin(psi) * math.sin(theta) - math.cos(psi) * math.cos(theta) * math.sin(phi),
#           math.cos(phi) * math.cos(theta)]]
#     return R
# def RotToRPY_ZXY(R):
#     # RotToRPY_ZXY Extract Roll, Pitch, Yaw from a world-to-body Rotation Matrix
#     # The rotation matrix in this function is world to body [bRw] you will
#     # need to transpose the matrix if you have a body to world [wRb] such
#     # that [wP] = [wRb] * [bP], where [bP] is a point in the body frame and
#     # [wP] is a point in the world frame
#     # bRw = [ cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta),
#     #           cos(theta)*sin(psi) + cos(psi)*sin(phi)*sin(theta),
#     #          -cos(phi)*sin(theta)]
#     #         [-cos(phi)*sin(psi), cos(phi)*cos(psi), sin(phi)]
#     #         [ cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi),
#     #            sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi),
#     #            cos(phi)*cos(theta)]
#
#     phi = math.asin(R[1, 2])
#     psi = math.atan2(-R[1, 0] / math.cos(phi), R[1, 1] / math.cos(phi))
#     theta = math.atan2(-R[0, 2] / math.cos(phi), R[2, 2] / math.cos(phi))
#     return [phi,theta,psi]
# def RotToQuat(R):
#     # ROTTOQUAT Converts a Rotation matrix into a Quaternion
#     # takes in W_R_B rotation matrix
#     tr = R[0,0] + R[1,1] + R[2,2]
#
#     if (tr > 0):
#       S = np.sqrt(tr+1.0) * 2 # S=4*qw
#       qw = 0.25 * S
#       qx = (R[2,1] - R[1,2]) / S
#       qy = (R[0,2] - R[2,0]) / S
#       qz = (R[1,0] - R[0,1]) / S
#
#     elif ((R[0,0] > R[1,1]) and (R[0,0] > R[2,2])):
#       S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2#  S=4*qx
#       qw = (R[2,1] - R[1,2]) / S
#       qx = 0.25 * S
#       qy = (R[0,1] + R[1,0]) / S
#       qz = (R[0,2] + R[2,0]) / S
#
#     elif (R[1,1] > R[2,2]):
#       S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2#  S=4*qy
#       qw = (R[0,2] - R[2,0]) / S
#       qx = (R[0,1] + R[1,0]) / S
#       qy = 0.25 * S
#       qz = (R[1,2] + R[2,1]) / S
#     else:
#       S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2#  S=4*qz
#       qw = (R[1,0] - R[0,1]) / S
#       qx = (R[0,2] + R[2,0]) / S
#       qy = (R[1,2] + R[2,1]) / S
#       qz = 0.25 * S
#
#     q = [qw,qx,qy,qz]
#     q = np.multiply(q,np.sign(qw))
#     return q
#
#
# for step in range(3000):
#     d_s = env.trajectory(traj, t)
#
#     s=stateToQd(env.state)
#     F,M=env.controller(d_s,env.state)
#     s_,_,_=env.step(F,M)
#     print(d_s[2])
#     # if abs(s[2][0]-s_[2][0])>S0:
#     #     S0=abs(s[2][0]-s_[2][0])
#     # if abs(s[2][1]-s_[2][1])>S1:
#     #     S1=abs(s[2][1]-s_[2][1])
#     # if abs(s[2][2]-s_[2][2])>S2:
#     #     S2=abs(s[2][2]-s_[2][2])
#     t=t+0.005
#
#     # print(Fmax,Fmin,M1,M2,M3)
# #     if abs(s[0]-s_[0])>S0:
# #         S0=abs(s[0]-s_[0])
# #     if abs(s[1]-s_[1])>S1:
# #         S1=abs(s[1]-s_[1])
# #     if abs(s[2]-s_[2])>S2:
# #         S2=abs(s[2]-s_[2])
# #     if abs(s[3]-s_[3])>S3:
# #         S3=abs(s[3]-s_[3])
# #     if abs(s[4]-s_[4])>S4:
# #         S4=abs(s[4]-s_[4])
# #     if abs(s[5]-s_[5])>S5:
# #         S5=abs(s[5]-s_[5])
# #     if abs(s[6]-s_[6])>S6:
# #         S6=abs(s[6]-s_[6])
# #     if abs(s[7]-s_[7])>S7:
# #         S7=abs(s[7]-s_[7])
# #     if abs(s[8]-s_[8])>S8:
# #         S8=abs(s[8]-s_[8])
# #     if abs(s[9]-s_[9])>S9:
# #         S9=abs(s[9]-s_[9])
# #     if abs(s[10]-s_[10])>S10:
# #         S10=abs(s[10]-s_[10])
# #     if abs(s[11]-s_[11])>S11:
# #         S11=abs(s[11]-s_[11])
# #     if abs(s[12]-s_[12])>S12:
# #         S12=abs(s[12]-s_[12])
# #     S=[S0,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12]
# #     # print(step,s_[0],s_[1],s_[2])
# # print(S)
