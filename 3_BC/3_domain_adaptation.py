# Convert the human demonstrations into robot demonstrations

import os
from io import StringIO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

########################################################################################################################
# ROBOT

data_0 = open('../Babbling/DataFolder/origin_ref/shift_idm_sinusoids_0.txt', 'r').read()
data_1 = open('../Babbling/DataFolder/origin_ref/shift_idm_parabola_0.txt', 'r').read()
data_2 = open('../Babbling/DataFolder/origin_ref/shift_idm_random_0.txt', 'r').read()
data_3 = open('../Babbling/DataFolder/origin_ref/shift_idm_plateau_0.txt', 'r').read()
data_4 = open('../Babbling/DataFolder/origin_ref/shift_idm_circles_0.txt', 'r').read()

file = data_0 + "\n" + data_1 + "\n" + data_2 + "\n" + data_3 + "\n" + data_4
file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
data = StringIO(file_1)
df_robot = pd.read_csv(data, sep=",", names=["count", "P1", "P2", "P3", "P4", "P5", "P6",
                                             "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
all_data_robot = df_robot.loc[:, "pmx":].to_numpy()
zdata_robot = all_data_robot[:, [2, 5]]
ydata_robot = all_data_robot[:, [1, 4]]
xdata_robot = all_data_robot[:, [0, 3]]


# PLOT OF ROBOT BABBLING DATA
# fig_rob = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=1, color='r', label='proximal')
# ax.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=1, color='b', label='distal')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax.set_title('Babbling data of the robot')
# plt.show()


########################################################################################################################
# AUXILIARY VARIABLES AND FUNCTIONS

all_data_original = all_data_robot.copy()

max_robot = np.amax(all_data_robot, axis=0)
min_robot = np.amin(all_data_robot, axis=0)

# Linear regression for proximal segment data inclination
X_lin_reg = xdata_robot[:, 0].reshape((-1, 1))
y_lin_reg = zdata_robot[:, 0].reshape((-1, 1))
reg = LinearRegression().fit(X_lin_reg, y_lin_reg)
m_coeff = reg.coef_[0][0]
m_angle = np.arctan(m_coeff)


########################################################################################################################
# TO SELECT IF I WANT TO ROTATE PROXIMAL SEGMENT
flag_rotate = False


def rotate(coord, alpha):
    x_or = coord[:, 0]
    y_or = coord[:, 1]
    z_or = coord[:, 2]
    x_rot = np.cos(alpha) * x_or + np.sin(alpha) * z_or
    y_rot = y_or
    z_rot = (-1) * np.sin(alpha) * x_or + np.cos(alpha) * z_or
    x_rot = np.reshape(x_rot, (len(x_rot), 1))
    y_rot = np.reshape(y_rot, (len(y_rot), 1))
    z_rot = np.reshape(z_rot, (len(z_rot), 1))
    rot_coord = np.hstack([x_rot, y_rot, z_rot])
    return rot_coord


if flag_rotate:
    all_data_robot[:, :3] = rotate(all_data_robot[:, :3], m_angle)
    zdata_robot = all_data_robot[:, [2, 5]]
    ydata_robot = all_data_robot[:, [1, 4]]
    xdata_robot = all_data_robot[:, [0, 3]]

# PLOT ROTATION PROXIMAL SEGMENT
# fig_rob = plt.figure()
# ax = plt.subplot(121, projection='3d')
# ax.scatter3D(all_data_original[:, 0], all_data_original[:, 1], all_data_original[:, 2], s=1, color='r', label='original')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax = plt.subplot(122, projection='3d')
# ax.scatter3D(all_data_robot[:, 0], all_data_robot[:, 1], all_data_robot[:, 2], s=1, color='b', label='rotated')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# plt.suptitle('Proximal segment coordinates before and after rotation')
# plt.show()

########################################################################################################################

center = np.mean(all_data_robot, axis=0)
aux_std = np.std(all_data_robot, axis=0)

# fig_distr, axs = plt.subplots(3, 2)
# fig_distr.suptitle('Distribution of the robot babbling data')
# for m in range(6):
#     row = m % 3
#     col = int(m/3)
#     axs[row, col].hist(all_data_robot[:, m], bins=500, density=True, alpha=0.6, color='b', label='histogram')
#     mu = center[m]
#     variance = aux_std[m]
#     xmin = min(all_data_robot[:, m])
#     xmax = max(all_data_robot[:, m])
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, variance)
#     axs[row, col].plot(x, p, 'r', linewidth=1, label='gaussian')
# plt.legend()
# axs[0, 0].title.set_text('Proximal segment')
# axs[0, 1].title.set_text('Distal segment')
# axs[0, 0].set(ylabel='X')
# axs[1, 0].set(ylabel='Y')
# axs[2, 0].set(ylabel='Z')
# axs[2, 0].set(xlabel='Value')
# axs[2, 1].set(xlabel='Value')
# plt.show()

confInt_min = center - 2 * aux_std  # 95% confidence interval
confInt_max = center + 2 * aux_std  # 95% confidence interval

# Parameters for the ellipsoids (2 times the std to have the 95% confidence interval)
a = 2 * aux_std[[0, 3]]
a = np.reshape(a, (len(a), 1))
b = 2 * aux_std[[1, 4]]
b = np.reshape(b, (len(b), 1))
c = 2 * aux_std[[2, 5]]
c = np.reshape(c, (len(c), 1))

t = np.arange(0, 2 * np.pi, np.pi / 200)
t = np.reshape(t, (1, len(t)))
theta = np.arange(0, 2 * np.pi, np.pi / 200)
theta = np.reshape(theta, (1, len(theta)))
phi = np.arange(0, np.pi, np.pi / 100)
phi = np.reshape(phi, (1, len(phi)))
x_c = center[[0, 3]]
x_c = np.reshape(x_c, (len(x_c), 1))
y_c = center[[1, 4]]
y_c = np.reshape(y_c, (len(y_c), 1))
z_c = center[[2, 5]]
z_c = np.reshape(z_c, (len(z_c), 1))
z_pro = z_c[0]  # central z position for proximal segment
z_dist = z_c[1]  # central z position for distal segment


def ellipsoid_coord(a_in, b_in, c_in, center_coord, theta_in, phi_in):
    x_or = a_in * np.transpose(np.cos(theta_in)) * np.sin(phi_in)
    y_or = b_in * np.transpose(np.sin(theta_in)) * np.sin(phi_in)
    z_or = c_in * np.cos(phi_in)
    # Rotated ellipsoid parametric equations
    x_tr = x_or + center_coord[0]
    y_tr = y_or + center_coord[1]
    z_tr = z_or + center_coord[2]
    return x_tr, y_tr, z_tr


def min_max_ellipse(a_in, b_in, c_in, center_coord, x_in):
    x_val = x_in - center_coord[0]
    if abs(x_val) - a_in == 0:
        x_val = np.sign(x_val) * (abs(x_val) - 1E-4)
    b_ellipse = np.sqrt(b_in ** 2 * (1 - (x_val ** 2) / (a_in ** 2)))
    c_ellipse = np.sqrt(c_in ** 2 * (1 - (x_val ** 2) / (a_in ** 2)))
    y_min = center_coord[1] - b_ellipse
    y_max = center_coord[1] + b_ellipse
    z_min = center_coord[2] - c_ellipse
    z_max = center_coord[2] + c_ellipse
    return y_min, y_max, z_min, z_max


# Set of points to plot the two ellipsoids (proximal and distal)
x_ell_pro, y_ell_pro, z_ell_pro = ellipsoid_coord(a[0], b[0], c[0], center[:3], theta, phi)
x_ell_dist, y_ell_dist, z_ell_dist = ellipsoid_coord(a[1], b[1], c[1], center[3:], theta, phi)


########################################################################################################################
# PLOTS OF HOW CONFIDENCE INTERVAL AND ELLIPSOID APPEAR IN THE ROBOT WORKSPACE

VecStart_x = [confInt_min[[0, 3]], confInt_max[[0, 3]], confInt_max[[0, 3]], confInt_min[[0, 3]]]
VecStart_y = [confInt_min[[1, 4]], confInt_min[[1, 4]], confInt_max[[1, 4]], confInt_max[[1, 4]]]
VecEnd_x = [confInt_max[[0, 3]], confInt_max[[0, 3]], confInt_min[[0, 3]], confInt_min[[0, 3]]]
VecEnd_y = [confInt_min[[1, 4]], confInt_max[[1, 4]], confInt_max[[1, 4]], confInt_min[[1, 4]]]

VecStart_x_1 = [min_robot[[0, 3]], max_robot[[0, 3]], max_robot[[0, 3]], min_robot[[0, 3]]]
VecStart_y_1 = [min_robot[[1, 4]], min_robot[[1, 4]], max_robot[[1, 4]], max_robot[[1, 4]]]
VecEnd_x_1 = [max_robot[[0, 3]], max_robot[[0, 3]], min_robot[[0, 3]], min_robot[[0, 3]]]
VecEnd_y_1 = [min_robot[[1, 4]], max_robot[[1, 4]], max_robot[[1, 4]], min_robot[[1, 4]]]


# fig_conf = plt.figure()
# ax = fig_conf.add_subplot(121, projection='3d')
# ax.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='y', label='babbling')
# for i in range(4):
#     ax.plot([VecStart_x[i][0], VecEnd_x[i][0]], [VecStart_y[i][0], VecEnd_y[i][0]], zs=z_pro, color='r')
#     if i == 0:
#         ax.plot([VecStart_x[i][0], VecEnd_x[i][0]], [VecStart_y[i][0], VecEnd_y[i][0]], zs=z_pro, color='r',
#                 label='confInt')
# for i in range(4):
#     ax.plot([VecStart_x_1[i][0], VecEnd_x_1[i][0]], [VecStart_y_1[i][0], VecEnd_y_1[i][0]], zs=z_pro, color='b')
#     if i == 0:
#         ax.plot([VecStart_x_1[i][0], VecEnd_x_1[i][0]], [VecStart_y_1[i][0], VecEnd_y_1[i][0]], zs=z_pro, color='b',
#                 label='minMax')
# ax.set_title('Proximal segment')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax = fig_conf.add_subplot(122, projection='3d')
# ax.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='y', label='babbling')
# for i in range(4):
#     ax.plot([VecStart_x[i][1], VecEnd_x[i][1]], [VecStart_y[i][1], VecEnd_y[i][1]], zs=z_dist, color='r')
#     if i == 0:
#         ax.plot([VecStart_x[i][1], VecEnd_x[i][1]], [VecStart_y[i][1], VecEnd_y[i][1]], zs=z_dist, color='r',
#                 label='confInt')
# for i in range(4):
#     ax.plot([VecStart_x_1[i][1], VecEnd_x_1[i][1]], [VecStart_y_1[i][1], VecEnd_y_1[i][1]], zs=z_dist, color='b')
#     if i == 0:
#         ax.plot([VecStart_x_1[i][1], VecEnd_x_1[i][1]], [VecStart_y_1[i][1], VecEnd_y_1[i][1]], zs=z_dist, color='b',
#                 label='minMax')
# ax.set_title('Distal segment')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# fig_conf.suptitle('Robot babbling data with delimiter ranges')
# plt.show()


def plot_parallelepiped(ax_in, min_vec, max_vec):
    # xy plane
    p = Rectangle((min_vec[0], min_vec[1]), max_vec[0] - min_vec[0], max_vec[1] - min_vec[1], alpha=0.2, color='y')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=min_vec[2], zdir="z")
    p = Rectangle((min_vec[0], min_vec[1]), max_vec[0] - min_vec[0], max_vec[1] - min_vec[1], alpha=0.2, color='y')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=max_vec[2], zdir="z")
    # xz plane
    p = Rectangle((min_vec[0], min_vec[2]), max_vec[0] - min_vec[0], max_vec[2] - min_vec[2], alpha=0.2, color='y')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=min_vec[1], zdir="y")
    p = Rectangle((min_vec[0], min_vec[2]), max_vec[0] - min_vec[0], max_vec[2] - min_vec[2], alpha=0.2, color='y')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=max_vec[1], zdir="y")
    # yz plane
    p = Rectangle((min_vec[1], min_vec[2]), max_vec[1] - min_vec[1], max_vec[2] - min_vec[2], alpha=0.2, color='y')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=min_vec[0], zdir="x")
    p = Rectangle((min_vec[1], min_vec[2]), max_vec[1] - min_vec[1], max_vec[2] - min_vec[2], alpha=0.2, color='y',
                  label='parallelepiped')
    ax_in.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=max_vec[0], zdir="x")


# fig_surf = plt.figure()
# fig_surf.suptitle('Robot babbling data with delimiter ranges')
# ax = fig_surf.add_subplot(121, projection='3d')
# ax.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='r', label='babbling')
# plot_parallelepiped(ax, confInt_min[:3], confInt_max[:3])
# surf = ax.plot_surface(x_ell_pro, y_ell_pro, z_ell_pro, alpha=0.3, label='ellipsoid')
# surf._facecolors2d = surf._facecolor3d
# surf._edgecolors2d = surf._edgecolor3d
# ax.legend()
# ax.set_title('Proximal segment')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax = fig_surf.add_subplot(122, projection='3d')
# ax.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='r', label='babbling')
# plot_parallelepiped(ax, confInt_min[3:], confInt_max[3:])
# surf = ax.plot_surface(x_ell_dist, y_ell_dist, z_ell_dist, alpha=0.3, label='ellipsoid')
# surf._facecolors2d = surf._facecolor3d
# surf._edgecolors2d = surf._edgecolor3d
# ax.legend()
# ax.set_title('Distal segment')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# plt.show()

########################################################################################################################
# HUMAN

data_0 = open('../Babbling/DataFolder_human/origin_ref/shift_data.txt', 'r').read()
file = data_0
file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
data = StringIO(file_1)
df_human = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
all_data_human = df_human.loc[:, "pmx":].to_numpy()
zdata_human = all_data_human[:, [2, 5]]
ydata_human = all_data_human[:, [1, 4]]
xdata_human = all_data_human[:, [0, 3]]

max_human = np.amax(all_data_human, axis=0)
min_human = np.amin(all_data_human, axis=0)

# PLOT OF HUMAN BABBLING DATA
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(xdata_human[:, 0], ydata_human[:, 0], zdata_human[:, 0], s=1, color='r', label='proximal')
# ax.scatter3D(xdata_human[:, 1], ydata_human[:, 1], zdata_human[:, 1], s=1, color='b', label='distal')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax.set_title('Babbling data of the human arm')
# plt.show()


# PLOT OF ALL BABBLING DATA
# fig_all = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=1, color='b', label='robot')
# ax.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=1, color='b')
# ax.scatter3D(xdata_human[:, 0], ydata_human[:, 0], zdata_human[:, 0], s=1, color='r', label='human')
# ax.scatter3D(xdata_human[:, 1], ydata_human[:, 1], zdata_human[:, 1], s=1, color='r')
# ax.legend()
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax.set_title('Babbling data: robot and human')
# plt.show()

########################################################################################################################
# PLOT HUMAN REMAPPED DATA (NORM NORMALIZATION) TOGETHER WITH ROBOT DATA

norm_robot = np.zeros((2,))  # a vector containing the norm of proximal and distal segments for robot
norm_robot_all = np.zeros((all_data_robot.shape[0], 2))
norm_robot_all[:, 0] = np.linalg.norm(all_data_robot[:, :3], axis=1)
norm_robot_all[:, 1] = np.linalg.norm(all_data_robot[:, 3:] - all_data_robot[:, :3], axis=1)
norm_robot[0] = np.mean(norm_robot_all[:, 0])  # proximal marker
norm_robot[1] = np.mean(norm_robot_all[:, 1])  # distal marker

norm_human = np.zeros((2,))  # a vector containing the norm of proximal and distal segments for human
norm_human_all = np.zeros((all_data_human.shape[0], 2))
norm_human_all[:, 0] = np.linalg.norm(all_data_human[:, :3], axis=1)
norm_human_all[:, 1] = np.linalg.norm(all_data_human[:, 3:] - all_data_human[:, :3], axis=1)
norm_human[0] = np.mean(norm_human_all[:, 0])  # proximal marker
norm_human[1] = np.mean(norm_human_all[:, 1])  # distal marker


def norm_remapping(data_original, demo_t):
    norm_val = norm_human
    if demo_t == 'robot':
        norm_all = np.zeros((data_original.shape[0], 2))
        norm_all[:, 0] = np.linalg.norm(data_original[:, :3], axis=1)
        norm_all[:, 1] = np.linalg.norm(data_original[:, 3:] - data_original[:, :3], axis=1)
        norm_val[0] = np.mean(norm_all[:, 0])  # proximal marker
        norm_val[1] = np.mean(norm_all[:, 1])  # distal marker
    remapped_aux = np.zeros(data_original.shape)
    remapped_aux[:, :3] = (data_original[:, :3] / norm_val[0]) * norm_robot[0]
    remapped_aux[:, 3:] = remapped_aux[:, :3] + ((data_original[:, 3:] - data_original[:, :3]) / norm_val[1]) * \
                          norm_robot[1]
    bias = np.mean(remapped_aux, axis=0) - np.mean(all_data_robot, axis=0)
    remapped_data = remapped_aux - bias
    return remapped_data


remapped_norm = norm_remapping(all_data_human, 'human')


# fig_n = plt.figure()
# ax_n = plt.axes(projection='3d')
# ax_n.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='b', label='robot')
# ax_n.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='b')
# ax_n.scatter3D(remapped_norm[:, 0], remapped_norm[:, 1], remapped_norm[:, 2], s=1, color='r', label='human remapped')
# ax_n.scatter3D(remapped_norm[:, 3], remapped_norm[:, 4], remapped_norm[:, 5], s=1, color='r')
# ax_n.legend()
# ax_n.set_xlabel('x', fontsize=15)
# ax_n.set_ylabel('y', fontsize=15)
# ax_n.set_zlabel('z', fontsize=15)
# ax_n.set_title('Babbling data: norm remapping')


# VecStart_x = [0, all_data_robot[0, 0]]
# VecStart_y = [0, all_data_robot[0, 1]]
# VecStart_z = [0, all_data_robot[0, 2]]
# VecEnd_x = [all_data_robot[0, 0], all_data_robot[0, 3]]
# VecEnd_y = [all_data_robot[0, 1], all_data_robot[0, 4]]
# VecEnd_z = [all_data_robot[0, 2], all_data_robot[0, 5]]
#
# VecStart_x_1 = [0, all_data_human[0, 0]]
# VecStart_y_1 = [0, all_data_human[0, 1]]
# VecStart_z_1 = [0, all_data_human[0, 2]]
# VecEnd_x_1 = [all_data_human[0, 0], all_data_human[0, 3]]
# VecEnd_y_1 = [all_data_human[0, 1], all_data_human[0, 4]]
# VecEnd_z_1 = [all_data_human[0, 2], all_data_human[0, 5]]
#
# fig_norms = plt.figure()
# ax = fig_norms.add_subplot(121, projection='3d')
# ax.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=1, color='y', label='babbling')
# ax.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=1, color='y')
# for i in range(2):
#     ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]],
#             zs=[VecStart_z[i], VecEnd_z[i]], color='r')
#     if i == 0:
#         ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]],
#                 zs=[VecStart_z[i], VecEnd_z[i]], color='r', label='robot')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# ax.set_title('Robot')
# ax = fig_norms.add_subplot(122, projection='3d')
# ax.scatter3D(xdata_human[:, 0], ydata_human[:, 0], zdata_human[:, 0], s=1, color='y', label='babbling')
# ax.scatter3D(xdata_human[:, 1], ydata_human[:, 1], zdata_human[:, 1], s=1, color='y')
# for i in range(2):
#     ax.plot([VecStart_x_1[i], VecEnd_x_1[i]], [VecStart_y_1[i], VecEnd_y_1[i]],
#             zs=[VecStart_z_1[i], VecEnd_z_1[i]], color='b')
#     if i == 0:
#         ax.plot([VecStart_x_1[i], VecEnd_x_1[i]], [VecStart_y_1[i], VecEnd_y_1[i]],
#                 zs=[VecStart_z_1[i], VecEnd_z_1[i]], color='b', label='human')
# ax.legend()
# ax.set_title('Human')
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('y', fontsize=15)
# ax.set_zlabel('z', fontsize=15)
# fig_norms.suptitle('Norms')
# plt.show()


########################################################################################################################
# PLOT HUMAN REMAPPED DATA (MIN-MAX NORMALIZATION) TOGETHER WITH ROBOT DATA

def minMax_remapping(data_original):
    min_local = np.min(data_original, axis=0)
    max_local = np.max(data_original, axis=0)
    remapped_data = ((data_original - min_local) / (max_local - min_local)) * (max_robot - min_robot) + min_robot
    return remapped_data


remapped_minMax = minMax_remapping(all_data_human)

# fig0 = plt.figure()
# ax0 = plt.axes(projection='3d')
# ax0.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='b', label='robot')
# ax0.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='b')
# ax0.scatter3D(remapped_minMax[:, 0], remapped_minMax[:, 1], remapped_minMax[:, 2], s=1, color='r',
#               label='human remapped')
# ax0.scatter3D(remapped_minMax[:, 3], remapped_minMax[:, 4], remapped_minMax[:, 5], s=1, color='r')
# ax0.legend()
# ax0.set_xlabel('x', fontsize=15)
# ax0.set_ylabel('y', fontsize=15)
# ax0.set_zlabel('z', fontsize=15)
# ax0.set_title('Babbling data: min-max remapping')

########################################################################################################################
# PLOT HUMAN REMAPPED DATA (Z-SCORE NORMALIZATION) TOGETHER WITH ROBOT DATA
mean_human = np.mean(all_data_human, axis=0)
std_human = np.std(all_data_human, axis=0)
mean_robot = np.mean(all_data_robot, axis=0)
std_robot = np.std(all_data_robot, axis=0)


def zScore_remapping(data_original, demo_t):
    mean_val = mean_human
    std_val = std_human
    if demo_t == 'robot':
        mean_val = np.mean(data_original, axis=0)
        std_val = np.std(data_original, axis=0)
    remapped_data = ((data_original - mean_val) / std_val) * std_robot + mean_robot
    return remapped_data


remapped_z = zScore_remapping(all_data_human, 'human')


# fig_z = plt.figure()
# ax_z = plt.axes(projection='3d')
# ax_z.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='b', label='robot')
# ax_z.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='b')
# ax_z.scatter3D(remapped_z[:, 0], remapped_z[:, 1], remapped_z[:, 2], s=1, color='r', label='human remapped')
# ax_z.scatter3D(remapped_z[:, 3], remapped_z[:, 4], remapped_z[:, 5], s=1, color='r')
# ax_z.legend()
# ax_z.set_xlabel('x', fontsize=15)
# ax_z.set_ylabel('y', fontsize=15)
# ax_z.set_zlabel('z', fontsize=15)
# ax_z.set_title('Babbling data: Z-score remapping')


########################################################################################################################
# PLOT HUMAN REMAPPED DATA (CONFIDENCE INTERVAL 95%) TOGETHER WITH ROBOT DATA

def confInt_remapping(data_original):
    min_local = np.min(data_original, axis=0)
    max_local = np.max(data_original, axis=0)
    remapped_data = confInt_min + ((data_original - min_local) / (max_local - min_local)) * (confInt_max - confInt_min)
    return remapped_data


remapped_confInt = confInt_remapping(all_data_human)


# fig_c = plt.figure()
# ax_c = plt.axes(projection='3d')
# ax_c.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='b', label='robot')
# ax_c.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='b')
# ax_c.scatter3D(remapped_confInt[:, 0], remapped_confInt[:, 1], remapped_confInt[:, 2], s=1, color='r',
#                label='human remapped')
# ax_c.scatter3D(remapped_confInt[:, 3], remapped_confInt[:, 4], remapped_confInt[:, 5], s=1, color='r')
# ax_c.legend()
# ax_c.set_xlabel('x', fontsize=15)
# ax_c.set_ylabel('y', fontsize=15)
# ax_c.set_zlabel('z', fontsize=15)
# ax_c.set_title('Babbling data: confInt remapping')


########################################################################################################################
# PLOT HUMAN REMAPPED DATA (ELLIPSOID CONFIDENCE INTERVAL) TOGETHER WITH ROBOT DATA

def ellipsoid_remapping(data_original):
    confInt_min_ell = center - 1.6 * aux_std
    confInt_max_ell = center + 1.6 * aux_std
    min_local = np.min(data_original, axis=0)
    max_local = np.max(data_original, axis=0)
    remapped_data = np.zeros(data_original.shape)
    for p in range(data_original.shape[0]):
        # x axis remapped with confInt_ell
        x_val = confInt_min_ell[[0, 3]] + ((data_original[p, [0, 3]] - min_local[[0, 3]]) /
                                           (max_local[[0, 3]] - min_local[[0, 3]])) * (
                        confInt_max_ell[[0, 3]] - confInt_min_ell[[0, 3]])
        remapped_data[p, [0, 3]] = x_val
        # Proximal segment
        y_min, y_max, z_min, z_max = min_max_ellipse(a[0], b[0], c[0], center[:3], x_val[0])
        remapped_data[p, 1] = y_min + ((data_original[p, 1] - min_local[1]) /
                                       (max_local[1] - min_local[1])) * (y_max - y_min)
        remapped_data[p, 2] = z_min + ((data_original[p, 2] - min_local[2]) /
                                       (max_local[2] - min_local[2])) * (z_max - z_min)
        # Distal segment
        y_min, y_max, z_min, z_max = min_max_ellipse(a[1], b[1], c[1], center[3:], x_val[1])
        remapped_data[p, 4] = y_min + ((data_original[p, 4] - min_local[4]) /
                                       (max_local[4] - min_local[4])) * (y_max - y_min)
        remapped_data[p, 5] = z_min + ((data_original[p, 5] - min_local[5]) /
                                       (max_local[5] - min_local[5])) * (z_max - z_min)
    if flag_rotate:
        remapped_data[:, :3] = rotate(remapped_data[:, :3], -m_angle)
    return remapped_data


remapped_ellipsoid = ellipsoid_remapping(all_data_human)

fig_e = plt.figure()
ax_e = plt.axes(projection='3d')
ax_e.scatter3D(xdata_robot[:, 0], ydata_robot[:, 0], zdata_robot[:, 0], s=0.3, color='b', label='robot')
ax_e.scatter3D(xdata_robot[:, 1], ydata_robot[:, 1], zdata_robot[:, 1], s=0.3, color='b')
ax_e.scatter3D(remapped_ellipsoid[:, 0], remapped_ellipsoid[:, 1], remapped_ellipsoid[:, 2], s=1, color='r',
               label='human remapped')
ax_e.scatter3D(remapped_ellipsoid[:, 3], remapped_ellipsoid[:, 4], remapped_ellipsoid[:, 5], s=1, color='r')
ax_e.legend()
ax_e.set_xlabel('x', fontsize=15)
ax_e.set_ylabel('y', fontsize=15)
ax_e.set_zlabel('z', fontsize=15)
ax_e.set_title('Babbling data: ellipsoid remapping')

####################################################################################################################
# WRITE REMAPPED DATA INTO A NEW FILE

# demo_type_list = ['robot', 'human']
demo_type_list = ['human']
file_names = ['circle', 'loop', 'cross']

for t in range(len(demo_type_list)):

    demo_type = demo_type_list[t]

    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(all_data_original[:, 0], all_data_original[:, 1], all_data_original[:, 2], s=0.3, c='y', label='babbling robot')
    ax1.scatter3D(all_data_original[:, 3], all_data_original[:, 4], all_data_original[:, 5], s=0.3, c='y')

    for n in range(len(file_names)):
        rev = 0
        file_name = file_names[n]

        while True:  # File to read data from
            path = 'DEMONSTRATIONS/' + demo_type + '/origin_ref/shift_' + file_name + '_' + str(rev) + '.txt'
            if not os.path.isfile(path):
                # If the file doesn't exist I exit the loop
                break
            file = open(path, 'r').read()
            file_3 = file.replace(',\n', ",")
            file_3 = file_3.replace('])', "")
            file_3 = file_3.replace('array([', "")
            file_3 = file_3.replace('[', "")
            file_3 = file_3.replace(']', "")
            data = StringIO(file_3)
            df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
            markers = df.loc[:, "pmx":].to_numpy()

            # REMAPPING
            markers_z = zScore_remapping(markers, demo_type_list[t])  # Z-score remapping
            markers_mM = minMax_remapping(markers)  # MIN-MAX remapping
            markers_n = norm_remapping(markers, demo_type_list[t])  # norm remapping
            markers_c = confInt_remapping(markers)  # confidence interval remapping
            markers_e = ellipsoid_remapping(markers)  # ellipsoid remapping

            # ax1.plot(markers_z[:, 0], markers_z[:, 1], markers_z[:, 2], color='r', label='Z-score remapping')
            # ax1.plot(markers_mM[:, 0], markers_mM[:, 1], markers_mM[:, 2], color='b', label='min_MAX remapping')
            # ax1.plot(markers_n[:, 0], markers_n[:, 1], markers_n[:, 2], color='g', label='Norm remapping')
            # ax1.plot(markers_c[:, 0], markers_c[:, 1], markers_c[:, 2], linewidth=0.5, color='b',
            #          label='confInt remap')
            ax1.plot(markers_e[:, 0], markers_e[:, 1], markers_e[:, 2], linewidth=0.5, color='k',
                     label='ellipsoid remapping')
            # ax1.plot(markers_z[:, 3], markers_z[:, 4], markers_z[:, 5], color='r')
            # ax1.plot(markers_mM[:, 3], markers_mM[:, 4], markers_mM[:, 5], color='b')
            # ax1.plot(markers_n[:, 3], markers_n[:, 4], markers_n[:, 5], color='g')
            # ax1.plot(markers_c[:, 3], markers_c[:, 4], markers_c[:, 5], color='b', linewidth=0.5)
            ax1.plot(markers_e[:, 3], markers_e[:, 4], markers_e[:, 5], color='k', linewidth=0.5)
            if rev == 0 and n == 0:
                ax1.legend()

            ############################################################################################################
            # # Z-score normalization
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '_remapped/zScore/shift_' + file_name + '_' + str(rev) + '.txt'
            # with open(file_wname, 'w') as f:
            #     for d in range(markers_z.shape[0]):
            #         f.write(str([d, markers_z[d, :]]).strip('[]'))
            #         f.write('\n')
            #
            # # min-Max normalization
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '_remapped/minMax/shift_' + file_name + '_' + str(rev) + '.txt'
            # with open(file_wname, 'w') as f:
            #     for d in range(markers_mM.shape[0]):
            #         f.write(str([d, markers_mM[d, :]]).strip('[]'))
            #         f.write('\n')
            #
            # # Norm normalization
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '_remapped/norm/shift_' + file_name + '_' + str(rev) + '.txt'
            # with open(file_wname, 'w') as f:
            #     for d in range(markers_n.shape[0]):
            #         f.write(str([d, markers_n[d, :]]).strip('[]'))
            #         f.write('\n')
            #
            # # Confidence interval normalization
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '_remapped/confInt/shift_' + file_name + '_' + str(
            #     rev) + '.txt'
            # with open(file_wname, 'w') as f:
            #     for d in range(markers_c.shape[0]):
            #         f.write(str([d, markers_c[d, :]]).strip('[]'))
            #         f.write('\n')
            #
            # # Ellipsoid normalization
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '_remapped/ellipsoid/shift_' + file_name + '_' + str(
            #     rev) + '.txt'
            # with open(file_wname, 'w') as f:
            #     for d in range(markers_e.shape[0]):
            #         f.write(str([d, markers_e[d, :]]).strip('[]'))
            #         f.write('\n')

            rev += 1

    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.set_zlabel("z", fontsize=15)
    fig1.suptitle(f'All {demo_type} demos remapped')
    plt.show()
