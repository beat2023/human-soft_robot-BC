# Visualize policy result together with the demonstrations

import os
from io import StringIO
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

########################################################################################################################

# demo_type = 'human' or 'robot'
demo_type = 'human'
# demo_name = 'elongation' or 'circle' or 'loop'
demo_name = 'cross'
# remap_type = 'minMax', 'norm', 'zScore', 'confInt' or 'ellipsoid'
remap_type = 'ellipsoid'

episode = 0
MAE = []
MAE_idm = []
idm_all = []

fig_all = plt.figure()
ax1_all = fig_all.add_subplot(121, projection='3d')
ax1_all.set_title('Proximal segment')
ax1_all.set_xlabel("x", fontsize=15)
ax1_all.set_ylabel("y", fontsize=15)
ax1_all.set_zlabel("z", fontsize=15)
ax2_all = fig_all.add_subplot(122, projection='3d')
ax2_all.set_title('Distal segment')
ax2_all.set_xlabel("x", fontsize=15)
ax2_all.set_ylabel("y", fontsize=15)
ax2_all.set_zlabel("z", fontsize=15)
fig_all.suptitle(f"All episodes of idm-passed demonstration: {demo_name} motion\n"
                 f"Max distance between paths reported")

while True:
    path = 'PostDemo/' + demo_type + '/' + remap_type + '/' + demo_name + '_' + str(episode) + '.txt'
    if not os.path.isfile(path):
        # If the file doesn't exist I exit the loop
        break
    data = open(path, 'r').read()
    data = data.replace(',\n', ",")
    data = data.replace('])', "")
    data = data.replace('array([', "")
    data = data.replace('[', "")
    data = data.replace(']', "")
    data = data.replace(', dtype=float32)', "")
    data = StringIO(data)
    df2 = pd.read_csv(data, sep=",", names=["i", "demo_pmx", "demo_pmy", "demo_pmz", "demo_dmx", "demo_dmy", "demo_dmz",
                                            "idm_pmx", "idm_pmy", "idm_pmz", "idm_dmx", "idm_dmy", "idm_dmz",
                                            "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
    states_demo = df2.loc[:, "demo_pmx":"demo_dmz"].to_numpy()
    resampled_idm = df2.loc[:, "idm_pmx":"idm_dmz"].to_numpy()
    resampled_policy = df2.loc[:, "pmx":"dmz"].to_numpy()

    MAE.append(mean_absolute_error(states_demo[:, 3:], resampled_policy[:, 3:]))
    MAE_idm.append(mean_absolute_error(resampled_idm[:, 3:], resampled_policy[:, 3:]))
    idm_all.append(resampled_idm)

    if MAE[episode] <= min(MAE):
        episode_best = episode
        states_demo_best = states_demo
        resampled_idm_best = resampled_idm
        resampled_policy_best = resampled_policy

    ax1_all.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g')
    ax2_all.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g')

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot(states_demo[:, 0], states_demo[:, 1], states_demo[:, 2], color='r', label='demonstration')
    # ax1.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g', label='idm')
    # ax1.plot(resampled_policy[:, 0], resampled_policy[:, 1], resampled_policy[:, 2], color='b', label='policy')
    # ax1.set_title('Proximal segment')
    # ax1.legend()
    # ax1.set_xlabel("x", fontsize=15)
    # ax1.set_ylabel("y", fontsize=15)
    # ax1.set_zlabel("z", fontsize=15)
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot(states_demo[:, 3], states_demo[:, 4], states_demo[:, 5], color='r', label='demonstration')
    # ax2.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g', label='idm')
    # ax2.plot(resampled_policy[:, 3], resampled_policy[:, 4], resampled_policy[:, 5], color='b', label='policy')
    # ax2.set_title('Distal segment')
    # ax2.legend()
    # ax2.set_xlabel("x", fontsize=15)
    # ax2.set_ylabel("y", fontsize=15)
    # ax2.set_zlabel("z", fontsize=15)
    # fig = plt.gcf()
    # fig.suptitle(f"Post-demonstration number {episode+1}\nMAE with original demo (distal segment): {round(MAE[episode], 3)}"
    #              f"\nMAE with idm-passed demo (distal segment): {round(MAE_idm[episode], 3)}")
    # plt.show()

    episode += 1

# Find the highest distance between paths for the 6 coordinates
max_dist_all = np.zeros((6,))
max_idx_all = np.zeros((6,))
max_dist_paths = np.zeros((2, 6))
for i in range(len(idm_all)):
    for j in range(len(idm_all)):
        distance = abs(idm_all[i] - idm_all[j])
        max_dist = np.max(distance, axis=0)
        max_idx = np.argmax(distance, axis=0)
        for k in range(len(max_dist)):
            if max_dist[k] > max_dist_all[k]:
                max_dist_all[k] = max_dist[k]
                max_idx_all[k] = max_idx[k]
                max_dist_paths[0, k] = i
                max_dist_paths[1, k] = j

max_idx_all = max_idx_all.astype(int)
max_dist_paths = max_dist_paths.astype(int)

x_prox_line = [idm_all[max_dist_paths[0, 0]][max_idx_all[0], 0], idm_all[max_dist_paths[1, 0]][max_idx_all[0], 0]]
x_prox_y = np.repeat(np.mean([idm_all[max_dist_paths[0, 0]][max_idx_all[0], 1], idm_all[max_dist_paths[1, 0]][max_idx_all[0], 1]]), 2)
x_prox_z = np.repeat(np.mean([idm_all[max_dist_paths[0, 0]][max_idx_all[0], 2], idm_all[max_dist_paths[1, 0]][max_idx_all[0], 2]]), 2)

y_prox_line = [idm_all[max_dist_paths[0, 1]][max_idx_all[1], 1], idm_all[max_dist_paths[1, 1]][max_idx_all[1], 1]]
y_prox_x = np.repeat(np.mean([idm_all[max_dist_paths[0, 1]][max_idx_all[1], 0], idm_all[max_dist_paths[1, 1]][max_idx_all[1], 0]]), 2)
y_prox_z = np.repeat(np.mean([idm_all[max_dist_paths[0, 1]][max_idx_all[1], 2], idm_all[max_dist_paths[1, 1]][max_idx_all[1], 2]]), 2)

z_prox_line = [idm_all[max_dist_paths[0, 2]][max_idx_all[2], 2], idm_all[max_dist_paths[1, 2]][max_idx_all[2], 2]]
z_prox_x = np.repeat(np.mean([idm_all[max_dist_paths[0, 2]][max_idx_all[2], 0], idm_all[max_dist_paths[1, 2]][max_idx_all[2], 0]]), 2)
z_prox_y = np.repeat(np.mean([idm_all[max_dist_paths[0, 2]][max_idx_all[2], 1], idm_all[max_dist_paths[1, 2]][max_idx_all[2], 1]]), 2)

x_dist_line = [idm_all[max_dist_paths[0, 3]][max_idx_all[3], 3], idm_all[max_dist_paths[1, 3]][max_idx_all[3], 3]]
x_dist_y = np.repeat(np.mean([idm_all[max_dist_paths[0, 3]][max_idx_all[3], 4], idm_all[max_dist_paths[1, 3]][max_idx_all[3], 4]]), 2)
x_dist_z = np.repeat(np.mean([idm_all[max_dist_paths[0, 3]][max_idx_all[3], 5], idm_all[max_dist_paths[1, 3]][max_idx_all[3], 5]]), 2)

y_dist_line = [idm_all[max_dist_paths[0, 4]][max_idx_all[4], 4], idm_all[max_dist_paths[1, 4]][max_idx_all[4], 4]]
y_dist_x = np.repeat(np.mean([idm_all[max_dist_paths[0, 4]][max_idx_all[4], 3], idm_all[max_dist_paths[1, 4]][max_idx_all[4], 3]]), 2)
y_dist_z = np.repeat(np.mean([idm_all[max_dist_paths[0, 4]][max_idx_all[4], 5], idm_all[max_dist_paths[1, 4]][max_idx_all[4], 5]]), 2)

z_dist_line = [idm_all[max_dist_paths[0, 5]][max_idx_all[5], 5], idm_all[max_dist_paths[1, 5]][max_idx_all[5], 5]]
z_dist_x = np.repeat(np.mean([idm_all[max_dist_paths[0, 5]][max_idx_all[5], 3], idm_all[max_dist_paths[1, 5]][max_idx_all[5], 3]]), 2)
z_dist_y = np.repeat(np.mean([idm_all[max_dist_paths[0, 5]][max_idx_all[5], 4], idm_all[max_dist_paths[1, 5]][max_idx_all[5], 4]]), 2)

ax1_all.plot(x_prox_line, x_prox_y, x_prox_z, color='r', label=f'x: {round(max_dist_all[0])} mm')
ax1_all.plot(y_prox_x, y_prox_line, y_prox_z, color='orange', label=f'y: {round(max_dist_all[1])} mm')
ax1_all.plot(z_prox_x, z_prox_y, z_prox_line, color='m', label=f'z: {round(max_dist_all[2])} mm')
ax2_all.plot(x_dist_line, x_dist_y, x_dist_z, color='r', label=f'x: {round(max_dist_all[3])} mm')
ax2_all.plot(y_dist_x, y_dist_line, y_dist_z, color='orange', label=f'y: {round(max_dist_all[4])} mm')
ax2_all.plot(z_dist_x, z_dist_y, z_dist_line, color='m', label=f'z: {round(max_dist_all[5])} mm')
ax1_all.legend()
ax2_all.legend()

plt.show()

MAE_smoothed = savgol_filter(MAE, 10, 2)

# fig_MAE = plt.figure
# plt.title(f'Post demonstration MAE evolution: {demo_name} motion')
# plt.plot(MAE, c='b', label='true')
# plt.plot(MAE_smoothed, c='r', label='smoothed')
# plt.legend()
# plt.ylabel('MAE')
# plt.xlabel('Episode')
# plt.xticks(np.arange(0, 20), labels=range(1, 21))
# plt.grid()
#
# fig_best = plt.figure()
# ax1_best = fig_best.add_subplot(121, projection='3d')
# ax1_best.plot(states_demo_best[:, 0], states_demo_best[:, 1], states_demo_best[:, 2], color='r', label='demonstration')
# ax1_best.plot(resampled_idm_best[:, 0], resampled_idm_best[:, 1], resampled_idm_best[:, 2], color='g', label='idm')
# ax1_best.plot(resampled_policy_best[:, 0], resampled_policy_best[:, 1], resampled_policy_best[:, 2], color='b',
#               label='policy')
# ax1_best.set_title('Proximal segment')
# ax1_best.legend()
# ax1_best.set_xlabel("x", fontsize=15)
# ax1_best.set_ylabel("y", fontsize=15)
# ax1_best.set_zlabel("z", fontsize=15)
# ax2_best = fig_best.add_subplot(122, projection='3d')
# ax2_best.plot(states_demo_best[:, 3], states_demo_best[:, 4], states_demo_best[:, 5], color='r', label='demonstration')
# ax2_best.plot(resampled_idm_best[:, 3], resampled_idm_best[:, 4], resampled_idm_best[:, 5], color='g', label='idm')
# ax2_best.plot(resampled_policy_best[:, 3], resampled_policy_best[:, 4], resampled_policy_best[:, 5], color='b',
#               label='policy')
# ax2_best.set_title('Distal segment')
# ax2_best.legend()
# ax2_best.set_xlabel("x", fontsize=15)
# ax2_best.set_ylabel("y", fontsize=15)
# ax2_best.set_zlabel("z", fontsize=15)
# fig_best.suptitle(f"Best post-demonstration for {demo_name} motion (episode {episode_best + 1})"
#                   f"\nMAE with original demo (distal segment): {round(MAE[episode_best], 3)}"
#                   f"\nMAE with idm-passed demo (distal segment): {round(MAE_idm[episode_best], 3)}")
# plt.show()
