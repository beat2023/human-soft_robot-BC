# Visualize policy test result together with the demonstrations

import os
from io import StringIO
import numpy as np
import pandas as pd
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
states_demo_all = []
resampled_idm_all = []
resampled_policy_all = []

fig_all = plt.figure()
ax1_all = fig_all.add_subplot(121, projection='3d')
ax2_all = fig_all.add_subplot(122, projection='3d')

while True:
    path = 'TestPolicy/' + demo_type + '/' + remap_type + '/' + demo_name + '_' + str(episode) + '.txt'
    if not os.path.isfile(path):
        # If the file doesn't exist I exit the loop
        break
    data = open(path, 'r').read()
    data = data.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
    data = StringIO(data)
    df2 = pd.read_csv(data, sep=",", names=["i", "demo_pmx", "demo_pmy", "demo_pmz", "demo_dmx", "demo_dmy", "demo_dmz",
                                            "idm_pmx", "idm_pmy", "idm_pmz", "idm_dmx", "idm_dmy", "idm_dmz",
                                            "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
    states_demo = df2.loc[:, "demo_pmx":"demo_dmz"].to_numpy()
    resampled_idm = df2.loc[:, "idm_pmx":"idm_dmz"].to_numpy()
    resampled_policy = df2.loc[:, "pmx":"dmz"].to_numpy()

    states_demo_all.append(states_demo)
    resampled_idm_all.append(resampled_idm)
    resampled_policy_all.append(resampled_policy)

    # MAE calculation
    MAE.append(mean_absolute_error(states_demo[:, 3:], resampled_policy[:, 3:]))
    MAE_idm.append(mean_absolute_error(resampled_idm[:, 3:], resampled_policy[:, 3:]))

    if episode == 0:
        ax1_all.plot(states_demo[:, 0], states_demo[:, 1], states_demo[:, 2], color='r', label='demonstration')
        ax1_all.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g', label='idm')
        ax2_all.plot(states_demo[:, 3], states_demo[:, 4], states_demo[:, 5], color='r', label='demonstration')
        ax2_all.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g', label='idm')
        ax1_all.plot(resampled_policy[:, 0], resampled_policy[:, 1], resampled_policy[:, 2], color='b', label='policy')
        ax2_all.plot(resampled_policy[:, 3], resampled_policy[:, 4], resampled_policy[:, 5], color='b', label='policy')
    else:
        ax1_all.plot(resampled_policy[:, 0], resampled_policy[:, 1], resampled_policy[:, 2], color='b')
        ax2_all.plot(resampled_policy[:, 3], resampled_policy[:, 4], resampled_policy[:, 5], color='b')

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
    # fig.suptitle(f"Testing number {episode}\nMAE with original demo (distal segment): {round(MAE[episode], 3)}"
    #              f"\nMAE with idm-passed demo (distal segment): {round(MAE_idm[episode], 3)}")
    # plt.show()

    episode += 1

MAE_avg = np.mean(MAE)
MAE_std = np.std(MAE)
MAE_idm_avg = np.mean(MAE_idm)
MAE_idm_std = np.std(MAE_idm)

# Plot all paths
ax1_all.set_title('Proximal segment')
ax1_all.set_xlabel("x", fontsize=15)
ax1_all.set_ylabel("y", fontsize=15)
ax1_all.set_zlabel("z", fontsize=15)
ax1_all.legend()
ax2_all.set_title('Distal segment')
ax2_all.set_xlabel("x", fontsize=15)
ax2_all.set_ylabel("y", fontsize=15)
ax2_all.set_zlabel("z", fontsize=15)
ax2_all.legend()
fig_all.suptitle(f"TESTING: avg\u00B1std"
                 f"\nMAE with original demo (distal segment): {round(float(MAE_avg), 3)}\u00B1"
                 f"{round(float(MAE_std), 3)}"
                 f"\nMAE with idm-passed demo (distal segment): {round(float(MAE_idm_avg), 3)}\u00B1"
                 f"{round(float(MAE_idm_std), 3)}")
plt.show()

########################################################################################################################
# Plot all paths except the outlier ones
MAE_median = np.median(MAE)
idx_outlier = abs(MAE - MAE_median) > 2.5  # more than 2.5 mm of difference in MAE

if not np.any(idx_outlier):
    print("No outlier trajectories detected")

else:
    MAE = np.array(MAE)
    MAE_idm = np.array(MAE_idm)
    MAE_new = MAE[~idx_outlier]
    MAE_idm_new = MAE_idm[~idx_outlier]
    MAE_new_avg = np.mean(MAE_new)
    MAE_new_std = np.std(MAE_new)
    MAE_idm_new_avg = np.mean(MAE_idm_new)
    MAE_idm_new_std = np.std(MAE_idm_new)

    fig_new = plt.figure()
    ax1_new = fig_new.add_subplot(121, projection='3d')
    ax2_new = fig_new.add_subplot(122, projection='3d')

    for j in range(len(MAE_new)):
        j_used = j
        while idx_outlier[j_used]:
            j_used += 1
        if j_used == 0:
            ax1_new.plot(states_demo_all[j_used][:, 0], states_demo_all[j_used][:, 1], states_demo_all[j_used][:, 2],
                         color='r', label='demonstration')
            ax1_new.plot(resampled_idm_all[j_used][:, 0], resampled_idm_all[j_used][:, 1],
                         resampled_idm_all[j_used][:, 2], color='g', label='idm')
            ax2_new.plot(states_demo_all[j_used][:, 3], states_demo_all[j_used][:, 4], states_demo_all[j_used][:, 5],
                         color='r', label='demonstration')
            ax2_new.plot(resampled_idm_all[j_used][:, 3], resampled_idm_all[j_used][:, 4],
                         resampled_idm_all[j_used][:, 5], color='g', label='idm')
            ax1_new.plot(resampled_policy_all[j_used][:, 0], resampled_policy_all[j_used][:, 1],
                         resampled_policy_all[j_used][:, 2], color='b',
                         label='policy')
            ax2_new.plot(resampled_policy_all[j_used][:, 3], resampled_policy_all[j_used][:, 4],
                         resampled_policy_all[j_used][:, 5], color='b',
                         label='policy')
        else:
            ax1_new.plot(resampled_policy_all[j_used][:, 0], resampled_policy_all[j_used][:, 1],
                         resampled_policy_all[j_used][:, 2], color='b')
            ax2_new.plot(resampled_policy_all[j_used][:, 3], resampled_policy_all[j_used][:, 4],
                         resampled_policy_all[j_used][:, 5], color='b')

    ax1_new.set_title('Proximal segment')
    ax1_new.set_xlabel("x", fontsize=15)
    ax1_new.set_ylabel("y", fontsize=15)
    ax1_new.set_zlabel("z", fontsize=15)
    ax1_new.legend()
    ax2_new.set_title('Distal segment')
    ax2_new.set_xlabel("x", fontsize=15)
    ax2_new.set_ylabel("y", fontsize=15)
    ax2_new.set_zlabel("z", fontsize=15)
    ax2_new.legend()
    fig_new.suptitle(f"TESTING (without outlier paths): avg\u00B1std"
                     f"\nMAE with original demo (distal segment): {round(float(MAE_new_avg), 3)}\u00B1"
                     f"{round(float(MAE_new_std), 3)}"
                     f"\nMAE with idm-passed demo (distal segment): {round(float(MAE_idm_new_avg), 3)}\u00B1"
                     f"{round(float(MAE_idm_new_std), 3)}")
    plt.show()


########################################################################################################################
# Distribution of policy testing
policy_no_out = np.array(resampled_policy_all)
policy_no_out = policy_no_out[~idx_outlier, :, :]

max_dist_all = np.zeros((6,))
max_idx_all = np.zeros((6,))
max_dist_paths = np.zeros((2, 6))
for i in range(len(policy_no_out)):
    for j in range(len(policy_no_out)):
        distance = abs(policy_no_out[i, :, :] - policy_no_out[j, :, :])
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

mrk_avg = np.zeros(policy_no_out[0].shape)
mrk_std = np.zeros(policy_no_out[0].shape)
for j in range(policy_no_out.shape[1]):
    mrk_avg[j, :] = np.mean(policy_no_out[:, j, :], axis=0)
for j in range(policy_no_out.shape[1]):
    mrk_std[j, :] = np.std(policy_no_out[:, j, :], axis=0)

fig, axs = plt.subplots(6, 2)
fig.suptitle(f'Testing {demo_name} policy')
axs[0, 0].title.set_text('All episodes')
axs[0, 1].title.set_text('Average \u00B1 2*standard deviation')
for s in range(policy_no_out.shape[0]):
    for m in range(policy_no_out.shape[2]):
        axs[m, 0].plot(policy_no_out[s, :, m], '--', linewidth=0.3)
        axs[m, 0].plot([max_idx_all[m], max_idx_all[m]], [policy_no_out[max_dist_paths[0, m], max_idx_all[m], m],
                            policy_no_out[max_dist_paths[1, m], max_idx_all[m], m]], color='red', linewidth=0.8,
                            label=f'{round(max_dist_all[m], 1)}mm')
        if s == 0:
            axs[m, 0].legend(loc='upper left')
axs[0, 0].set(ylabel='x proximal')
axs[1, 0].set(ylabel='y proximal')
axs[2, 0].set(ylabel='z proximal')
axs[3, 0].set(ylabel='x distal')
axs[4, 0].set(ylabel='y distal')
axs[5, 0].set(ylabel='z distal')
for m in range(policy_no_out.shape[2]):
    axs[m, 1].plot(mrk_avg[:, m], '-', color='k', linewidth=1.5, label='avg')
    axs[m, 1].plot(mrk_avg[:, m] + 2*mrk_std[:, m], '--', color='k', linewidth=0.8, label='2*std')
    axs[m, 1].plot(mrk_avg[:, m] - 2*mrk_std[:, m], '--', color='k', linewidth=0.8)
for ax in axs.flat:
    ax.set(xlabel='Sample number')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.legend()
plt.show()

all_vals = []
for m in range(6):
    val = 2*np.mean(mrk_std[:, m])
    all_vals.append(val)
    print(f'2*std dimension {m}: {round(val, 1)}')
all_vals = np.array(all_vals)
all_vals_avg = np.mean(all_vals)
print(f'Average value: {round(all_vals_avg, 1)}')
