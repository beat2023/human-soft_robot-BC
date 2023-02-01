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
demo_name = 'loop'
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
