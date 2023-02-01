# Visualize policy result together with the demonstrations
import os
from io import StringIO
import pandas as pd
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np

# demo_type = 'human' or 'robot'
demo_type = 'human'
demo_name = ['circle', 'loop', 'cross']
# demo_name = ['circle']
# remap_type = 'minMax', 'norm', 'zScore', 'confInt' or 'ellipsoid'
remap_type = 'ellipsoid'

flag_plot_all = False   # if I want to plot the trajectories


for n in range(len(demo_name)):

    episode = 0
    MAE = []
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
    fig_all.suptitle(f"All idm-passed demonstrations: {demo_name[n]} motion")

    while True:
        path = 'IdmPerform/' + demo_type + '/' + remap_type + '/' + demo_name[n] + '_' + str(episode) + '.txt'
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
                                                "idm_pmx", "idm_pmy", "idm_pmz", "idm_dmx", "idm_dmy", "idm_dmz"])
        states_demo = df2.loc[:, "demo_pmx":"demo_dmz"].to_numpy()
        resampled_idm = df2.loc[:, "idm_pmx":"idm_dmz"].to_numpy()

        MAE.append(mean_absolute_error(states_demo[:, 3:], resampled_idm[:, 3:]))
        if MAE[episode] <= min(MAE):
            episode_best = episode
            states_demo_best = states_demo
            resampled_idm_best = resampled_idm

        if episode == 0:
            print(f'\n{demo_type} {demo_name[n]} {remap_type}')

        print(f'MAE of idm_perform, episode {episode}: {round(MAE[episode], 1)}')

        ax1_all.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g')
        ax2_all.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g')

        if flag_plot_all:
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection='3d')
            # ax1.plot(states_demo[0, 0], states_demo[0, 1], states_demo[0, 2], color='g', marker='o', label='rest position')
            # ax1.plot(states_demo[:9, 0], states_demo[:9, 1], states_demo[:9, 2], color='b', label='added')
            # ax1.plot(states_demo[8:, 0], states_demo[8:, 1], states_demo[8:, 2], color='r', label='demonstration')
            ax1.plot(states_demo[:, 0], states_demo[:, 1], states_demo[:, 2], color='r', label='demonstration')
            ax1.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g', label='idm')
            ax1.set_title('Proximal segment')
            ax1.legend()
            ax1.set_xlabel("x", fontsize=15)
            ax1.set_ylabel("y", fontsize=15)
            ax1.set_zlabel("z", fontsize=15)
            ax2 = fig.add_subplot(122, projection='3d')
            # ax2.plot(states_demo[0, 3], states_demo[0, 4], states_demo[0, 5], color='g', marker='o', label='rest position')
            # ax2.plot(states_demo[:9, 3], states_demo[:9, 4], states_demo[:9, 5], color='b', label='added')
            # ax2.plot(states_demo[8:, 3], states_demo[8:, 4], states_demo[8:, 5], color='r', label='demonstration')
            ax2.plot(states_demo[:, 3], states_demo[:, 4], states_demo[:, 5], color='r', label='demonstration')
            ax2.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g', label='idm')
            ax2.set_title('Distal segment')
            ax2.legend()
            ax2.set_xlabel("x", fontsize=15)
            ax2.set_ylabel("y", fontsize=15)
            ax2.set_zlabel("z", fontsize=15)
            fig = plt.gcf()
            # fig.suptitle(f'Demonstration and added path to reach its initial point: {demo_name[n]}')
            fig.suptitle(f"Demonstration number {episode}\nMAE of distal segment: {round(MAE[episode], 3)}")
            plt.show()

        episode += 1

    print(f'Best episode for {demo_name[n]} movement: {episode_best}')
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(states_demo_best[:, 0], states_demo_best[:, 1], states_demo_best[:, 2], color='r', label='demonstration')
    ax1.plot(resampled_idm_best[:, 0], resampled_idm_best[:, 1], resampled_idm_best[:, 2], color='g', label='idm')
    ax1.set_title('Proximal segment')
    ax1.legend()
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.set_zlabel("z", fontsize=15)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(states_demo_best[:, 3], states_demo_best[:, 4], states_demo_best[:, 5], color='r', label='demonstration')
    ax2.plot(resampled_idm_best[:, 3], resampled_idm_best[:, 4], resampled_idm_best[:, 5], color='g', label='idm')
    ax2.set_title('Distal segment')
    ax2.legend()
    ax2.set_xlabel("x", fontsize=15)
    ax2.set_ylabel("y", fontsize=15)
    ax2.set_zlabel("z", fontsize=15)
    fig = plt.gcf()
    fig.suptitle(f"Best demonstration for {demo_name[n]} motion (episode {episode_best+1})\n"
                 f"MAE of distal segment: {round(MAE[episode_best], 3)}")
    plt.show()
