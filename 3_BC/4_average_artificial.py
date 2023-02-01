# Load all demonstrations and average them to save the "average trajectory" and their standard deviation
# Then use these parameters to infer artificial data (gaussian distribution is used)

import pandas as pd
import numpy as np
from io import StringIO
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from functions import normalize

########################################################################################################################
# AVERAGE AND STANDARD DEVIATION CALCULATED

# demo_type_list = ['robot', 'human']
demo_type_list = ['human']
demo_name_list = ['circle', 'loop', 'cross']
# dirs = ['minMax', 'norm', 'zScore', 'confInt', 'ellipsoid']
dirs = ['ellipsoid']

artificial_creation = False   # if I want to create artificial demonstrations

for t in range(len(demo_type_list)):

    demo_type = demo_type_list[t]

    for d in range(len(dirs)):
        path_dir = 'DEMONSTRATIONS/' + demo_type + '_remapped/' + dirs[d] + '/'
        for n in range(len(demo_name_list)):
            demo_name = demo_name_list[n]
            markers_all = []
            rev = 0
            while True:
                path = path_dir + 'shift_' + demo_name + '_' + str(rev) + '.txt'
                if not os.path.isfile(path):
                    # If the file doesn't exist I exit the loop
                    break
                file = open(path, 'r').read()
                file_1 = file.replace(',\n', ",")
                file_1 = file_1.replace('])', "")
                file_1 = file_1.replace('array([', "")
                file_1 = file_1.replace('[', "")
                file_1 = file_1.replace(']', "")
                data = StringIO(file_1)
                df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
                markers = (df.loc[:, "pmx":].to_numpy())
                markers_all.append(markers)
                rev += 1

            markers_all = np.array(markers_all)

            mrk_avg = np.zeros(markers_all[0].shape)
            mrk_std = np.zeros(markers_all[0].shape)
            for j in range(markers_all.shape[1]):
                mrk_avg[j, :] = np.mean(markers_all[:, j, :], axis=0)
            for j in range(markers_all.shape[1]):
                mrk_std[j, :] = np.std(markers_all[:, j, :], axis=0)

            episode2el = []

            fig, axs = plt.subplots(6, 2)
            fig.suptitle(f'Before removal: {demo_type} {demo_name} {dirs[d]}')
            axs[0, 0].title.set_text('All episodes')
            axs[0, 1].title.set_text('Average and standard deviation')
            for s in range(markers_all.shape[0]):
                for m in range(markers_all.shape[2]):
                    axs[m, 0].plot(markers_all[s, :, m], '--', linewidth=0.3)
                    if any(abs(markers_all[s, :, m] - mrk_avg[:, m]) > 3 * mrk_std[:, m]):
                        episode2el.append(s)
            axs[0, 0].set(ylabel='x proximal')
            axs[1, 0].set(ylabel='y proximal')
            axs[2, 0].set(ylabel='z proximal')
            axs[3, 0].set(ylabel='x distal')
            axs[4, 0].set(ylabel='y distal')
            axs[5, 0].set(ylabel='z distal')
            for m in range(markers_all.shape[2]):
                axs[m, 1].plot(mrk_avg[:, m], '-', color='k', linewidth=1.5, label='avg')
                axs[m, 1].plot(mrk_avg[:, m] + mrk_std[:, m], '--', color='k', linewidth=0.8, label='std')
                axs[m, 1].plot(mrk_avg[:, m] - mrk_std[:, m], '--', color='k', linewidth=0.8)
            for ax in axs.flat:
                ax.set(xlabel='Sample number')
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
            plt.legend()
            plt.show()

            # Calculate the new avg and std without the outlier trajectories
            if len(episode2el) > 0:
                episode2el = np.array(episode2el)
                ep2el_unique = np.unique(episode2el)
                print(f'Episodes to eliminate for {demo_type} {demo_name} {dirs[d]}: {ep2el_unique}')
                markers_all = np.delete(markers_all, ep2el_unique, axis=0)

                mrk_avg = np.zeros(markers_all[0].shape)
                mrk_std = np.zeros(markers_all[0].shape)
                for j in range(markers_all.shape[1]):
                    mrk_avg[j, :] = np.mean(markers_all[:, j, :], axis=0)
                for j in range(markers_all.shape[1]):
                    mrk_std[j, :] = np.std(markers_all[:, j, :], axis=0)

                fig, axs = plt.subplots(6, 2)
                fig.suptitle(f'After removal: {demo_type} {demo_name} {dirs[d]}')
                axs[0, 0].title.set_text('All episodes')
                axs[0, 1].title.set_text('Average and standard deviation')
                for s in range(markers_all.shape[0]):
                    for m in range(markers_all.shape[2]):
                        axs[m, 0].plot(markers_all[s, :, m], '--', linewidth=0.3)
                axs[0, 0].set(ylabel='x proximal')
                axs[1, 0].set(ylabel='y proximal')
                axs[2, 0].set(ylabel='z proximal')
                axs[3, 0].set(ylabel='x distal')
                axs[4, 0].set(ylabel='y distal')
                axs[5, 0].set(ylabel='z distal')
                for m in range(markers_all.shape[2]):
                    axs[m, 1].plot(mrk_avg[:, m], '-', color='k', linewidth=1.5, label='avg')
                    axs[m, 1].plot(mrk_avg[:, m] + mrk_std[:, m], '--', color='k', linewidth=0.8, label='std')
                    axs[m, 1].plot(mrk_avg[:, m] - mrk_std[:, m], '--', color='k', linewidth=0.8)
                for ax in axs.flat:
                    ax.set(xlabel='Sample number')
                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()
                plt.legend()
                plt.show()
            else:
                print('No episodes to eliminate')

            ############################################################################################################
            # SAVE AVG AND STD FILES

            # file_name_avg = path_dir + 'shift_' + demo_name + '_avg.txt'
            # file_name_std = path_dir + 'shift_' + demo_name + '_std.txt'
            # os.makedirs(os.path.dirname(file_name_avg), exist_ok=True)
            # os.makedirs(os.path.dirname(file_name_std), exist_ok=True)
            # with open(file_name_avg, 'w') as f:
            #     for j in range(mrk_avg.shape[0]):
            #         f.write(str([j, mrk_avg[j, :]]).strip('[]'))
            #         f.write('\n')
            # with open(file_name_std, 'w') as f:
            #     for j in range(mrk_std.shape[0]):
            #         f.write(str([j, mrk_std[j, :]]).strip('[]'))
            #         f.write('\n')

            ############################################################################################################
            # ARTIFICIAL DEMOS FROM THE EXTRACTED DISTRIBUTION

            if artificial_creation:
                n_artificial = 10   # the number of artificial demonstrations I want to create
                artificial_data = np.zeros((n_artificial, markers_all.shape[1], markers_all.shape[2]))
                art_data_smoothed = np.zeros((n_artificial, markers_all.shape[1], markers_all.shape[2]))

                # fig, axs = plt.subplots(6, 1)
                # fig.suptitle(f'Artificial {demo_type} {demo_name} {dirs[d]}')

                for i in range(n_artificial):
                    for j in range(artificial_data.shape[1]):
                        artificial_data[i, j, :] = np.random.normal(mrk_avg[j, :], mrk_std[j, :])
                    for m in range(artificial_data.shape[2]):
                        # window size 15, polynomial order 3
                        art_data_smoothed[i, :, m] = savgol_filter(artificial_data[i, :, m], 13, 3)

                    # Save the artificial data into new files
                    file_name = path_dir + 'shift_' + demo_name + '_artificial_' + str(i) + '.txt'
                    os.makedirs(os.path.dirname(file_name), exist_ok=True)
                    with open(file_name, 'w') as f:
                        for k in range(art_data_smoothed.shape[1]):
                            f.write(str([k, art_data_smoothed[i, k, :]]).strip('[]'))
                            f.write('\n')

                #     for m in range(art_data_smoothed.shape[2]):
                #         axs[m].plot(mrk_avg[:, m], '-', color='k', linewidth=0.3, label='avg')
                #         axs[m].plot(mrk_avg[:, m] + mrk_std[:, m], '--', color='k', linewidth=0.3, label='std')
                #         axs[m].plot(mrk_avg[:, m] - mrk_std[:, m], '--', color='k', linewidth=0.3)
                #         axs[m].plot(art_data_smoothed[i, :, m], '-', color='r', linewidth=0.3, label='artificial')
                #     if i == 0:
                #         plt.legend()
                #
                # axs[0].set(ylabel='x proximal')
                # axs[1].set(ylabel='y proximal')
                # axs[2].set(ylabel='z proximal')
                # axs[3].set(ylabel='x distal')
                # axs[4].set(ylabel='y distal')
                # axs[5].set(ylabel='z distal')
                # for ax in axs.flat:
                #     ax.set(xlabel='Sample number')
                # # Hide x labels and tick labels for top plots and y ticks for right plots.
                # for ax in axs.flat:
                #     ax.label_outer()
                # plt.show()
