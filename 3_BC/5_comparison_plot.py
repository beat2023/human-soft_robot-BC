# PLOT ROBOT AND HUMAN TRAJECTORIES TOGETHER

from io import StringIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

########################################################################################################################

demo_name_list = ['circle', 'loop', 'cross']
norm_type_list = ['minMax', 'norm', 'zScore', 'confInt', 'ellipsoid']

for n in range(len(demo_name_list)):

    demo_name = demo_name_list[n]
    file_name_robot_avg = 'DEMONSTRATIONS/robot_remapped/ellipsoid/shift_' + demo_name + '_avg.txt'
    file_name_robot_std = 'DEMONSTRATIONS/robot_remapped/ellipsoid/shift_' + demo_name + '_std.txt'

    # Robot avg trajectory
    file = open(file_name_robot_avg, 'r').read()
    file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
    data = StringIO(file_1)
    df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
    robot_avg = (df.loc[:, "pmx":].to_numpy())

    # Robot std trajectory
    file = open(file_name_robot_std, 'r').read()
    file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
    data = StringIO(file_1)
    df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
    robot_std = (df.loc[:, "pmx":].to_numpy())

    human_avg_all = []
    human_std_all = []
    for t in range(len(norm_type_list)):

        norm_type = norm_type_list[t]
        file_name_human_avg = 'DEMONSTRATIONS/human_remapped/' + norm_type + '/shift_' + demo_name + '_avg.txt'
        file_name_human_std = 'DEMONSTRATIONS/human_remapped/' + norm_type + '/shift_' + demo_name + '_std.txt'

        # Human avg trajectories
        file = open(file_name_human_avg, 'r').read()
        file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
        data = StringIO(file_1)
        df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
        human_avg = (df.loc[:, "pmx":].to_numpy())
        human_avg_all.append(human_avg)

        # Human std trajectory
        file = open(file_name_human_std, 'r').read()
        file_1 = file.replace(',\n', ",").replace('])', "").replace('array([', "").replace('[', "").replace(']', "")
        data = StringIO(file_1)
        df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
        human_std = (df.loc[:, "pmx":].to_numpy())
        human_std_all.append(human_std)

    fig, axs = plt.subplots(6, len(human_avg_all))
    fig.suptitle(f'Comparison robot-human expert trajectories: {demo_name} movement\n'
                 f'Robot data remapped with "ellipsoid" method')
    for t in range(len(norm_type_list)):
        axs[0, t].title.set_text(f'{norm_type_list[t]}')
        for m in range(robot_avg.shape[1]):
            axs[m, t].plot(robot_avg[:, m], '-', color='k', linewidth=1.5, label='robot')
            axs[m, t].plot(robot_avg[:, m] + robot_std[:, m], '--', color='k', linewidth=0.8)
            axs[m, t].plot(robot_avg[:, m] - robot_std[:, m], '--', color='k', linewidth=0.8)
            axs[m, t].plot(human_avg_all[t][:, m], '-', color='r', linewidth=1.5, label='human')
            axs[m, t].plot(human_avg_all[t][:, m] + human_std_all[t][:, m], '--', color='r', linewidth=0.8)
            axs[m, t].plot(human_avg_all[t][:, m] - human_std_all[t][:, m], '--', color='r', linewidth=0.8)
            if m == 0 and t == 0:
                fig.legend(loc='center right')
        axs[0, t].set(ylabel='x proximal')
        axs[1, t].set(ylabel='y proximal')
        axs[2, t].set(ylabel='z proximal')
        axs[3, t].set(ylabel='x distal')
        axs[4, t].set(ylabel='y distal')
        axs[5, t].set(ylabel='z distal')
        axs[5, t].set(xlabel='Sample number')
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

