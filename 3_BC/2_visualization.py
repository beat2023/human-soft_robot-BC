"""Visualize the demonstrations in 3D"""

import os
from io import StringIO
import pandas as pd
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np

# demo_type = 'human' or 'robot'
demo_type = 'human'
# demo_name = 'circle', 'loop' or 'cross'
demo_name = 'cross'
# dir_name = 'raw_data/' or 'origin_ref/shift_'
dir_name = 'origin_ref/shift_'

########################################################################################################################

episode = 0

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

if dir_name == 'raw_data/':
    col = 'r'
    type_dem = 'raw'
elif dir_name == 'origin_ref/shift_':
    col = 'b'
    type_dem = 'pre-processed'

while True:
    path = 'DEMONSTRATIONS/' + demo_type + '/' + dir_name + demo_name + '_' + str(episode) + '.txt'
    if not os.path.isfile(path):
        # If the file doesn't exist I exit the loop
        break
    data = open(path, 'r').read()
    data = data.replace(',\n', ",")
    data = data.replace('])', "")
    data = data.replace('array([', "")
    data = StringIO(data)
    df2 = pd.read_csv(data, sep=",", names=["i", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
    states_demo = df2.loc[:, "pmx":"dmz"].to_numpy()

    ax1.plot(states_demo[:, 0], states_demo[:, 1], states_demo[:, 2], color=col)
    ax1.set_title('Proximal segment')
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.set_zlabel("z", fontsize=15)
    ax2.plot(states_demo[:, 3], states_demo[:, 4], states_demo[:, 5], color=col)
    ax2.set_title('Distal segment')
    ax2.set_xlabel("x", fontsize=15)
    ax2.set_ylabel("y", fontsize=15)
    ax2.set_zlabel("z", fontsize=15)
    fig = plt.gcf()
    fig.suptitle(f"Demonstrations {type_dem}: {demo_type} {demo_name}")

    episode += 1

plt.show()
