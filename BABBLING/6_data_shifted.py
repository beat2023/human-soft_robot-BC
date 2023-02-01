# Modify the data taken, so that it is referred to the origin

import os
import pandas as pd
from io import StringIO
import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################

# folder_name = 'DataFolder_human' or 'DataFolder'
folder_name = 'DataFolder'

########################################################################################################################

# ORIGIN
file_origin = open(folder_name + '/origin.txt', 'r').read()
file_origin = file_origin.replace('[', "").replace(']', "").split(' ')
origin = []
for el in file_origin:
    if el != '':
        origin.append(float(el))
origin = np.array(origin)
origin_repeated = np.append(origin, origin)

# DIRECTORY WITH ALL RAW DATA
path_dir = folder_name + '/raw_data'
files_names = os.listdir(path_dir)
files_path = []
for i in range(len(files_names)):
    files_path.append(path_dir + '/' + files_names[i])

for i in range(len(files_names)):
    file = open(files_path[i], 'r').read().replace('array([', "").replace('])', "").replace('[', "").replace(']', "")
    data = StringIO(file)
    if folder_name == 'DataFolder':
        df = pd.read_csv(data, sep=",",
                         names=["count", "P1", "P2", "P3", "P4", "P5", "P6",
                                         "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])
        pressures = df.loc[:, "P1":"P6"].to_numpy()
    elif folder_name == 'DataFolder_human':
        df = pd.read_csv(data, sep=",", names=["count", "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])

    markers = df.loc[:, "pmx":].to_numpy()
    markers_shift = markers - origin_repeated

    ####################################################################################################################
    # PLOT TOGETHER

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # proximal segment
    ax1.plot(markers_shift[:, 0], markers_shift[:, 1], markers_shift[:, 2], color='r', label='shifted')
    ax1.plot(markers[:, 0], markers[:, 1], markers[:, 2], color='b', label='original')
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.title.set_text('Proximal segment')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # distal segment
    ax2.plot(markers_shift[:, 3], markers_shift[:, 4], markers_shift[:, 5], color='r', label='shifted')
    ax2.plot(markers[:, 3], markers[:, 4], markers[:, 5], color='b', label='original')
    ax2.legend()
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.title.set_text('Distal segment')
    fig = plt.gcf()
    fig.suptitle(f"Demo name: {files_names[i]}")
    plt.show()

    ####################################################################################################################
    # WRITE MODIFIED DATA INTO A NEW FILE

    # File where to write data
    file_wname = folder_name + '/origin_ref/shift_' + files_names[i]

    with open(file_wname, 'w') as f:
        for d in range(markers_shift.shape[0]):
            if folder_name == 'DataFolder':
                f.write(str([d, pressures[d, :].tolist(), markers_shift[d, :]]).strip('[]'))
            elif folder_name == 'DataFolder_human':
                f.write(str([d, markers_shift[d, :]]).strip('[]'))
            f.write('\n')

