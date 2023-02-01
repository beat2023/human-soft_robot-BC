# Modify the data taken, so that it is referred to the origin

import os
import sys
import pandas as pd
from io import StringIO
import numpy as np
from functions import resampling
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter, argrelextrema

########################################################################################################################

demo_type_list = ['robot', 'human']
demo_name_list = ['circle', 'loop', 'cross']
len_demo = [40, 35, 40]

for d in range(len(demo_type_list)):
    demo_type = demo_type_list[d]

    path_origin = 'DEMONSTRATIONS/' + demo_type + '/origin.txt'  # The file containing the origin
    file_origin = open(path_origin, 'r').read()
    file_origin = file_origin.replace('[', "").replace(']', "").split(' ')
    origin = []
    for el in file_origin:
        if el != '':
            origin.append(float(el))
    origin = np.array(origin)
    origin_repeated = np.append(origin, origin)

    for n in range(len(demo_name_list)):
        demo_name = demo_name_list[n]

        rev = 0  # initialize number of episode
        while True:
            file_name = demo_name + '_' + str(rev)
            path = 'DEMONSTRATIONS/' + demo_type + '/raw_data/' + file_name + '.txt'
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

            markers_or = df.loc[:, "pmx":].to_numpy()

            # Shift w.r.t. the origin
            markers_shift = markers_or - origin_repeated
            # Savitzky Golay filter to smooth the signal
            for m in range(markers_shift.shape[1]):
                markers_shift[:, m] = savgol_filter(markers_shift[:, m], 10, 3)  # window size 10, polynomial order 3
            # Eliminate states in which the robot is stuck
            rows2el = []
            for i in range(markers_shift.shape[0] - 1):
                if sum(abs(markers_shift[i+1, :] - markers_shift[i, :])) < 15:
                    rows2el.append(i+1)
            markers_shift = np.delete(markers_shift, rows2el, axis=0)

            # If the demo type is loop, I trunkate in the end of one loop
            if demo_name == 'loop':
                # Local minima
                if demo_type == 'human':
                    local_min = argrelextrema(markers_shift[:, 2], np.less)[0]
                else:
                    local_min = argrelextrema(markers_shift[:, 5], np.less)[0]
                local_min = local_min.tolist()
                if len(local_min) == 3:
                    markers_shift = markers_shift[local_min[0]-1:local_min[2]+2, :]
                elif len(local_min) == 2:
                    if local_min[1] > markers_shift.shape[0]*2/3:
                        markers_shift = markers_shift[:local_min[1]+2, :]
                    elif local_min[0] < markers_shift.shape[0]/3:
                        markers_shift = markers_shift[local_min[0]-1:, :]
                elif len(local_min) > 3:
                    if local_min[2] > markers_shift.shape[0]*2/3:
                        markers_shift = markers_shift[:local_min[2]+2, :]
                    else:
                        markers_shift = markers_shift[local_min[1]-1:local_min[3]+2, :]
                # To avoid having many coincident points
                markers_shift = markers_shift[2:-2, :]

            # Resample so that all demonstrations have the same number of points
            markers_shift = resampling(markers_shift, len_demo[n])

            if rev == 0:
                fig, axs = plt.subplots(6, 2)
                fig.suptitle(f'Example of preprocessing of {demo_type} {demo_name}')
                axs[0, 0].title.set_text('Before pre-processing')
                axs[0, 1].title.set_text('After pre-processing')
                for m in range(markers_shift.shape[1]):
                    axs[m, 0].plot(markers_or[:, m])
                    axs[m, 1].plot(markers_shift[:, m])
                    axs[m, 0].set(xlabel='Sample number')
                    axs[m, 0].set(xlabel='Sample number')
                axs[0, 0].set(ylabel='x proximal')
                axs[1, 0].set(ylabel='y proximal')
                axs[2, 0].set(ylabel='z proximal')
                axs[3, 0].set(ylabel='x distal')
                axs[4, 0].set(ylabel='y distal')
                axs[5, 0].set(ylabel='z distal')
                axs[0, 0].xaxis.set_ticklabels([])
                axs[1, 0].xaxis.set_ticklabels([])
                axs[2, 0].xaxis.set_ticklabels([])
                axs[3, 0].xaxis.set_ticklabels([])
                axs[4, 0].xaxis.set_ticklabels([])
                axs[5, 0].set(xlabel='Sample number')
                axs[0, 1].xaxis.set_ticklabels([])
                axs[1, 1].xaxis.set_ticklabels([])
                axs[2, 1].xaxis.set_ticklabels([])
                axs[3, 1].xaxis.set_ticklabels([])
                axs[4, 1].xaxis.set_ticklabels([])
                axs[5, 1].set(xlabel='Sample number')
                plt.show()

            ############################################################################################################
            # WRITE MODIFIED DATA INTO A NEW FILE
            # file_wname = 'DEMONSTRATIONS/' + demo_type + '/origin_ref/shift_' + file_name + '.txt'
            # with open(file_wname, 'w') as f:
            #     for m in range(markers_shift.shape[0]):
            #         f.write(str([m, markers_shift[m, :]]).strip('[]'))
            #         f.write('\n')

            # Increment the episode number
            rev += 1
