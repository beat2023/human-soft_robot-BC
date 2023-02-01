# Code to test the babbling data, needed for the IDM

import matlab.engine
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import struct
import time
from time import sleep

########################################################################################################################
# ################### write on Matlab: matlab.engine.shareEngine ##################### #

print('Starting communication with MATLAB')
eng = matlab.engine.start_matlab()
s = eng.genpath('../Matlab_communication')
eng.addpath(s, nargout=0)
eng.initialize_system()
print('Connection established')


# Frequency and time of acquisition
rateAcq = 5  # sampling frequency [Hz]
sampl_T = 1 / rateAcq
sec = 120
n_samples = sec * rateAcq

# File where to write data
file_wname = 'DataFolder_human/raw_data/data.txt'

with open(file_wname, 'w') as f:
    positions_all = []
    print(f'Start of acquisition of human babbling data in 2 second')
    time.sleep(2)
    print('Started')
    time_record = time.time()
    time_start = time.time()
    for d in range(n_samples):
        if d % (rateAcq*10) == 0:
            print(f'Remaining seconds of acquisition {sec - d / rateAcq}')
        pos = eng.positions()
        pos2array = np.array(pos)
        pos2array = np.reshape(pos2array, (6,))
        positions_all.append(pos)
        f.write(str([d, pos2array]).strip('[]'))
        f.write('\n')
        time_to_sleep = sampl_T - (time.time() - time_record)
        if np.sign(time_to_sleep) != -1:
            time.sleep(time_to_sleep)
        time_record = time.time()
    time_stop = time.time()
    print('End of acquisition')
    print(f'Total acquisition time: {time_stop - time_start}, instead of {sec} seconds')

# PLOTTING THE DATA ACQUIRED
positions_all = np.array(positions_all)
positions_all = np.reshape(positions_all, [positions_all.shape[0], 6])
fig = plt.figure
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
x = positions_all[:, [0, 3]]
y = positions_all[:, [1, 4]]
z = positions_all[:, [2, 5]]
ax.scatter3D(x[:, 0], y[:, 0], z[:, 0], marker='*', s=1, label='proximal')
ax.scatter3D(x[:, 1], y[:, 1], z[:, 1], marker='o', s=1, label='distal')
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)
ax.legend(loc=1)
plt.show()

# Release communication with Matlab
eng.release_system()
eng.quit()
