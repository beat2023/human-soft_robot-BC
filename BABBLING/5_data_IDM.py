# Code to test the babbling data, needed for the IDM

import matlab.engine
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np
import time
from time import sleep


def set_sup(pressure_array):
    """check if pressures in range [0, 1]"""
    # min_pressure : 0.0 bar
    # max_pressure : 1.0 bar
    pressure_array = np.around(pressure_array)
    pressure_array = np.array(pressure_array, dtype='int')
    d1 = int(pressure_array[0])  # 0 - 160
    d2 = int(pressure_array[1])  # 0 - 160
    d3 = int(pressure_array[2])  # 0 - 160
    d4 = int(pressure_array[3])  # 0 - 160
    d5 = int(pressure_array[4])  # 0 - 52
    d6 = int(pressure_array[5])  # 0 - 160

    # Channels 1,2,3,4,6
    DATA_temp1 = [d1, d2, d3, d4, d6]
    while min(DATA_temp1) < 0:
        min_index = np.argmin(DATA_temp1)
        DATA_temp1[min_index] = 0
    while max(DATA_temp1) > 160:
        max_index = np.argmax(DATA_temp1)
        DATA_temp1[max_index] = 160
    # Channel 5
    DATA_temp2 = d5
    if DATA_temp2 < 0:
        DATA_temp2 = 0
    elif DATA_temp2 > 52:
        DATA_temp2 = 52

    DATA_write = np.array([106, DATA_temp1[0], DATA_temp1[1], DATA_temp1[2], DATA_temp1[3], 0,
                           DATA_temp1[4], DATA_temp2, 0, 0, 0], dtype=np.uint8)
    DATA_save = [DATA_temp1[0], DATA_temp1[1], DATA_temp1[2], DATA_temp1[3], DATA_temp2, DATA_temp1[4]]
    return DATA_write, DATA_save


#######################################################################################################################
# Load pressure data and plot it

for_plot = []
file_name = 'sinusoids_0'
file_agent1 = open('DataFolder/' + file_name + '.txt', 'r').read()
data_agent1 = file_agent1.split('\n')

prev_d1, prev_d2, prev_d3, prev_d4, prev_d5, prev_d6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
diff = []
for line in data_agent1:
    if len(line) > 10:
        count, d1, d2, d3, d4, d5, d6 = line.split(', ')

        pres_temp = [float(d1), float(d2), float(d3), float(d4), float(d5), float(d6)]
        diff_temp = [prev_d1 - float(d1), prev_d2 - float(d2), prev_d3 - float(d3),
                     prev_d4 - float(d4), prev_d5 - float(d5), prev_d6 - float(d6)]
        prev_d1, prev_d2, prev_d3, prev_d4, prev_d5, prev_d6 = float(d1), float(d2), float(d3), float(d4), float(d5), \
                                                               float(d6)
        for_plot.append([pres_temp])
        diff.append([diff_temp])

for_plot = (np.array(for_plot)).reshape(len(for_plot), 6)
diff = (np.array(diff)).reshape(len(diff), 6)
print('Dimension of the input dataset: ', for_plot.shape)
fig_1, axs = plt.subplots(3, 1)
fig_1.suptitle('Pressures proximal segment')
axs[0].plot(for_plot[:, 0], 'r')
axs[1].plot(for_plot[:, 1], 'g')
axs[2].plot(for_plot[:, 2], 'k')
fig_2, axs2 = plt.subplots(3, 1)
fig_2.suptitle('Pressures distal segment')
axs2[0].plot(for_plot[:, 3], 'g')
axs2[1].plot(for_plot[:, 4], 'k')
axs2[2].plot(for_plot[:, 5], 'r')
fig_3, axs3 = plt.subplots(3, 1)
axs3[0].plot(diff[:, 0], 'r')
axs3[1].plot(diff[:, 1], 'g')
axs3[2].plot(diff[:, 2], 'k')
fig_4, axs4 = plt.subplots(3, 1)
axs4[0].plot(diff[:, 3], 'g')
axs4[1].plot(diff[:, 4], 'k')
axs4[2].plot(diff[:, 5], 'r')
plt.show()

########################################################################################################################

# ################### write on Matlab: matlab.engine.shareEngine ##################### #

eng = matlab.engine.start_matlab()
s = eng.genpath('../Matlab_communication')
eng.addpath(s, nargout=0)
eng.initialize_system()

# find the used comport first
comlist = serial.tools.list_ports.comports()
connected = []
for element in comlist:
    connected.append(element.device)
PORT_ID = connected[0]
dev = serial.Serial(str(PORT_ID), baudrate=115200)

print("Connected COM ports: " + str(PORT_ID))
print("Is port open? ", dev.isOpen())

# Frequency and time of acquisition
n_samples = for_plot.shape[0]
rateAcq = 5  # sampling frequency [Hz]
sampl_T = 1 / rateAcq
sec = n_samples * sampl_T

# Initialization with zero values
DATA_to_write, _ = set_sup(pressure_array=[0, 0, 0, 0, 0, 0])
for ii in range(5):
    dev.write(bytearray(DATA_to_write))
    sleep(sampl_T)

# File where to write data
file_wname = 'DataFolder/raw_data/idm_' + file_name + '.txt'

with open(file_wname, 'w') as f:
    time_record = time.time()
    time_start = time.time()
    print('Start of acquisition')
    print(f'Expected time required: {sec} seconds')

    for d in range(for_plot.shape[0]):

        pos = eng.positions()  # record markers position
        pos2array = np.array(pos)
        pos2array = np.reshape(pos2array, (6,))

        DATA_to_write, DATA_to_save = set_sup(pressure_array=for_plot[d, :])
        dev.write(bytearray(DATA_to_write))

        # Write the action and observed state in the text file
        pressure_data = DATA_to_save
        EE1_marks = pos2array[:3]  # proximal
        EE2_marks = pos2array[3:]  # distal
        f.write(str([d, pressure_data, EE1_marks, EE2_marks]).strip('[]'))
        f.write('\n')

        time_to_sleep = sampl_T - (time.time() - time_record)
        if np.sign(time_to_sleep) != -1:
            sleep(time_to_sleep)
        time_record = time.time()

    time_stop = time.time()
    print('End of acquisition')
    print(f'Total acquisition time: {time_stop - time_start}, instead of {sec} seconds')

DATA_to_write, _ = set_sup(pressure_array=[0, 0, 0, 0, 0, 0])
for ii in range(5):
    dev.write(bytearray(DATA_to_write))
    sleep(sampl_T)

# Release communication with Matlab
eng.release_system()
eng.quit()
