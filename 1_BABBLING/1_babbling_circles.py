# BABBLING CIRCLES

import numpy as np
from matplotlib import pyplot as plt
from random import randint

step = 16

data = []
delay = 4


# for v in range(1, len(values)):  # each time a different max pressure is reached
#     for t in range(1):  # repeat the circle of same radius 1 times
#         for i in range(v+1):
#             data.append(values[i])
#         for i in range(delay):
#             data.append(values[v])
#         for i in range(v):
#             data.append(values[v-i])
#         for i in range(delay):
#             data.append(values[0])
# for v in range(len(values)-2, 0, -1):  # inverse procedure
#     for t in range(1):  # repeat the circle of same radius 1 times
#         for i in range(v+1):
#             data.append(values[i])
#         for i in range(delay):
#             data.append(values[v])
#         for i in range(v):
#             data.append(values[v-i])
#         for i in range(delay):
#             data.append(values[0])

for v in range(70):  # repeat many times a random choice of height
    max_val = randint(1, 160)
    vals = np.arange(0, max_val, step)

    for t in range(2):  # repeat the circle of same radius 2 times
        for i in range(len(vals)-1):
            data.append(vals[i])
        for i in range(delay):
            data.append(vals[-1])
        for i in range(len(vals)-1, 0, -1):
            data.append(vals[i])
        for i in range(delay):
            data.append(vals[0])

for i in range(5*delay):
    data.append(0.0)

array_all = np.zeros((len(data), 6))
array_all[:, 0] = np.array(data)
for chamb in range(5):
    data_trasl = []
    delay_chamb = (chamb+1)*delay
    for i in range(delay_chamb):
        data_trasl.append(0)
    for i in range(len(data)-delay_chamb):
        data_trasl.append(data[i])
    array_all[:, chamb+1] = np.array(data_trasl)


fig, axs = plt.subplots()
axs.plot(array_all[:, 0]/160, 'r', label='chamber 1')
axs.plot(array_all[:, 1]/160, 'g', label='chamber 2')
axs.plot(array_all[:, 2]/160, 'y', label='chamber 3')
axs.plot(array_all[:, 3]/160, 'b', label='chamber 4')
axs.plot(array_all[:, 4]/160, 'c', label='chamber 5')
axs.plot(array_all[:, 5]/160, 'k', label='chamber 6')
plt.xlabel('Sample number')
plt.ylabel('Pressure value (Bars)')
plt.legend()
plt.title('Babbling: circles')
plt.show()

########################################################################################################################
# SAVE DATA INTO FILE
# clockwise = [1, 5, 2, 6, 3, 4]  # the order in which I want to activate the chambers to achieve a circle motion
# counterclockwise = [1, 4, 3, 6, 2, 5]  # the order in which I want to activate the chambers to achieve a circle motion
#
# rev = 0
# file_wname = 'DataFolder/circles_' + str(rev) + '.txt'
# with open(file_wname, 'w') as f:
#     for d in range(array_all.shape[0]):
#         if d < array_all.shape[0]/2:
#             f.write(str([d, array_all[d, 0], array_all[d, 2], array_all[d, 4],
#                          array_all[d, 5], (array_all[d, 1]/160)*52, array_all[d, 3]]).strip('[]'))
#             f.write('\n')
#         else:
#             f.write(str([d, array_all[d, 0], array_all[d, 4], array_all[d, 2],
#                          array_all[d, 1], (array_all[d, 5] / 160) * 52, array_all[d, 3]]).strip('[]'))
#             f.write('\n')
