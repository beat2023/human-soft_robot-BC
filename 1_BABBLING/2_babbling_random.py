# BABBLING RANDOM TRAJECTORIES

import numpy as np
from matplotlib import pyplot as plt
from random import randint, uniform

n_steps = 10
values = np.linspace(0, 160, num=n_steps)
step = 160./n_steps
data = []

array_all = np.zeros((2500, 6))

for i in range(1, 2500):  # repeat many times a random choice of height
    for j in range(6):
        prev_val = array_all[i-1, j]
        inf = prev_val-step
        if inf < 0:
            inf = 0
        sup = prev_val+step
        if sup > values[-1]:
            sup = values[-1]
        vals = [inf, prev_val, sup]
        random_index = randint(0, 2)
        array_all[i, j] = vals[random_index]


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
plt.title('Babbling: random')
plt.show()

########################################################################################################################
# SAVE DATA INTO FILE

# rev = 0
# file_wname = 'DataFolder/random_' + str(rev) + '.txt'
# with open(file_wname, 'w') as f:
#     for d in range(array_all.shape[0]):
#         f.write(str([d, array_all[d, 0], array_all[d, 1], array_all[d, 2],
#                      array_all[d, 3], (array_all[d, 4]/160)*52, array_all[d, 5]]).strip('[]'))
#         f.write('\n')
