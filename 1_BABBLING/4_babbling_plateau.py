# BABBLING CIRCLES

import numpy as np
from matplotlib import pyplot as plt
from random import randint

step = 8

delay = 10
chamb_all = []

for chamb in range(6):

    prev_plat = 0
    chamb_vals = []
    for v in range(120):  # repeat many times a random choice of height
        plat_flag = False
        while not plat_flag:
            plat = randint(0, 160)
            if abs(plat - prev_plat) > 50:
                plat_flag = True
        sign = 1
        if plat - prev_plat < 0:
            sign = -1
        vals = np.arange(prev_plat, plat, sign*step)

        for i in range(len(vals)):
            chamb_vals.append(vals[i])
        for i in range(delay):
            chamb_vals.append(vals[-1])
        prev_plat = vals[-1]

    chamb_vals = np.array(chamb_vals)
    chamb_all.append(chamb_vals)

ok = False
while not ok:
    for chamb in range(6):
        if chamb_all[chamb][-1] != 0:
            chamb_all[chamb] = np.append(chamb_all[chamb], chamb_all[chamb][-1] - step)
            if chamb_all[chamb][-1] < 0:
                chamb_all[chamb][-1] = 0
    n_ok = 0
    for chamb in range(6):
        if chamb_all[chamb][-1] == 0:
            n_ok += 1
    if n_ok == 6:
        ok = True

max_len = 0
for chamb in range(6):
    if len(chamb_all[chamb]) > max_len:
        max_len = len(chamb_all[chamb])
for chamb in range(6):
    while len(chamb_all[chamb]) < max_len:
        chamb_all[chamb] = np.append(chamb_all[chamb], 0)

fig, axs = plt.subplots()
axs.plot(chamb_all[0]/160, 'r', label='chamber 1')
axs.plot(chamb_all[1]/160, 'g', label='chamber 2')
axs.plot(chamb_all[2]/160, 'y', label='chamber 3')
axs.plot(chamb_all[3]/160, 'b', label='chamber 4')
axs.plot(chamb_all[4]/160, 'c', label='chamber 5')
axs.plot(chamb_all[5]/160, 'k', label='chamber 6')
plt.xlabel('Sample number')
plt.ylabel('Pressure value (Bars)')
plt.legend()
plt.title('Babbling: plateau')
plt.show()

########################################################################################################################
# SAVE DATA INTO FILE

# rev = 0
# file_wname = 'DataFolder/plateau_' + str(rev) + '.txt'
# with open(file_wname, 'w') as f:
#     for d in range(len(chamb_all[0])):
#         f.write(str([d, chamb_all[0][d], chamb_all[1][d], chamb_all[2][d],
#                      chamb_all[3][d], (chamb_all[4][d]/160)*52, chamb_all[5][d]]).strip('[]'))
#         f.write('\n')
