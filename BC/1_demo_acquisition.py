# Demonstrations recording and data saving

import time
import matlab.engine
import numpy as np
from matplotlib import pyplot as plt

# ######### Before running it, write on Matlab: matlab.engine.shareEngine ##############################################

print('Starting communication with MATLAB')
eng = matlab.engine.start_matlab()
s = eng.genpath('../Matlab_communication')
eng.addpath(s, nargout=0)
eng.initialize_system()
print('Connection established')

########################################################################################################################

# demo_type = 'human' or 'robot'
demo_type = 'robot'
# operation type = 'demo', 'origin' or 'center_frame'
operation = 'demo'

########################################################################################################################

if operation == 'demo':

    demo_name = 'cross'

    for rev in range(20):
        file_name = 'DEMONSTRATIONS/' + demo_type + '/raw_data/' + demo_name + '_' + str(rev) + '.txt'
        sec = 15  # how many seconds of recording
        rateAcq = 5  # sampling frequency [Hz]
        sampl_T = 1/rateAcq
        n_samples = sec*rateAcq

        with open(file_name, 'w') as f:
            positions_all = []
            print(f'Start of acquisition of demonstration number {rev} in 2 second')
            time.sleep(2)
            print('Started')
            time_record = time.time()
            time_start = time.time()
            for d in range(n_samples):
                if d % rateAcq == 0:
                    print(f'Remaining seconds of acquisition {sec - d/rateAcq}')
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


elif operation == 'origin':

    file_name = 'DEMONSTRATIONS/' + demo_type + '/origin.txt'
    sec = 5  # how many seconds of recording
    rateAcq = 5  # sampling frequency [Hz]
    sampl_T = 1 / rateAcq
    n_samples = sec * rateAcq

    with open(file_name, 'w') as f:
        positions_all = []
        time_record = time.time()
        time_start = time.time()
        print('Start of acquisition')
        for d in range(n_samples):
            pos = eng.positions()
            positions_all.append(pos)  # I am only interested in the proximal marker
            time_to_sleep = sampl_T - (time.time() - time_record)
            if np.sign(time_to_sleep) != -1:
                time.sleep(time_to_sleep)
            time_record = time.time()
        time_stop = time.time()
        print('End of acquisition')
        print(f'Total acquisition time: {time_stop - time_start}, instead of {sec} seconds')
        positions_all = np.array(positions_all)
        positions_all = np.reshape(positions_all, [positions_all.shape[0], 6])
        origin = positions_all[:, 0:3]
        origin_mean = np.mean(origin, axis=0)
        f.write(str(origin_mean))  # .strip('[]'))

    fig = plt.figure
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    x = origin[:, 0]
    y = origin[:, 1]
    z = origin[:, 2]
    ax.scatter3D(x, y, z, marker='*', s=20, label='origin')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc=1)
    plt.show()


eng.release_system()
eng.quit()
