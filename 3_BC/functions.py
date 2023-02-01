"""Useful functions for BCO"""

import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
from copy import copy
from sklearn.metrics import mean_absolute_error

########################################################################################################################
# If I am calling 'functions' from a subdirectory of BC
added_subdir = ''
current_path = os.getcwd()
# if current_path != 'C:/Users/tosib/Documents/UNIVERSITÀ/2-MAGISTRALE/THESIS/2_Python/BC':
#     added_subdir = '../'

########################################################################################################################
# POSITIONS

f_max_pos = open(added_subdir + '../IDM/Normalization_parameters/max_pos.txt').read().split('\n')
max_pos = []
for line in f_max_pos:
    if len(line) > 1:
        max_pos.append(float(line))
max_pos = np.reshape(np.asarray(max_pos), (1, 6))

f_min_pos = open(added_subdir + '../IDM/Normalization_parameters/min_pos.txt').read().split('\n')
min_pos = []
for line in f_min_pos:
    if len(line) > 1:
        min_pos.append(float(line))
min_pos = np.reshape(np.asarray(min_pos), (1, 6))

scaler_pos = MinMaxScaler(feature_range=(-1, 1))
norm_pars_pos = np.append(min_pos, max_pos, axis=0)
scaler_pos.fit(norm_pars_pos)

########################################################################################################################
# PRESSURES

f_max_press = open(added_subdir + '../IDM/Normalization_parameters/max_press.txt').read().split('\n')
max_press = []
for line in f_max_press:
    if len(line) > 1:
        max_press.append(int(float(line)))
max_press = np.reshape(np.asarray(max_press), (1, 6))

f_min_press = open(added_subdir + '../IDM/Normalization_parameters/min_press.txt').read().split('\n')
min_press = []
for line in f_min_press:
    if len(line) > 1:
        min_press.append(int(float(line)))
min_press = np.reshape(np.asarray(min_press), (1, 6))

scaler_press = MinMaxScaler(feature_range=(-1, 1))
norm_pars_press = np.append(min_press, max_press, axis=0)
scaler_press.fit(norm_pars_press)


########################################################################################################################
# FUNCTIONS

def normalize(data, id_data):
    """Normalize data w.r.t. minima and maxima"""
    data = np.array(data)
    flag = False
    if data.ndim == 1:
        flag = True
        data = np.reshape(data, (1, len(data)))
    if id_data == 'state':
        pos_scaled = scaler_pos.transform(data)
        result = pos_scaled
    elif id_data == 'action':
        press_scaled = scaler_press.transform(data)
        result = press_scaled
    if flag:
        result = np.reshape(result, (result.shape[1],))
    return result


def denormalize(data, id_data):
    """De-normalize data w.r.t. minima and maxima"""
    data = np.array(data)
    flag = False
    if data.ndim == 1:
        flag = True
        data = np.reshape(data, (1, len(data)))
    if id_data == 'state':
        pos_original = scaler_pos.inverse_transform(data)
        result = pos_original
    elif id_data == 'action':
        press_original = scaler_press.inverse_transform(data)
        result = press_original
    if flag:
        result = np.reshape(result, (result.shape[1],))
    return result


def min_action():
    return np.reshape(min_press, (6,))


def max_action():
    return np.reshape(max_press, (6,))


def min_state():
    return np.reshape(min_pos, (6,))


def max_state():
    return np.reshape(max_pos, (6,))


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


def load_demonstration(input_filename):
    """Load demonstration trajectories from the file specified"""
    inputs = []
    targets = []
    for trajectory in open(input_filename):
        s, s_next = trajectory.replace("\n", "").replace(",,", ",").split(" ")
        s = eval(s)
        s_next = eval(s_next)
        inputs.append(s)
        targets.append(s_next)
    return inputs, targets


def load_origin():
    # load the origin of the robot
    file_origin = open(added_subdir + 'DEMONSTRATIONS/robot/origin.txt', 'r').read()
    file_origin = file_origin.replace('[', "").replace(']', "").split(' ')
    origin = []
    for el in file_origin:
        if el != '':
            origin.append(float(el))
    origin = np.array(origin)
    return origin


def limit(prev_action, action, how_much):
    prev_action = denormalize(prev_action, 'action')
    action = denormalize(action, 'action')
    limit_val = 20  # limit value for range [0, 160]
    if how_much == 'more':
        limit_val = limit_val/2
    for i in range(len(action)):
        if i == 4:
            limit_factor = limit_val*52/160  # limit factor for range [0, 52]
        else:
            limit_factor = limit_val
        if action[i] - prev_action[i] > limit_factor:
            action[i] = prev_action[i] + limit_factor
        elif action[i] - prev_action[i] < - limit_factor:
            action[i] = prev_action[i] - limit_factor
    return normalize(action, 'action')


def load_babbling_data():
    file = open(added_subdir + '../IDM/BabblingData/all_normalized.txt', 'r').read()
    file = file.replace('[', "").replace(']', "")
    data = StringIO(file)
    df = pd.read_csv(data, sep=",", names=["count", "P1", "P2", "P3", "P4", "P5", "P6",
                                           "f1", "f2", "f3", "f4", "f5", "f6",
                                           "f7", "f8", "f9", "f10", "f11", "f12",
                                           "f13", "f14", "f15", "f16", "f17", "f18",
                                           "f19", "f20", "f21", "f22", "f23", "f24",
                                           "f25", "f26", "f27", "f28", "f29", "f30",
                                           "f31", "f32", "f33", "f34", "f35", "f36",
                                           "f37", "f38", "f39", "f40", "f41", "f42",
                                           "f43", "f44", "f45", "f46", "f47", "f48"])
    features = df.loc[:, "f1":].to_numpy()
    pressures = df.loc[:, "P1":"P6"].to_numpy()
    return features, pressures


def extract_random():
    babbling_features, babbling_action = load_babbling_data()
    num = babbling_features.shape[0]
    tmp = np.arange(num)
    np.random.shuffle(tmp)
    small_range = tmp[:1000]  # Only a part of samples
    features = []
    actions = []
    for i in range(len(small_range)):
        features.append(babbling_features[small_range[i], :])
        actions.append(babbling_action[small_range[i], :])
    return np.array(features), np.array(actions)


def resampling(data_series, new_length):
    data_series = np.array(data_series)
    resampled_data = np.zeros((new_length, data_series.shape[1]))
    x = np.linspace(0, data_series.shape[0] - 1, data_series.shape[0])
    x_interp = np.linspace(0, data_series.shape[0] - 1, data_series.shape[0] * 100)
    sel = np.around(np.linspace(0, data_series.shape[0] * 100 - 1, new_length)).astype(int)
    for s in range(data_series.shape[1]):
        y_interp = np.interp(x_interp, x, data_series[:, s])
        resampled_data[:, s] = y_interp[sel]
    return resampled_data


def save_post_demo(states_demo, states_idm, states_reached, demo_type, remap_type, demo_name, rev):
    # File where to write states used for the current demo and states reached
    states_demo = denormalize(states_demo, 'state')
    states_idm = denormalize(states_idm, 'state')
    states_reached = denormalize(states_reached, 'state')
    # Resampling policy data so that it has same number of samples of demos
    resampled_idm = resampling(states_idm, states_demo.shape[0])
    resampled_policy = resampling(states_reached, states_demo.shape[0])

    dir_name = added_subdir + 'PostDemo/' + demo_type + '/' + remap_type
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_wname = dir_name + '/' + demo_name + '_' + str(rev) + '.txt'
    with open(file_wname, 'w') as f:
        for d in range(states_demo.shape[0]):
            f.write(str([d, states_demo[d, :], resampled_idm[d, :], resampled_policy[d, :]]).strip('[]'))
            f.write('\n')


def check_action(action):
    if action.ndim == 2:
        action = np.reshape(action, (action.shape[1],))
    action_den = denormalize(action, 'action')
    for i in range(len(action_den)):
        if action_den[i] < min_action()[i]:
            action_den[i] = min_action()[i]
        if action_den[i] > max_action()[i]:
            action_den[i] = max_action()[i]
    return normalize(action_den, 'action')


def shift(states, first, current):
    """Add a line to reach the starting position"""
    states = denormalize(np.array(states), 'state')
    first = denormalize(np.array(first), 'state')
    current = denormalize(current, 'state')
    n_points = 8
    add_points = np.zeros((n_points, 6))
    for i in range(6):
        add_points[:, i] = np.linspace(current[i], first[i], num=n_points)
    states = np.insert(states, 0, add_points, axis=0)
    states = normalize(states, 'state')
    return states.tolist()


def clear_post_demo(demo_mode, remap_type, demo_name):
    """It clears the post demo files relative to a specific type"""
    dir_name = added_subdir + 'PostDemo/' + demo_mode + '/' + remap_type + '/' + demo_name
    episode = 0
    while True:
        file_name = dir_name + '_' + str(episode) + '.txt'
        if not os.path.isfile(file_name):
            break  # If the file doesn't exist I exit the loop
        os.remove(file_name)
        episode += 1


def get_features(states_in, actions_in):
    # DA TOGLIERE COMMENTO
    states = copy(states_in)
    actions = copy(actions_in)
    # states = states_in
    # actions = actions_in
    Features = []
    # Initialize the first 2 past states
    prev2_state = states[0]
    prev_state = states[0]
    prev2_action = actions[0]
    prev_action = actions[0]
    for d in range(3):
        states = np.append(states, np.reshape(states[-1], (1, 6)), axis=0)
    for d in range(states.shape[0] - 3):
        features = np.hstack((states[d], prev2_state, prev_state, states[d + 1], states[d + 2], states[d + 3],
                              prev2_action, prev_action))
        Features.append(list(features))
        prev2_state = prev_state
        prev_state = states[d]
        prev2_action = prev_action
        prev_action = actions[d]
    return np.array(Features)


def save_idm_perform(states_demo, states_idm, demo_mode, remap_type, demo_name, rev):
    # File where to write states used for the current demo and states reached
    states_demo = np.array(states_demo)
    states_idm = np.array(states_idm)
    states_demo = denormalize(states_demo, 'state')
    states_idm = denormalize(states_idm, 'state')
    # Resampling policy data so that it has same number of samples of demos
    resampled_idm = resampling(states_idm, states_demo.shape[0])
    dir_name = added_subdir + 'IdmPerform/' + demo_mode + '/' + remap_type
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_wname = dir_name + '/' + demo_name + '_' + str(rev) + '.txt'
    with open(file_wname, 'w') as f:
        for d in range(states_demo.shape[0]):
            f.write(str([d, states_demo[d, :], resampled_idm[d, :]]).strip('[]'))
            f.write('\n')


def visualize_test(targets, outputs, title, MAE, MAE_idm):
    targets = np.array(targets)
    outputs = np.array(outputs)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # proximal segment
    ax1.plot(targets[:, 0], targets[:, 1], targets[:, 2], color='r', label='demonstration')
    ax1.plot(outputs[:, 0], outputs[:, 1], outputs[:, 2], color='b', label='output')
    ax1.legend()
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.set_zlabel("z", fontsize=15)
    ax1.title.set_text('Proximal segment')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # distal segment
    ax2.plot(targets[:, 3], targets[:, 4], targets[:, 5], color='r', label='demonstration')
    ax2.plot(outputs[:, 3], outputs[:, 4], outputs[:, 5], color='b', label='output')
    ax2.legend()
    ax2.set_xlabel("x", fontsize=15)
    ax2.set_ylabel("y", fontsize=15)
    ax2.set_zlabel("z", fontsize=15)
    ax2.title.set_text('Distal segment')
    fig = plt.gcf()
    fig.suptitle(f'{title}\n'
                 f'MAE with original demo (distal segment): {MAE}\n'
                 f'MAE with idm-passed demo (distal segment): {MAE_idm}')
    plt.show()


def visualize_idm(demo_type, remap_type, demo_name):
    episode = 0
    MAE = []
    while True:
        path = 'IdmPerform/' + demo_type + '/' + remap_type + '/' + demo_name + '_' + str(episode) + '.txt'
        if not os.path.isfile(path):
            # If the file doesn't exist I exit the loop
            break
        data = open(path, 'r').read()
        data = data.replace(',\n', ",")
        data = data.replace('])', "")
        data = data.replace('array([', "")
        data = data.replace('[', "")
        data = data.replace(']', "")
        data = data.replace(', dtype=float32)', "")
        data = StringIO(data)
        df2 = pd.read_csv(data, sep=",", names=["i", "demo_pmx", "demo_pmy", "demo_pmz", "demo_dmx", "demo_dmy", "demo_dmz",
                                                "idm_pmx", "idm_pmy", "idm_pmz", "idm_dmx", "idm_dmy", "idm_dmz"])
        states_demo = df2.loc[:, "demo_pmx":"demo_dmz"].to_numpy()
        resampled_idm = df2.loc[:, "idm_pmx":"idm_dmz"].to_numpy()

        MAE.append(mean_absolute_error(states_demo[:, 3:], resampled_idm[:, 3:]))

        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(states_demo[:, 0], states_demo[:, 1], states_demo[:, 2], color='r', label='demonstration')
        ax1.plot(resampled_idm[:, 0], resampled_idm[:, 1], resampled_idm[:, 2], color='g', label='idm')
        ax1.set_title('Proximal segment')
        ax1.legend()
        ax1.set_xlabel("x", fontsize=15)
        ax1.set_ylabel("y", fontsize=15)
        ax1.set_zlabel("z", fontsize=15)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(states_demo[:, 3], states_demo[:, 4], states_demo[:, 5], color='r', label='demonstration')
        ax2.plot(resampled_idm[:, 3], resampled_idm[:, 4], resampled_idm[:, 5], color='g', label='idm')
        ax2.set_title('Distal segment')
        ax2.legend()
        ax2.set_xlabel("x", fontsize=15)
        ax2.set_ylabel("y", fontsize=15)
        ax2.set_zlabel("z", fontsize=15)
        fig = plt.gcf()
        fig.suptitle(f"Post-demonstration number {episode+1}\nMAE of distal segment: {round(MAE[episode], 3)}")
        plt.show()

        episode += 1


def load_idm_perform_best(demo_type, remap_type, demo_name):
    path = 'IdmPerform/' + demo_type + '/' + remap_type + '/' + demo_name + '_best.txt'
    data = open(path, 'r').read()
    data = data.replace(',\n', ",").replace('])', "").replace('array([', "")
    data = StringIO(data)
    df2 = pd.read_csv(data, sep=",",
                      names=["i", "demo_pmx", "demo_pmy", "demo_pmz", "demo_dmx", "demo_dmy", "demo_dmz",
                             "idm_pmx", "idm_pmy", "idm_pmz", "idm_dmx", "idm_dmy", "idm_dmz"])
    # states_demo = df2.loc[:, "demo_pmx":"demo_dmz"].to_numpy()
    resampled_idm = df2.loc[:, "idm_pmx":"idm_dmz"].to_numpy()

    return resampled_idm


def save_test_policy(targets, output_idm, outputs, demo_mode, remap_type, demo_name, numb):
    dir_name = added_subdir + 'TestPolicy/' + demo_mode + '/' + remap_type
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_wname = dir_name + '/' + demo_name + '_' + str(numb) + '.txt'
    with open(file_wname, 'w') as f:
        for d in range(targets.shape[0]):
            f.write(str([d, targets[d, :], output_idm[d, :], outputs[d, :]]).strip('[]'))
            f.write('\n')


def save_test_MAE(MAE, MAE_idm, demo_mode, remap_type, demo_name):
    MAE_avg = np.mean(MAE)
    MAE_idm_avg = np.mean(MAE_idm)
    dir_name = added_subdir + 'TestPolicy/' + demo_mode + '/' + remap_type
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_wname = dir_name + '/' + demo_name + '_MAE.txt'
    with open(file_wname, 'w') as f:
        f.write('MAE with original demonstration\n')
        f.write(f'Average: {MAE_avg}\n')
        f.write(f'All values: {MAE}\n\n')
        f.write('MAE with idm-passed demonstration\n')
        f.write(f'Average: {MAE_idm_avg}\n')
        f.write(f'All values: {MAE_idm}')

