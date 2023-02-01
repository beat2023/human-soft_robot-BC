# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html

import struct
import gym
from gym import spaces
import serial.tools.list_ports
import matlab.engine
from time import sleep
from functions import *

# ######### Before running it, write on Matlab: matlab.engine.shareEngine ############################


def port_init():
    """Initializes communication with serial port"""
    comlist = serial.tools.list_ports.comports()
    connected = []
    for element in comlist:
        connected.append(element.device)
    PORT_ID = connected[0]
    dev = serial.Serial(str(PORT_ID), baudrate=115200)
    print("Connected COM ports: " + str(PORT_ID))
    print("Is port open? ", dev.isOpen())
    return dev


def tracking_init():
    """Establishes a communication with the tracking system"""
    print('Starting communication with MATLAB')
    eng = matlab.engine.start_matlab()
    s = eng.genpath('../Matlab_communication')
    eng.addpath(s, nargout=0)
    print('Initializing communication with the tracking system')
    eng.initialize_system()
    print('Connection established')
    return eng


class SoftRobotEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, origin_pos):
        super(SoftRobotEnv, self).__init__()

        # Define action and observation space
        # self.action_space = spaces.Box(low=min_action(), high=max_action(), dtype=np.float32)
        # self.observation_space = spaces.Box(low=min_state(), high=max_state(), dtype=np.float32)

        # Normalized action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1., -1., -1., -1., -1., -1.]),
                                       high=np.array([1., 1., 1., 1., 1., 1.]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1., -1., -1., -1., -1., -1.]),
                                            high=np.array([1., 1., 1., 1., 1., 1.]), dtype=np.float32)

        self.origin_repeated = np.append(origin_pos, origin_pos)

        self.dev = port_init()
        self.eng = tracking_init()

        self.agent_pos = np.zeros((6,))  # position of agent NOT normalized
        self.action = np.zeros((6,))  # action of the agent NOT normalized

        # Initialize the manipulator
        self.reset()

    def reset(self):
        # Re-initialize the agent
        all_good = False
        action = np.array(self.action)  # the last action taken
        while not all_good:
            action[:4] = action[:4] - 18
            action[4] = action[4] - 6
            action[5] = action[5] - 18
            for el in range(len(action)):
                if action[el] < 0:
                    action[el] = 0
            DATA_to_write, DATA_to_save = set_sup(pressure_array=action)
            self.action = DATA_to_save
            print('Data to write to reset position: ', DATA_to_save)
            self.dev.write(bytearray(DATA_to_write))
            sleep(0.1)   # 10Hz for resetting position
            if all(action == 0):
                all_good = True
        pos = np.array(self.eng.positions()) - self.origin_repeated
        self.agent_pos = np.reshape(pos, (6,))  # position NOT normalized
        return normalize(self.agent_pos, 'state').astype(np.float32)

    def step(self, action_in):
        """ The system communicates the action (pressure) to the manipulator and receives the state achieved """
        action = (denormalize(action_in, 'action')).astype(int)
        self.action = action
        DATA_to_write, DATA_to_save = set_sup(pressure_array=action)
        print('Data to write: ', DATA_to_save)
        self.dev.write(bytearray(DATA_to_write))
        pos = np.array(self.eng.positions()) - self.origin_repeated
        self.agent_pos = np.reshape(pos, (6,))
        reward = 0.0   # not useful for us
        done = False   # not useful for us
        info = {}   # not useful for us
        return normalize(self.agent_pos, 'state').astype(np.float32), reward, done, info

    def get_state(self):
        pos = np.array(self.eng.positions()) - self.origin_repeated
        self.agent_pos = np.reshape(pos, (6,))
        return normalize(self.agent_pos, 'state').astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        self.reset()
        print('Releasing the tracking system')
        self.eng.release_system()
        print('Closing connection with MATLAB')
        self.eng.quit()
