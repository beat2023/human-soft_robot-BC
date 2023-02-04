"""PRESSURE INPUT"""

import numpy as np
import serial.tools.list_ports
from time import sleep
from functions import set_sup


# ############################### Find the used comport first ###############################
comlist = serial.tools.list_ports.comports()
connected = []
for element in comlist:
    connected.append(element.device)
    # print('device: ', element.device)
PORT_ID = connected[0]
dev = serial.Serial(str(PORT_ID), baudrate=115200)
print("Connected COM ports: " + str(PORT_ID))
print("Is port open? ", dev.isOpen())


########################################################################################################################

n_steps = 15
values = np.zeros((6, n_steps))
values[[0, 1, 2, 3, 5], :] = np.linspace(0, 160, num=n_steps)
values[4, :] = np.linspace(0, 52, num=n_steps)


for i in range(5):
    DATA_to_write, DATA_to_write = set_sup(pressure_array=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dev.write(bytearray(DATA_to_write))
    sleep(0.2)

for i in range(6):
    for j in range(2*values.shape[1]-1):
        press = [0, 0, 0, 0, 0, 0]
        if j < values.shape[1]:
            press[i] = values[i, j]
        else:
            press[i] = values[i, 2*values.shape[1] - (j+1)]
        # print(press)
        DATA_to_write, DATA_to_save = set_sup(pressure_array=press)
        dev.write(bytearray(DATA_to_write))
        if j == (values.shape[1]-1):
            sleep(3)
        else:
            sleep(0.1)

# for i in range(40):
#     DATA_to_write, DATA_to_write = set_sup(pressure_array=[0.0, 0.0, 0.0, 0.0, 52.0, 0.0])
#     dev.write(bytearray(DATA_to_write))
#     sleep(0.2)


for i in range(5):
    DATA_to_write, DATA_to_write = set_sup(pressure_array=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dev.write(bytearray(DATA_to_write))
    sleep(0.1)

