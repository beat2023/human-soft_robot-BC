# Import data and analyze it

import pandas as pd
from io import StringIO
from matplotlib import pyplot as plt

# Change the name of the directories and of the file
folder_type = 'DataFolder'
directory = 'raw_data'
file_name = 'idm_plateau_0'
file = open(folder_type + '/' + directory + '/' + file_name + '.txt', 'r').read()

file_1 = file.replace(',\n', ",")
file_1 = file_1.replace('])', "")
file_1 = file_1.replace('array([', "")
file_1 = file_1.replace('[', "")
file_1 = file_1.replace(']', "")

data = StringIO(file_1)
df = pd.read_csv(data, sep=",",
                 names=["count", "P1", "P2", "P3", "P4", "P5", "P6",
                        "pmx", "pmy", "pmz", "dmx", "dmy", "dmz"])

markers = df.loc[:, "pmx":].to_numpy()
pressures = df.loc[:, "P1":"P6"].to_numpy()

fig = plt.figure
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
x = markers[:, [0, 3]]
y = markers[:, [1, 4]]
z = markers[:, [2, 5]]
ax.scatter3D(x[:, 0], y[:, 0], z[:, 0], marker='*', s=20, label='proximal')
ax.scatter3D(x[:, 1], y[:, 1], z[:, 1], marker='o', s=20, label='distal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc=1)
plt.show()
