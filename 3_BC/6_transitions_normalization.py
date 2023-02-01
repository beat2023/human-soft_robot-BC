# Create the normalized transitions starting from the demonstration data

import numpy as np
import pandas as pd
from io import StringIO
import os
from matplotlib import pyplot as plt
from functions import normalize


demo_type_list = ['robot', 'human']
dirs = ['minMax', 'norm', 'zScore', 'confInt', 'ellipsoid']

for t in range(len(demo_type_list)):

    demo_type = demo_type_list[t]

    for d in range(len(dirs)):
        path_dir = 'DEMONSTRATIONS/' + demo_type + '_remapped/' + dirs[d]

        files_names = os.listdir(path_dir)
        files_path = []
        for i in range(len(files_names)):
            files_path.append(path_dir + '/' + files_names[i])

        for i in range(len(files_names)):
            # Only if the file is of demo (no origin) I take it into consideration
            if len(open(files_path[i]).readlines()) > 1:
                file_1 = open(files_path[i], 'r').read().replace(',\n', ",").replace('])', "").replace('array([', "")
                data = StringIO(file_1)
                df = pd.read_csv(data, sep=",",
                                 names=["count", "pmx", "pmy", "pmz",
                                                 "dmx", "dmy", "dmz"])
                # Markers of interest selected
                positions = df.loc[:, "pmx":"dmz"].to_numpy()
                positions = normalize(positions, 'state')

                file_name = 'DEMOS_NORMALIZED/' + demo_type + '/' + dirs[d] + '/transitions_' + files_names[i]
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                # To use all the states of the trajectory
                positions = np.append(positions, np.reshape(positions[-1, :], (1, 6)), axis=0)
                with open(file_name, 'w') as f:
                    for j in range(positions.shape[0]-1):
                        f.write(str(positions[j, :].tolist()).replace(", ", ",") + ' ' +
                                str(positions[j+1, :].tolist()).replace(", ", ","))
                        f.write('\n')

