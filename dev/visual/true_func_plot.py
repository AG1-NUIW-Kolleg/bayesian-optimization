from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dev.models.AD_Hill_System_HMC_Py import observe_blackbox_simulation

min_stretch_m1 = 12
max_stretch_m1 = 17
min_stretch_m2 = 13
max_stretch_m2 = 18
num_stretches_per_muscle = 20
stretches_m1 = np.linspace(min_stretch_m1, max_stretch_m1,
                           num_stretches_per_muscle)
stretches_m2 = np.linspace(min_stretch_m2, max_stretch_m2,
                           num_stretches_per_muscle)

data = []

for stretch_m1 in stretches_m1:
    for stretch_m2 in stretches_m2:
        stretch_pair = [stretch_m1, stretch_m2]

        result = observe_blackbox_simulation(stretch_pair)
        min_m1 = result[1]
        max_m1 = result[0]
        min_m2 = result[3]
        max_m2 = result[2]
        range_of_motion_m1 = max_m2 - min_m2
        stretch_dict = {
            'pre_stretches': stretch_pair,
            'stretch_score': range_of_motion_m1,
        }
        data.append(stretch_dict)


def plot_stretch_data_3d(data):
    x = [entry['pre_stretches'][0] for entry in data]
    y = [entry['pre_stretches'][1] for entry in data]
    z = [entry['stretch_score'] for entry in data]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=z)
    ax.plot(x, y, z, color='red')

    ax.set_xlabel('stretch M1 [cm]')
    ax.set_ylabel('stretch M2 [cm]')
    fig.colorbar(sc, ax=ax, label='range of motion M2 [cm]')

    plt.savefig('plots/true_func.png')


plot_stretch_data_3d(data)
