from __future__ import annotations

import matplotlib.pyplot as plt


class RangeOfMotionPlotter():
    def __init__(self, initial_muscle_lengths, range_of_motions):
        self._muscle_lengths = initial_muscle_lengths
        self._range_of_motions = range_of_motions

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self._muscle_lengths[:, 0]
        y = self._muscle_lengths[:, 1]
        z = self._range_of_motions.flatten()

        sc = ax.scatter(x, y, z, c=z)
        ax.set_xlabel('stretch muscle 1 [cm]')
        ax.set_ylabel('stretch muscle 2 [cm]')
        ax.plot(x, y, z, color='red')

        fig.colorbar(sc, ax=ax, label='range of motion muscle 1 [cm]')

        plt.savefig('plots/true_func.png')
