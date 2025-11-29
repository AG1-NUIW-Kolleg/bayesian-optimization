from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class MaxRomsPlotter():
    def __init__(self, csv_filepath):
        df = pd.read_csv(csv_filepath)
        self._rho = df['rho'].values
        self._am = df['Am'].values
        self._range_of_motion = df['range_of_motion'].values

    def save_as_csv(self, filename):
        df = pd.DataFrame(
            {'rho': self._rho,
             'Am': self._am,
             'range_of_motion': self._range_of_motion}
        )
        df.to_csv(f'out/data_{filename}.csv', index=False)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self._rho
        y = self._am
        z = self._range_of_motion

        sc = ax.scatter(x, y, z, c=z)
        ax.set_xlabel('rho')
        ax.set_ylabel('Am')
        ax.set_zlabel('range of motion')

        fig.colorbar(sc, ax=ax, label='range of motion')

        plt.savefig('out/max_roms_plot.png')
