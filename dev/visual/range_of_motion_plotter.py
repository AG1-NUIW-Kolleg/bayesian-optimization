from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class RangeOfMotionPlotter():
    def __init__(self, initial_muscle_lengths, range_of_motions):
        self._muscle_lengths = initial_muscle_lengths
        self._range_of_motions = range_of_motions

    def save_as_csv(self):
        range_of_motions = self._range_of_motions.flatten()
        stretched_muscle_length_one = self._muscle_lengths[:, 0]
        stretched_muscle_length_two = self._muscle_lengths[:, 1]
        df = pd.DataFrame(
            {'range_of_motion': range_of_motions,
             'stretched_muscle_length_one': stretched_muscle_length_one,
             'stretched_muscle_length_two': stretched_muscle_length_two}
        )
        df.to_csv('out/data.csv')

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

        plt.savefig('out/range_of_motion_optimized.png')
