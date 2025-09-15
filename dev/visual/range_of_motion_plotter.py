from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RangeOfMotionPlotter():
    def __init__(self, initial_muscle_lengths, range_of_motions, params):
        self._muscle_lengths = initial_muscle_lengths
        self._range_of_motions = range_of_motions
        self._params = params

    def save_as_csv(self, filename):
        range_of_motions = self._range_of_motions.flatten()
        stretched_muscle_length_one = self._muscle_lengths[:, 0]
        stretched_muscle_length_two = self._muscle_lengths[:, 1]

        relaxed_muscle_length_one = self._params['Length_Slack_M1']
        relaxed_muscle_length_two = self._params['Length_Slack_M2']

        df_len = len(range_of_motions)
        relaxed_muscle_lengths_one = np.full(df_len, relaxed_muscle_length_one)
        relaxed_muscle_lengths_two = np.full(df_len, relaxed_muscle_length_two)

        df = pd.DataFrame(
            {'range_of_motion': range_of_motions,
             'stretched_muscle_length_one': stretched_muscle_length_one,
             'stretched_muscle_length_two': stretched_muscle_length_two,
             'relaxed_muscle_length_one': relaxed_muscle_lengths_one,
             'relaxed_muscle_length_two': relaxed_muscle_lengths_two}
        )
        df.to_csv(f'out/data_{filename}.csv')

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
