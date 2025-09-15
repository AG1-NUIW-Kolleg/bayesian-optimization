from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from dev.constants import ADDITIONAL_STRETCH_LENGTH
from dev.models.hill_type_model_wrapper import HillTypeModelWrapper

n = 13

relaxed_muscle_lengths = [
    (9.0, 12.0),
    (11.0, 12.0),
    (11.0, 15.0),
    (12.0, 13.0),
    (14.0, 15.0)
]

for length_pair in relaxed_muscle_lengths:
    range_of_motions = []
    used_stretched_muscle_lengths_1 = []
    used_stretched_muscle_lengths_2 = []

    params = {
        'Length_Slack_M1': length_pair[0],
        'Length_Slack_M2': length_pair[1],
    }

    stretched_muscle_lengths_1 = np.linspace(
        length_pair[0], length_pair[0]+ADDITIONAL_STRETCH_LENGTH, n)
    stretched_muscle_lengths_2 = np.linspace(
        length_pair[1], length_pair[1]+ADDITIONAL_STRETCH_LENGTH, n)

    model = HillTypeModelWrapper(params)

    for muscle_length_1 in tqdm(
            stretched_muscle_lengths_1,
            desc=f'm1 = {length_pair[0]}cm | m2 = {length_pair[1]}cm'):
        for muscle_length_2 in stretched_muscle_lengths_2:
            range_of_motion = model.simulate_forward_step(
                muscle_length_1, muscle_length_2)
            range_of_motions.append(range_of_motion)
            used_stretched_muscle_lengths_1.append(muscle_length_1)
            used_stretched_muscle_lengths_2.append(muscle_length_2)

    k = len(range_of_motions)
    relaxed_muscle_lengths_1 = np.full(k, length_pair[0])
    relaxed_muscle_lengths_2 = np.full(k, length_pair[1])

    df = pd.DataFrame(
        {
            'range_of_motion': range_of_motions,
            'stretched_muscle_length_one': used_stretched_muscle_lengths_1,
            'stretched_muscle_length_two': used_stretched_muscle_lengths_2,
            'relaxed_muscle_length_one': relaxed_muscle_lengths_1,
            'relaxed_muscle_length_two': relaxed_muscle_lengths_2,
        }
    )

    df.to_csv(f'out/param_sweep_{length_pair[0]}_m2_{length_pair[1]}.csv')
