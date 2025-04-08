"""
File: EV Battery Swapping Demand Calculation and Data Processing
Purpose: Calculate battery swapping demand probabilities for EV-Load-Open-Data.
Key Features:
- Calculate demand probabilities using a sigmoid function.
Outputs:
- output/{name}/demand.csv: Contains 'Start' and calculated 'Demand' columns.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def read_csv_with_progress(file_path):
    total_lines = sum(1 for _ in open(file_path, 'r'))
    with tqdm(total=total_lines, desc=f"Reading {file_path}") as pbar:
        df = pd.read_csv(file_path, chunksize=1000)
        result = []
        for chunk in df:
            result.append(chunk)
            pbar.update(len(chunk))
    return pd.concat(result)

def save_csv_with_progress(df, file_path, chunksize=1000):
    num_chunks = len(df) // chunksize + (1 if len(df) % chunksize != 0 else 0)
    with tqdm(total=num_chunks, desc=f"Saving {file_path}") as pbar:
        for i in range(0, len(df), chunksize):
            if i == 0:
                df[i:i + chunksize].to_csv(file_path, index=False)
            else:
                df[i:i + chunksize].to_csv(file_path, mode='a', index=False, header=False)
            pbar.update(1)


names = ['acn', 'boulder_2021', 'palo_alto', 'sap', 'dundee', 'paris', 'perth']

for name in names:
    output_dir = f"output/{name}"
    os.makedirs(output_dir, exist_ok=True)

    name_df = read_csv_with_progress(f'datasets/{name}.csv')

    name_df = name_df.sort_values(by='Start')

    # Parameters for calculating battery swapping demand probability
    alpha = 0.08  # Weight for charging time cost
    beta = 0.08  # Weight for charging energy
    gamma = -0.8  # Weight for battery swapping time cost
    replacement_time_cost = 3

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Determine duration column
    if 'Charge.Duration' in name_df.columns:
        duration_column = name_df['Charge.Duration']
    else:
        duration_column = name_df['Park.Duration']

    # Calculate demand probability
    name_df['Demand'] = sigmoid(alpha * duration_column + beta * name_df['Energy'] + gamma * replacement_time_cost)

    # Keep only 'Start' and 'Demand' columns
    result_df = name_df[['Start', 'Demand']]

    save_csv_with_progress(result_df, f'output/{name}/demand.csv')