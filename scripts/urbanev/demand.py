"""
File: EV Battery Swapping Demand Calculation and Data Processing for UrbanEV Dataset
Purpose: Calculate the demand probability of battery replacement at each area.
Key Features:
- Calculate the average charging times and update the demand.
Outputs:
- output/urbanev/demand.csv: The final processed data with battery replacement demand.
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

duration_df = read_csv_with_progress('datasets/urbanev/duration.csv')
volume_df = read_csv_with_progress('datasets/urbanev/volume.csv')

areas = duration_df.columns[1:]

results = []

# Parameters for calculating battery swapping demand probability
alpha = 0.08  # Weight for charging time cost
beta = 0.08  # Weight for charging energy
gamma = -0.8  # Weight for battery swapping time cost
replacement_time_cost = 3
timestamp = 60

for area in tqdm(areas, desc="Processing areas"):
    duration_col = duration_df[area]
    volume_col = volume_df[area]
    i = 0
    total_charge_count = 0
    segment_count = 0
    area_results = []
    while i < len(duration_col):
        current_duration = duration_col[i]
        if current_duration > 0:
            consecutive_count = 1
            total_volume = volume_col[i]
            # Find consecutive rows with the same duration
            while i + consecutive_count < len(duration_col) and duration_col[i + consecutive_count] == current_duration:
                consecutive_count += 1
            next_index = i + consecutive_count
            if next_index < len(duration_col) and duration_col[next_index] >= duration_col[next_index - 1]:
                i = next_index
                continue
            elif next_index < len(duration_col) and duration_col[next_index] < duration_col[next_index - 1]:
                temp = current_duration * 60 / timestamp - (duration_col[next_index] * 60 / timestamp)
                if temp > 0 and temp < 1:
                    charge_count = 1
                else:
                    charge_count = int(temp)
            elif next_index >= len(duration_col):
                i = next_index
                continue
            avg_charge_duration = timestamp * consecutive_count
            avg_charge_volume = (total_volume / (int(current_duration * 60 / timestamp) if int(current_duration * 60 / timestamp) != 0 else 1) * charge_count if charge_count > 0 else 0) * consecutive_count
            start_timestamp = pd.to_datetime(duration_df.iloc[i, 0])

            # Calculate the probability of battery replacement demand
            z = alpha * avg_charge_duration + beta * avg_charge_volume + gamma * replacement_time_cost
            probability = 1 / (1 + np.exp(-z))
            total_charge_count += charge_count
            segment_count += 1
            demand = probability

            area_results.append({
                'Start': start_timestamp,
                'Area': area,
                'Demand': demand
            })
            i = next_index
        else:
            i += 1

    # Calculate the average charging times and update the demand
    if segment_count > 0:
        avg_charge_count = total_charge_count / segment_count
        for result in area_results:
            result['Demand'] = np.round(result['Demand'] * avg_charge_count).astype(int)

    results.extend(area_results)

with tqdm(total=1, desc="Creating DataFrame") as pbar:
    result_df = pd.DataFrame(results)
    pbar.update(1)

output_dir = "output/urbanev"
os.makedirs(output_dir, exist_ok=True)
save_csv_with_progress(result_df, f'{output_dir}/demand.csv')