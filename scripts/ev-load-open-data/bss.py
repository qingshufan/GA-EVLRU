"""
File: Demand Data Processing for EV-Load-Open-Data 
Purpose: Aggregate hourly demand.
Key Features:
- Convert 'Start' column to datetime and extract time components.
- Aggregate demand by hourly intervals.
Outputs:
- output/{name}/bss.csv: Processed demand data with time components separated.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def read_csv_with_progress(file_path):
    try:
        total_lines = sum(1 for _ in open(file_path, 'r'))
        with tqdm(total=total_lines, desc=f"Reading {file_path}") as pbar:
            df = pd.read_csv(file_path, chunksize=1000)
            result = []
            for chunk in df:
                result.append(chunk)
                pbar.update(len(chunk))
        return pd.concat(result)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

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

    result_df = read_csv_with_progress(f'output/{name}/demand.csv')
    if result_df is None:
        continue

    result_df['Start'] = pd.to_datetime(result_df['Start'])

    start_time = result_df['Start'].min()

    # Generate hourly time series
    time_index = pd.date_range(start=start_time, end=result_df['Start'].max(), freq='H')
    results = []
    j = 0

    for current_time in tqdm(time_index, desc="Processing time steps"):
        rows_added = False
        amt = 0
        while j < len(result_df) and current_time >= result_df.iloc[j]['Start']:
            row = result_df.iloc[j].to_dict()
            amt += row['Demand']
            j += 1
            rows_added = True
        if not rows_added:
            # Create a row with all 0s, keep 'Start' as current_time
            row = {'Start': current_time, 'Demand': 0}
            results.append(row)
        else:
            amt = np.round(amt).astype(int)
            row = {'Start': current_time, 'Demand': amt}
            results.append(row)

    new_result_df = pd.DataFrame(results)
    new_result_df = new_result_df[['Start', 'Demand']]
    time_index = new_result_df.columns.get_loc('Start')
    new_columns = ['month', 'day', 'year', 'hour', 'minute', 'second']
    for i, col in enumerate(new_columns):
        new_result_df[col] = getattr(new_result_df['Start'].dt, col)
        new_result_df.insert(time_index + i, col, new_result_df.pop(col))
    new_result_df = new_result_df.drop(columns=['Start'])

    save_csv_with_progress(new_result_df, f'output/{name}/bss.csv')