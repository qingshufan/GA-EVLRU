"""
File: Demand Data Processing for UrbanEV Dataset
Purpose: Aggregate hourly demand.
Key Features:
- Grouping and aggregating data in intervals.
- Matching and combining rows from different DataFrames based on timestamps.
Outputs:
- output/urbanev/bss.csv: The final processed and aggregated data.
"""

import pandas as pd
from tqdm import tqdm
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

replace_df = read_csv_with_progress('output/urbanev/demand.csv')
duration_df = read_csv_with_progress('datasets/urbanev/duration.csv')
cols = duration_df.columns[1:]
for col in cols:
    duration_df[col] = 0
    duration_df[col] = duration_df[col].astype(int)


duration_df = duration_df.set_index('time')
for index, row in tqdm(replace_df.iterrows(), total=len(replace_df), desc="Processing rows"):
    start_time = row['Start']
    area = str(int(row['Area']))
    demand = row['Demand']
    if start_time in duration_df.index and area in duration_df.columns:
        duration_df.at[start_time, area] = demand
duration_df = duration_df.reset_index()

grouped = duration_df.groupby(duration_df.index).agg({
    'time': 'first',
    **{col: 'sum' for col in duration_df.columns if col != 'time'}
})

grouped['time'] = pd.to_datetime(grouped['time'])

time_index = grouped.columns.get_loc('time')

new_columns = ['month', 'day', 'year', 'hour', 'minute', 'second']

for i, col in enumerate(new_columns):
    grouped[col] = getattr(grouped['time'].dt, col)
    grouped.insert(time_index + i, col, grouped.pop(col))

grouped = grouped.drop(columns=['time'])

output_dir = "output/urbanev"
os.makedirs(output_dir, exist_ok=True)
save_csv_with_progress(grouped, f'{output_dir}/bss.csv')