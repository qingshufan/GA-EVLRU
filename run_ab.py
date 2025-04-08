"""
File: Battery Swapping Demand Decomposition
Purpose: Decompose future 24 - hour battery swapping demand into A and B battery demands.
Key Features:
- Select top 5 areas with highest total demand.
- Randomly allocate demand to A and B batteries.
- Optionally integrate electricity price data.
Outputs:
- {area}_ab.csv per area: decomposed demand and optionally price data.
"""

import numpy as np
import pandas as pd
import os
import sys
import yaml

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_ab.py <config_name>")
        sys.exit(1)

    config_name = sys.argv[1]
    with open(f'config/{config_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    names = config.get('name', [])
    model = config.get('model')
    eprice = config.get('eprice')
    timestamp = config.get('timestamp')
    has_price = config.get('has_price')

    for n in names:
        # Read model prediction data
        model_df = pd.read_csv(f"output/{n}/predict/model/{model}.csv")
        column_sums = []
        # Calculate sum of each column
        for col in model_df.columns[0:]:
            column_sum = model_df[col].sum()
            column_sums.append((col, column_sum))
        # Sort columns by sum in descending order
        column_sums.sort(key=lambda x: x[1], reverse=True)
        for item in column_sums[:5]:
            a, b = item
            print(f"Total daily battery swapping demand in traffic area {a}: {b}")
        area = [t[0] for t in column_sums[:5]]
        all_digits = all(element.isdigit() for element in area)
        if all_digits:
            area.sort(key=lambda x: int(x))
        print(area)
        config["areas"]=area
        with open(f'config/{config_name}.yaml', 'w') as file:
            yaml.dump(config, file)
        for a in area:
            try:
                os_df = pd.read_csv(f'output/{n}/predict/model/{model}.csv')
                if has_price:
                    price_df = pd.read_csv(f'datasets/{eprice}.csv')
                count_a = 0
                count_b = 0
                rs = []
                for i in range(0, len(os_df)):
                    row = os_df.loc[i, a]
                    if has_price:
                        price = price_df[a][i * (60 / timestamp)]
                    for _ in range(int(row)):
                        random_num = np.random.rand()
                        if random_num < 5 / (8 + 5):
                            count_a += 1
                        else:
                            count_b += 1
                    if has_price:
                        rs.append((count_a, count_b, price))
                    else:
                        rs.append((count_a, count_b))
                    count_a = 0
                    count_b = 0
                
                if has_price:
                    rs_df = pd.DataFrame(rs, columns=['a', 'b', 'price'])
                else:
                    rs_df = pd.DataFrame(rs, columns=['a', 'b'])
                
                output_dir = f'output/{n}/predict/area'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                rs_df.to_csv(f'{output_dir}/{a}_ab.csv', index=False)
            except FileNotFoundError:
                print(f"File not found: There was a problem during processing for name={n} and area={a}. Please check the file path and file name.")