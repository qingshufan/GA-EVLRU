"""
File: Best Model Selection Based on Evaluation Metrics
Purpose: Select the best model from evaluation results.
Key Features:
- Calculate a total score considering MAE, MSE, RMSE, and time.
- Sort models by the total score in ascending order.
- Identify the best model with the lowest total score.
Outputs:
- eva.csv: updated evaluation file with total scores.
"""

import pandas as pd
import sys
import yaml
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_score.py <config_name>")
        sys.exit(1)
    config_name = sys.argv[1]
    with open(f'config/{config_name}.yaml', 'r') as file:
        config2 = yaml.safe_load(file)
    names = config2["name"]
    for name in names:
        data_file_path = f"output/{name}/predict/eva.csv"
        data = pd.read_csv(data_file_path)
        data['total_score'] = ((data['MAE'] + data['MSE'] + data['RMSE']) * 0.8 + data['time'] * 0.2).round(4)
        sorted_data = data.sort_values(by='total_score')
        best_model = sorted_data.iloc[0]['model']
        print("best model:", best_model)
        data.to_csv(data_file_path, index=False)