"""
File: Multiple Baseline Predict Models Evaluation 
Purpose: Evaluate baseline models for predicting specific areas.
Key Features:
- baseline: SVM, XGBoost, LightGBM, MLP, RandomForest.
- Model evaluation using MAE, MSE, RMSE, and time consumption.
Outputs:
- {model}.csv per model: prediction results for each area.
- eva.csv: aggregated evaluation metrics for all models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import os
import time
from tqdm import tqdm
import sys
import yaml

def lightgbm(X_train, y_train, X_test, X_test2):
    reg = lgb.LGBMRegressor(verbosity=-1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred2 = reg.predict(X_test2)
    return y_pred, y_pred2


def mlp(X_train, y_train, X_test, X_test2):
    reg = MLPRegressor(max_iter=500, learning_rate='adaptive', solver='adam', learning_rate_init=0.001)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred2 = reg.predict(X_test2)
    return y_pred, y_pred2


def rf(X_train, y_train, X_test, X_test2):
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred2 = reg.predict(X_test2)
    return y_pred, y_pred2


def xgboost(X_train, y_train, X_test, X_test2):
    reg = xgb.XGBRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred2 = reg.predict(X_test2)
    return y_pred, y_pred2


def svm(X_train, y_train, X_test, X_test2):
    reg = SVR(kernel="rbf")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred2 = reg.predict(X_test2)
    return y_pred, y_pred2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <config_name>")
        sys.exit(1)

    config_name = sys.argv[1]
    with open(f'config/{config_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    names = config.get('name', [])
    rs = config.get('r', [])
    if not isinstance(names, list):
        names = [names]
    if not isinstance(rs, list):
        rs = [rs]
    if len(names) != len(rs):
        print("Error: The length of 'name' and 'r' in the config file must be the same.")
        sys.exit(1)

    # List of models to use
    # models = ['svm', 'xgboost', 'lightgbm', 'mlp', 'rf']
    models = ['lightgbm']
    t = 0.2
    for name, r in zip(names, rs):
        try:
            data = pd.read_csv(f"output/{name}/bss.csv")
        except FileNotFoundError:
            print(f"File output/{name}/bss.csv not found. Skipping {name}.")
            continue

        X = data.iloc[:, :6]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # List of areas to process
        columns = data.columns[6:]

        # Loop through each model
        for m in models:
            predictions = {}
            mae_list = []
            mse_list = []
            rmse_list = []
            time_list = []

            for col in tqdm(columns, desc=f"Processing columns with {m} for {name}", unit="col"):
                y = data[col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=t, shuffle=False
                )
                X_test2 = X[r: r + 24, :]
                y_test2 = y.iloc[r: r + 24]

                start_time = time.time()
                if m == 'svm':
                    y_pred, y_pred2 = svm(X_train, y_train, X_test, X_test2)
                elif m == 'xgboost':
                    y_pred, y_pred2 = xgboost(X_train, y_train, X_test, X_test2)
                elif m == 'lightgbm':
                    y_pred, y_pred2 = lightgbm(X_train, y_train, X_test, X_test2)
                elif m == 'mlp':
                    y_pred, y_pred2 = mlp(X_train, y_train, X_test, X_test2)
                elif m == 'rf':
                    y_pred, y_pred2 = rf(X_train, y_train, X_test, X_test2)
                y_pred_rounded = np.round(y_pred).astype(int)
                y_pred_rounded2 = np.round(y_pred2).astype(int)
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_list.append(elapsed_time)

                predictions[col] = y_pred_rounded2

                mae = mean_absolute_error(y_test, y_pred_rounded)
                mse = mean_squared_error(y_test, y_pred_rounded)
                rmse = np.sqrt(mse)

                mae_list.append(mae)
                mse_list.append(mse)
                rmse_list.append(rmse)

            mae_avg = np.mean(mae_list)
            mse_avg = np.mean(mse_list)
            rmse_avg = np.mean(rmse_list)
            time_avg = np.mean(time_list)

            print(
                f"Model: {m}, Name: {name}, Average MAE: {mae_avg:.4f}, Average MSE: {mse_avg:.4f}, Average RMSE: {rmse_avg:.4f}, Average Time: {time_avg:.4f}"
            )

            predict_model_dir = f"output/{name}/predict/model"
            if not os.path.exists(predict_model_dir):
                os.makedirs(predict_model_dir)

            result_df = pd.DataFrame(predictions)

            result_df.to_csv(f"{predict_model_dir}/{m}.csv", index=False)

            predict_dir = f"output/{name}/predict"
            if not os.path.exists(predict_dir):
                os.makedirs(predict_dir)

            result_file_path = f"{predict_dir}/eva.csv"
            if os.path.exists(result_file_path):
                result_df_result = pd.read_csv(result_file_path)
            else:
                result_df_result = pd.DataFrame(columns=["model", "name", "MAE", "MSE", "RMSE", "time"])

            new_row = {
                "model": m,
                "name": name,
                "MAE": round(mae_avg, 4),
                "MSE": round(mse_avg, 4),
                "RMSE": round(rmse_avg, 4),
                "time": round(time_avg, 4),
            }

            result_df_result = pd.concat(
                [result_df_result, pd.DataFrame([new_row])], ignore_index=True
            )
            result_df_result.to_csv(result_file_path, index=False)