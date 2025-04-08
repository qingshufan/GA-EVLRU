"""
File: Evaluation of Estimated Battery Swapping Demand 
Purpose: Calculate and evaluate the trend stability, periodicity matching degree, 
         and outlier ratio of Estimated Battery Swapping Demand for EV-Load-Open-Data.
Key Features:
- Calculation of trend stability by segmenting the curve and analyzing slopes.
- Determination of periodicity matching degree using Fast Fourier Transform (FFT) 
  and cosine similarity with expected periods.
- Computation of outlier ratio based on mean and standard deviation of the curve.
- Reading data from CSV files and saving evaluation results if required.
Outputs:
- Console output of evaluation metrics for each dataset.
- output/exr1.csv: Appended evaluation results (stability, periodicity, outlier ratio) for each dataset.
"""
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.spatial.distance import cosine

def trend_stability(curve, time_periods=5):
    segment_length = len(curve) // time_periods
    slopes = []
    for i in range(time_periods):
        start = i * segment_length
        end = start + segment_length
        x = np.arange(segment_length)
        y = curve[start:end]
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return np.std(slopes)


def periodicity_match(curve, expected_periods):
    n = len(curve)
    yf = fft(curve)
    xf = np.fft.fftfreq(n)
    simulated_periods = []
    for i in range(n // 2):
        if abs(xf[i]) == 0:
            continue
        period = 1 / abs(xf[i])
        if period > 1:
            simulated_periods.append((period, abs(yf[i])))
    simulated_periods.sort(key=lambda x: x[1], reverse=True)
    simulated_periods = [p[0] for p in simulated_periods[:len(expected_periods)]]
    similarity = 1 - cosine(simulated_periods, expected_periods)
    return similarity


def outlier_ratio(curve):
    mean = np.mean(curve)
    std = np.std(curve)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = [x for x in curve if x < lower_bound or x > upper_bound]
    return len(outliers) / len(curve)



names = ['acn', 'boulder_2021', 'palo_alto', 'sap', 'dundee', 'paris', 'perth']
area = "Demand"
has_output = True

for name in names:
    try:
        df = pd.read_csv(f'output/{name}/bss.csv')
        # Get simulated curve data
        simulated_curve = df[area]
        # Calculate various metrics
        stability = trend_stability(simulated_curve)
        periodicity = periodicity_match(simulated_curve, expected_periods=[24, 168])
        outlier = outlier_ratio(simulated_curve)

        print(f"Name: {name}")
        print(f"Trend stability: {stability}")
        print(f"Periodicity matching degree: {periodicity}")
        print(f"Outlier ratio: {outlier}")
        if has_output:
            df = pd.DataFrame([[stability, periodicity, outlier]])
            df.to_csv('output/exr1.csv', mode='a', header=False, index=False)
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: File output/{name}/bss.csv not found.")
    except KeyError:
        print(f"Error: Column {area} not found in output/{name}/bss.csv.")
    