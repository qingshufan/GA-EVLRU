"""
File: Comparison of Fitness Metrics for Genetic Algorithm Results
Purpose: Calculate and compare fitness metrics for GA and GA-EVLRU results.
Key Features:
- Calculate best fitness, best generation, and average fitness.
Outputs:
- exr2.csv: A CSV file containing comparison results of fitness metrics for each area.
"""

import pandas as pd
import yaml
import sys


def calculate_metrics(fitness_data):
    best_fitness = min(fitness_data)
    best_generation = fitness_data.index(best_fitness)
    average_fitness = sum(fitness_data) / len(fitness_data)
    return best_fitness, best_generation, average_fitness


def read_fitness_data(file_path):
    fitness_data = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                values = line.strip().split()
                for value in values:
                    try:
                        fitness_data.append(float(value))
                    except ValueError:
                        print(
                            f"Warning: Unable to convert '{value}' to float, skipping this value."
                        )
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")
    return fitness_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ga_evaluation.py <config_name>")
        sys.exit(1)
    config_name = sys.argv[1]
    with open(f"config/{config_name}.yaml", "r") as file:
        config2 = yaml.safe_load(file)
    name = config2["name"][0]
    areas = config2["areas"]
    results = []
    for area in areas:
        original_file_path = f"output/{name}/ga/{area}/ga_best_fitness_data.txt"
        improved_file_path = f"output/{name}/ga/{area}/ga_evlru_best_fitness_data.txt"

        original_data = read_fitness_data(original_file_path)
        improved_data = read_fitness_data(improved_file_path)

        if original_data and improved_data:
            # Calculate metrics for the GA
            original_best, original_gen, original_avg_fitness = calculate_metrics(
                original_data
            )
            # Calculate metrics for the GA-EVLRU
            improved_best, improved_gen, improved_avg_fitness = calculate_metrics(
                improved_data
            )

            result = {
                "Area": f"{area}",
                "Pre_Best_Fitness": original_best,
                "Pre_Best_Generation": original_gen,
                "Pre_Avg_Fitness": original_avg_fitness,
                "Post_Best_Fitness": improved_best,
                "Post_Best_Generation": improved_gen,
                "Post_Avg_Fitness": improved_avg_fitness,
            }
            results.append(result)

    df = pd.DataFrame(results)

    try:
        with open("output/exr2.csv", "r") as f:
            df.to_csv("output/exr2.csv", mode='a', index=False, header=False)
    except FileNotFoundError:
        df.to_csv("output/exr2.csv", mode='w', index=False, header=True)
