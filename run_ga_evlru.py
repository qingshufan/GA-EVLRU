"""
File: GA-EVLRU charging strategy optimization main 
Purpose: Optimize charging schedules for type A/B batteries.
Key Features:
- Multi-area processing with parallel computation.
- Cost minimization & satisfaction maximization.
Outputs:
- record.csv per area: charging plans and storage status.
- result.csv: optimization metrics summary.
"""

import random
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
from multiprocessing import Pool
import yaml
import sys

with open('config/ga.yaml', 'r') as file:
    config = yaml.safe_load(file)

POP_SIZE = config['genetic_algorithm']['POP_SIZE']
CROSS_RATE = config['genetic_algorithm']['CROSS_RATE']
MUTATION_RATE = config['genetic_algorithm']['MUTATION_RATE']
GENERATIONS = config['genetic_algorithm']['GENERATIONS']
PER_A = eval(str(config['genetic_algorithm']['PER_A']))
PER_B = eval(str(config['genetic_algorithm']['PER_B']))
MAX_A = config['genetic_algorithm']['MAX_A']
MAX_B = config['genetic_algorithm']['MAX_B']
SA = config['genetic_algorithm']['SA']

# Process a single area
def process_area(name,area,folder_path):
    # Read demand data
    data = pd.read_csv(os.path.join(folder_path, f"{area}_ab.csv"))
    demand = data.values
    DEMAND_A = demand[:, 0]
    DEMAND_B = demand[:, 1]
    PRICE = demand[:, 2]
    global_max_value = np.max(np.maximum(DEMAND_A, DEMAND_B))
    MAX_A = 5  # Max number of type A batteries
    MAX_B = 8  # Max number of type B batteries

    # Calculate maximum cost
    def cal_cost_max():
        full_a, empty_a = MAX_A, 0
        full_b, empty_b = MAX_B, 0
        cost = 0
        for i, d in enumerate(DEMAND_A):
            d = min(d, full_a)
            empty_a += d
            full_a -= d
            charge = d
            full_a += charge
            cost += PRICE[i] * PER_A * charge
        for i, d in enumerate(DEMAND_B):
            d = min(d, full_b)
            empty_b += d
            full_b -= d
            charge = d
            full_b += charge
            cost += PRICE[i] * PER_B * charge
        return cost

    cost_max = cal_cost_max()

    # Generate an individual
    def generate_individual(MAX_A, MAX_B, DEMAND_A, DEMAND_B):
        individual = []
        full_a, empty_a = MAX_A, 0
        full_b, empty_b = MAX_B, 0
        for j, d in enumerate(DEMAND_A):
            d = min(d, full_a)
            empty_a += d
            full_a -= d
            minx = empty_a
            if minx > 0:
                if DEMAND_A[j] > DEMAND_B[j]:
                    diff = min(DEMAND_A[j] - DEMAND_B[j], minx)
                    charge = random.randint(diff, minx)
                else:
                    charge = random.randint(0, minx)
            else:
                charge = 0
            full_a += charge
            empty_a -= charge
            individual.append(charge)
        for j, d in enumerate(DEMAND_B):
            d = min(d, full_b)
            empty_b += d
            full_b -= d
            minx = empty_b
            if minx > 0:
                if DEMAND_B[j] > DEMAND_A[j]:
                    diff = min(DEMAND_B[j] - DEMAND_A[j], minx)
                    charge = random.randint(diff, minx)
                else:
                    charge = random.randint(0, minx)
            else:
                charge = 0
            full_b += charge
            empty_b -= charge
            individual.append(charge)
        return individual

    # Calculate cost
    def calculate_cost(individual):
        return np.sum(individual[:24] * PRICE * PER_A) + np.sum(
            individual[24:] * PRICE * PER_B
        )

    # Calculate satisfaction rate
    def calculate_satisfaction(individual):
        total_demand_a = sum(DEMAND_A)
        total_demand_b = sum(DEMAND_B)
        total_supply_a = sum(individual[:24])
        total_supply_b = sum(individual[24:])
        satisfaction = abs(total_demand_a - total_supply_a) + abs(
            total_demand_b - total_supply_b
        )
        rate_satisfaction = satisfaction / (total_demand_a + total_demand_b)
        return rate_satisfaction

    # Initialize population
    def init_population():
        return [
            generate_individual(MAX_A, MAX_B, DEMAND_A, DEMAND_B)
            for _ in range(POP_SIZE)
        ]

    # Calculate fitness
    def fitness(individual):
        rate_cost = calculate_cost(individual) / cost_max
        rate_satisfaction = calculate_satisfaction(individual)
        score = rate_cost + (1 - rate_satisfaction)
        score += 2 if (1 - rate_satisfaction) <= SA else 0
        return score

    # Selection
    def selection(population):
        return [
            population[
                (
                    random.sample(range(POP_SIZE), 2)[0]
                    if fitness(population[random.sample(range(POP_SIZE), 2)[0]])
                    < fitness(population[random.sample(range(POP_SIZE), 2)[1]])
                    else random.sample(range(POP_SIZE), 2)[1]
                )
            ]
            for _ in range(POP_SIZE)
        ]

    # Crossover
    def crossover(parent1, parent2):
        if random.random() < CROSS_RATE:
            cross_point = len(parent1) // 2
            return (
                parent1[:cross_point] + parent2[cross_point:],
                parent2[:cross_point] + parent1[cross_point:],
            )
        return parent1, parent2

    # Mutation
    def mutation(individual):
        return (
            generate_individual(MAX_A, MAX_B, DEMAND_A, DEMAND_B)
            if random.random() < MUTATION_RATE
            else individual
        )

    # Genetic algorithm main loop
    def genetic_algorithm():
        start_time = time.time()
        population = init_population()
        for _ in tqdm(range(GENERATIONS), desc=f"Iteration-{area}-GA-EVLRU"):
            new_population = []
            for i in range(0, POP_SIZE, 2):
                parent1, parent2 = (
                    selection(population)[i],
                    selection(population)[i + 1],
                )
                child1, child2 = crossover(parent1, parent2)
                new_population.extend([mutation(child1), mutation(child2)])
            population = new_population
        best_individual = min(population, key=fitness)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return best_individual[:24], best_individual[24:], best_individual, elapsed_time

    a_supply, b_supply, best_individual, elapsed_time = genetic_algorithm()
    init_cost = round(cost_max, 4)
    last_cost = round(calculate_cost(best_individual), 4)
    optimization = round((1 - (last_cost / init_cost)) * 100, 4)
    satisfaction_level = round((1 - calculate_satisfaction(best_individual)) * 100, 4)
    elapsed_time = round(elapsed_time, 4)

    # Calculate storage
    storage_a = []
    stor_a = 5
    for i in range(len(DEMAND_A)):
        v = max(stor_a - DEMAND_A[i], 0)
        stor_a = v
        storage_a.append(v + a_supply[i])
        stor_a += a_supply[i]
    storage_b = []
    stor_b = 8
    for i in range(len(DEMAND_B)):
        v = max(stor_b - DEMAND_B[i], 0)
        stor_b = v
        storage_b.append(v + b_supply[i])
        stor_b += b_supply[i]

    col_folder_path = f"output/{name}/ga/{area}"
    if not os.path.exists(col_folder_path):
        os.makedirs(col_folder_path)

    record_df = pd.DataFrame(
        {
            "timestamp": range(24),
            "demand_A": DEMAND_A.astype(int),
            "plan_charge_A": np.array(a_supply).astype(int),
            "fully_A": np.array(storage_a).astype(int),
            "demand_B": DEMAND_B.astype(int),
            "plan_charge_B": np.array(b_supply).astype(int),
            "fully_B": np.array(storage_b).astype(int),
            "price": PRICE,
        }
    )
    record_df.to_csv(os.path.join(col_folder_path, "record.csv"), index=False)
    optimization = f"{optimization}%"
    satisfaction_level = f"{satisfaction_level}%"
    return {
        "area": area,
        "init_cost": init_cost,
        "last cost": last_cost,
        "optimization": optimization,
        "satisfaction level": satisfaction_level,
        "time": elapsed_time,
        "ymax": np.array(global_max_value).astype(int),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ga_convergence.py <config_name>")
        sys.exit(1)
    config_name = sys.argv[1]
    with open(f'config/{config_name}.yaml', 'r') as file:
        config2 = yaml.safe_load(file)
    name = config2["name"][0]
    areas = config2["areas"]
    folder_path = f"output/{name}/predict/area"
    output_folder = f"output/{name}/ga"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    results = []
    with Pool() as pool:
        args = [(name,area, folder_path) for area in areas]
        results = pool.starmap(process_area, args)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"output/{name}/ga/result.csv", index=False)
