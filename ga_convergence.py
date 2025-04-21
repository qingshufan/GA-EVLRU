"""
File: Genetic Algorithm Comparison for EV Charging Strategy with and without LRU
Purpose: Compare genetic algorithms with and without LRU strategy to optimize charging schedules 
         for type A and B batteries in electric vehicles across multiple areas.
Key Features:
- Parallel processing of multiple areas' charging demand data.
- Configuration of genetic algorithm parameters: population size, crossover rate, mutation rate, etc.
- Separate individual generation functions for LRU and non-LRU cases.
- Computation of cost, satisfaction, and fitness metrics for solution evaluation.
- Selection, crossover, and mutation operations in the genetic algorithm loop.
- Saving best and average fitness values over generations for both algorithms.
Outputs:
- For each area:
    - ga_evlru_best_fitness_data.txt: Best fitness values over generations for GA with LRU.
    - ga_best_fitness_data.txt: Best fitness values over generations for GA without LRU.
"""
import random
import pandas as pd
import numpy as np
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

def cal_cost_max(DEMAND_A, DEMAND_B, PRICE):
    full_batteries_a = MAX_A
    empty_batteries_a = 0
    full_batteries_b = MAX_B
    empty_batteries_b = 0
    cost = 0
    i = 0
    for d in DEMAND_A:
        d = min(d, full_batteries_a)
        empty_batteries_a += d
        full_batteries_a -= d
        charge = d
        full_batteries_a += charge
        cost += PRICE[i] * PER_A * charge
        i += 1
    i = 0
    for d in DEMAND_B:
        d = min(d, full_batteries_b)
        empty_batteries_b += d
        full_batteries_b -= d
        charge = d
        full_batteries_b += charge
        cost += PRICE[i] * PER_B * charge
        i += 1
    return cost


def generate_individual_with_lru(MAX_A, MAX_B, DEMAND_A, DEMAND_B):
    individual = []
    full_batteries_a = MAX_A
    empty_batteries_a = 0
    full_batteries_b = MAX_B
    empty_batteries_b = 0
    j = 0
    for d in DEMAND_A:
        d = min(d, full_batteries_a)
        empty_batteries_a += d
        full_batteries_a -= d
        minx = empty_batteries_a
        if minx > 0:
            if DEMAND_A[j] > DEMAND_B[j]:  # LRU
                diff = min(DEMAND_A[j] - DEMAND_B[j], minx)
                charge = random.randint(diff, minx)
            else:
                charge = random.randint(0, minx)
        else:
            charge = 0
        full_batteries_a += charge
        empty_batteries_a -= charge
        individual.append(charge)
        j += 1
    j = 0
    for d in DEMAND_B:
        d = min(d, full_batteries_b)
        empty_batteries_b += d
        full_batteries_b -= d
        minx = empty_batteries_b
        if minx > 0:
            if DEMAND_B[j] > DEMAND_A[j]:  # LRU
                diff = min(DEMAND_B[j] - DEMAND_A[j], minx)
                charge = random.randint(diff, minx)
            else:
                charge = random.randint(0, minx)
        else:
            charge = 0
        full_batteries_b += charge
        empty_batteries_b -= charge
        individual.append(charge)
        j += 1
    return individual


def generate_individual_without_lru(MAX_A, MAX_B, DEMAND_A, DEMAND_B):
    individual = []
    full_batteries_a = MAX_A
    empty_batteries_a = 0
    full_batteries_b = MAX_B
    empty_batteries_b = 0
    for d in DEMAND_A:
        d = min(d, full_batteries_a)
        empty_batteries_a += d
        full_batteries_a -= d
        minx = empty_batteries_a
        if minx > 0:
            charge = random.randint(0, minx)
        else:
            charge = 0
        full_batteries_a += charge
        empty_batteries_a -= charge
        individual.append(charge)
    for d in DEMAND_B:
        d = min(d, full_batteries_b)
        empty_batteries_b += d
        full_batteries_b -= d
        minx = empty_batteries_b
        if minx > 0:
            charge = random.randint(0, minx)
        else:
            charge = 0
        full_batteries_b += charge
        empty_batteries_b -= charge
        individual.append(charge)
    return individual


def calculate_cost(individual, PRICE, PER_A, PER_B):
    cost = np.sum(individual[:24] * PRICE * PER_A) + np.sum(individual[24:] * PRICE * PER_B)
    return cost


def calculate_dissatisfaction(individual, DEMAND_A, DEMAND_B):
    total_demand_a = sum(DEMAND_A)
    total_demand_b = sum(DEMAND_B)
    total_supply_a = sum(individual[:24])
    total_supply_b = sum(individual[24:])
    dissatisfaction = abs(total_demand_a - total_supply_a) + abs(total_demand_b - total_supply_b)
    rate_dissatisfaction = dissatisfaction / (total_demand_a + total_demand_b)
    return rate_dissatisfaction


def fitness(individual, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max):
    cost = calculate_cost(individual, PRICE, PER_A, PER_B)
    rate_cost = cost / cost_max
    rate_dissatisfaction = calculate_dissatisfaction(individual, DEMAND_A, DEMAND_B)
    score = rate_cost + rate_dissatisfaction
    score += 2 if (1 - rate_dissatisfaction) < SA else 0
    return score


def selection(population, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max):
    selected = []
    for _ in range(POP_SIZE):
        i, j = random.sample(range(POP_SIZE), 2)
        if fitness(population[i], PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max) < fitness(population[j], PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max):
            selected.append(population[i])
        else:
            selected.append(population[j])
    return selected


def crossover(parent1, parent2):
    if random.random() < CROSS_RATE:
        cross_point = len(parent1) // 2
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]
        return child1, child2
    else:
        return parent1, parent2


def mutation(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(individual) - 1)
        if index < 24:
            individual[index] = random.randint(0, MAX_A)
        else:
            individual[index] = random.randint(0, MAX_B)
    return individual


def genetic_algorithm(generate_individual_func, alg, area, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B):
    fitness_values = []
    population = [generate_individual_func(MAX_A, MAX_B, DEMAND_A, DEMAND_B) for _ in range(POP_SIZE)]
    for _ in tqdm(range(GENERATIONS), desc=f"Iteration-{area}-{alg}"):
        new_population = []
        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = selection(population, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max)[i], selection(population, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max)[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutation(child1), mutation(child2)])
        population = new_population
        best_fitness = min([fitness(ind, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max) for ind in population])
        all_fitness = [fitness(ind, PRICE, PER_A, PER_B, DEMAND_A, DEMAND_B, cost_max) for ind in population]
        avg_fitness = np.mean(all_fitness)
        fitness_values.append((best_fitness, avg_fitness))
    return fitness_values


def run_ga_with_lru(args):
    area, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B = args
    return genetic_algorithm(generate_individual_with_lru, "GA-EVLRU", area, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B)


def run_ga_without_lru(args):
    area, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B = args
    return genetic_algorithm(generate_individual_without_lru, "GA-EV", area, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B)


def process_area(area,folder_path):
    data = pd.read_csv(f"{folder_path}/{area}_ab.csv")
    demand = data.values
    col = area
    DEMAND_A = demand[:, 0]
    DEMAND_B = demand[:, 1]
    PRICE = demand[:, 2]
    max_values = np.maximum(DEMAND_A, DEMAND_B)
    global_max_value = np.max(max_values)
    if global_max_value % 2 != 0:
        global_max_value += 1
    cost_max = cal_cost_max(DEMAND_A, DEMAND_B, PRICE)
    return col, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B


def call_function(func, arg):
    return func(arg)


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
    area_results = []
    with Pool() as pool:
        args = [(area, folder_path) for area in areas]
        area_results = pool.starmap(process_area, args)
    all_args = []
    for col, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B in area_results:
        args = (col, DEMAND_A, DEMAND_B, PRICE, MAX_A, MAX_B, cost_max, PER_A, PER_B)
        all_args.extend([(run_ga_with_lru, args), (run_ga_without_lru, args)])
    with Pool() as pool:
        results = pool.starmap(call_function, all_args)
    for i in range(0, len(results), 2):
        col = area_results[i // 2][0]
        ga_evlru_all_runs = [results[i]]
        ga_all_runs = [results[i + 1]]
        ga_evlru_best_fitness = np.array([[run[j][0] for j in range(GENERATIONS)] for run in ga_evlru_all_runs])
        ga_best_fitness = np.array([[run[j][0] for j in range(GENERATIONS)] for run in ga_all_runs])
        ga_evlru_avg_fitness = np.array([[run[j][1] for j in range(GENERATIONS)] for run in ga_evlru_all_runs])
        ga_avg_fitness = np.array([[run[j][1] for j in range(GENERATIONS)] for run in ga_all_runs])
        col_folder = os.path.join(output_folder, col)
        if not os.path.exists(col_folder):
            os.makedirs(col_folder)
        np.savetxt(os.path.join(col_folder, "ga_evlru_best_fitness_data.txt"), ga_evlru_best_fitness)
        np.savetxt(os.path.join(col_folder, "ga_best_fitness_data.txt"), ga_best_fitness)