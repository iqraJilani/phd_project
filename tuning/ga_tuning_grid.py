import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

from ga import GeneticAlgorithm


pop_size = 100
num_gen = 4
obj_func_weights = list([0.4, 0.4, 0.2])


param_grid = {
    "prob_crossover": [0.8, 0.9, 1.0],
    "prob_mutation": [0.2, 0.3, 0.4],
    }

param_list = [dict(zip(param_grid, values)) for values in zip(*param_grid.values())]

def evaluate_ga_fitness(ga_config):
    print(ga_config)
    p_c = ga_config["prob_crossover"]
    p_m = ga_config["prob_mutation"]
    ga = GeneticAlgorithm(pop_size, num_gen, p_c, p_m, *obj_func_weights, run_name="ga_advance_grid")
    a_s, p_r, lcoe, final_schedule, best_fv = ga.evaluate()
    #return a_s, p_r, lcoe, final_schedule
    return best_fv



param_list = [{key: [value] for key, value in params.items()} for params in [{k: v for k, v in zip(param_grid.keys(), values)} for values in zip(*param_grid.values())]]

# Create a Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scorer for fitness
scorer = make_scorer(lambda score: -score)  # Negative sign for maximizing fitness

# Perform grid search with Random Forest to tune parameters
grid_search = GridSearchCV(rf, param_list, cv=cv, scoring=scorer)
grid_search.fit([[0]] * len(param_list), [evaluate_ga_fitness(params) for params in param_list])
best_params = grid_search.best_params_
print("Best parameters:", best_params)