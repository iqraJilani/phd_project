import concurrent.futures
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt import optimizer

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from ga import GeneticAlgorithm

# fixed battery properties and other fixed parameters such as no of slots in a day
#24 time slots in one day as scheduling horizon for scheduler is 24 hours and resolution time period is 1 hour
#Since the system uses persistance method for calculating user satisfaction/priority matrix, that's why week 1 slots are 168 (7 * 24 ) and week two slots are 336 (14 * 24)

# batt_properties = {
#     "batt_thresh_min": 0.4,
#     "batt_thresh_max": 0.99,
#     "batt_voltage": 48,
#     "batt_capacity": 10560,
#     "batt_cut_off":0.4,
#     "ch_eff": 0.99,
#     "dis_eff": 0.00139,
#     "dis_eff_cost": 0.885,
#     "c_soc_0":6.614 * (10**(-5)),
#     "c_soc_min":3.307 * (10**(-3)),
#     "c_batt_init":10000,
#     "unit_cost":0.0005,
#     "initial_cost":100000,
#     "annual_cost":100,
#     "R":0.07,
#     "pen":5000,
#     "month":12,
#     "month_slot":24,
#     "slot_day": 24,
#     "KWH_TP":5544000,
#     "week_one_slots":168,
#     "week_two_slots":336       
#     }


#GA parameters to be tuned
prob_crossover = 0.9
prob_mutation = 0.3
pop_size = 400
num_gen = 200
obj_func_weights = list([0.4, 0.4, 0.2])
config = {
    "prob_crossover": 0.9,
    "prob_mutation": 0.3,
    }

def ga_evaluate():
    ga = GeneticAlgorithm(pop_size, num_gen, prob_crossover, prob_mutation, *obj_func_weights, run_name="ga_untuned")
    a_s, p_r, lcoe, final_schedule, best_fv, duration = ga.evaluate() 
    print(a_s, p_r, lcoe, best_fv, duration)
    print({ 
            "task": "evaluation",
            "a_s": a_s,
            "p_r": p_r,
            "lcoe": lcoe,
            "best_fv": best_fv,
            "duration": duration
        })
    
    return best_fv



# def tuning_skopt():
#     # Define the parameter space for Bayesian Optimization
#     space = [Real(0.8, 0.9, name='prob_crossover'),
#             Real(0.2, 0.3, name='prob_mutation')]

#     # Perform Bayesian Optimization to tune the parameters
#     result = gp_minimize(ga_evaluate, space, n_calls=20, random_state=42)

#     # Get the best parameters found by Bayesian Optimization
#     best_params = result.x
#     best_fitness = result.fun

#     print("Best parameters:", best_params)
#     print("Best fitness:", best_fitness)

#def tuning_smac(ga_config):
#     config_space = ConfigurationSpace(ga_config)
#     scenario = Scenario(config_space, deterministic=True)
#     smac = HyperparameterOptimizationFacade(scenario, run_ga_smac)
#     incumbent = smac.optimize()
#     return incumbent


# tuners_list = [ga_evaluate, tuning_skopt]

# # #****** Thread pool Executor******

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = []
#     for tuner in tuners_list:
#         futures.append(executor.submit(tuner))
#     for future in concurrent.futures.as_completed(futures):
#         print(future.result())

best_fv = ga_evaluate()

# 

# #Ij_to_do: read data here and then send to ga
# #data = pd.read_excel('hall_declining_scenario_data.xlsx', 'Sheet3') 
# #Ij_to_do: replace train with evaluate/run
# a_s, p_r, lcoe, final_schedule = ga.train()




     