import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time
import wandb


def soc_factor(soc_stack, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges):
    for j in range(len(soc_stack) - len(soc_t), len(soc_stack)):

        # obtain least soc since last full charge
        if soc_stack[j, 0] < least_soc:
            least_soc = soc_stack[j, 0]
        elif soc_stack[j, 0] >= BATT_THRESH_MAX:
            least_soc = BATT_THRESH_MAX
        else:
            least_soc = least_soc
        l_soc = np.vstack((l_soc, least_soc))

        # time since last full charge
        if soc_stack[j, 0] >= BATT_THRESH_MAX:
            time_delta = 0.00001
        else:
            time_delta = time_delta + 1
        tslfc = np.vstack((tslfc, time_delta))

    # no of bad charges
        if (soc_stack[j-1, 0] > 0.90) and (soc_stack[j-1, 0] < BATT_THRESH_MAX) and (soc_stack[j, 0] < soc_stack[j-1, 0]) and (soc_stack[j-1, 0] > soc_stack[j-2, 0]):
            bad_charges = (0.0025-((0.95-soc_stack[j-1, 0])**2))/0.0025
        elif soc_stack[j, 0] >= BATT_THRESH_MAX:
            bad_charges = 0
        else:
            bad_charges = bad_charges
        n = np.vstack((n, bad_charges))

    return(l_soc, tslfc, n, least_soc, time_delta, bad_charges)


def soc_ch_dch(forecast_demand, energy_avail_gen, schedule, soc, least_soc):
    energy_total = np.empty((0, 1))  # hourly total energy (PV generation + available battery power)
    max_charge_power = 1000
    charge_power = np.empty((0, 1))  # hourly charge power of battery
    discharge_power = np.empty((0, 1))  # hourly discharge power of battery
    soc_t = np.empty((0, 1))
    for i in range(len(forecast_demand)):  # soc calculation model , soc, schedule

        battery_soc_to_full = BATT_CAP - (BATT_CAP*soc)  # energy required to fill up the battery

        # This block determines the charging/discharge power of the battery
        batt = energy_avail_gen['gen_cap'][i] - ((forecast_demand.iloc[i, :]*schedule[i, :]).sum())  # amount of energy to charge or discharge

        if (batt > 0) and (battery_soc_to_full > 0):  # PV generation is available and battery is not fully charged
            if (battery_soc_to_full > max_charge_power):
                if (batt > max_charge_power):
                    ch = max_charge_power
                else:
                    ch = batt
            else:
                if (batt > battery_soc_to_full):
                    ch = battery_soc_to_full
                else:
                    ch = batt
            energy_user_batt = 0  # energy discharged from battery

        elif (batt > 0) and (battery_soc_to_full == 0):
            ch = 0
            energy_user_batt = 0

        elif (batt < 0):
            energy_user_batt = -batt
            ch = 0

        elif(batt == 0):
            ch = 0
            energy_user_batt = 0

        charge_power = np.vstack((charge_power, ch))  # stack of hourly_charged_power
        discharge_power = np.vstack((discharge_power, energy_user_batt))  # stack of hourly_discharged_power

        energy_avail_total_single = energy_avail_gen['gen_cap'][i] + ((soc*BATT_CAP) - (BATT_THRESH_MIN*BATT_CAP))
        energy_total = np.vstack((energy_total, energy_avail_total_single))  # sum of avilable energy from battery and PV
        soc = (soc) + (ch/BATT_CAP) - (energy_user_batt/BATT_CAP)  # soc estimation model - coulomb counting
        soc_t = np.vstack((soc_t, soc))  # stack of hourly soc values per day
        
    return(soc_t, charge_power, discharge_power, energy_total)


def obj_func_1(forecast_demand, schedule, energy_avail_gen, abs_sat, soc, soc_stack, least_soc, time_delta, bad_charges, final_soc):
    '''This function computes the performance ratio of the per schedule solution'''
    y_f = (forecast_demand*schedule).sum().sum()  # calculation of daily final yield (total energy consumption) for scheduling solution.
    y_r = energy_avail_gen.sum().sum()  # calculation of daily reference yield (potential energy)
    a_s = ((abs_sat * schedule).sum().sum())/(abs_sat).sum().sum()  # calculation of daily absolute index
    p_r = y_f/y_r  # performance ratio /capacity utilisation factor
    bad_charges = 0  # no. of bad charges
    tslfc = np.empty((0, 1))  # hourly 'time since last full charge'
    n = np.empty((0, 1))  # hourly 'number of bad charges'
    l_soc = np.empty((0, 1))  # hourly 'least charge since last full recharge'

    soc_t, charge_power, discharge_power, energy_total = soc_ch_dch(forecast_demand, energy_avail_gen, schedule, soc, least_soc)
    soc_stack = np.vstack((soc_stack, soc_t))

    l_soc, tslfc, n = soc_factor(soc_stack, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[0:3]
    s = np.sum((n/10.8) + np.log(tslfc) + np.log(1-l_soc))
    s = (s + (16.1181*SLOT_DAY))/((5.2156*SLOT_DAY)+(16.1181*SLOT_DAY))
    fit_val = (W_1*p_r) + (W_2*a_s) - (W_3*s)  # weighted objective function
    soc_stack = final_soc  # redeclaring the original soc
    return(fit_val, soc_t, energy_total, charge_power, discharge_power, time_delta, least_soc, bad_charges)


def penalty(array, forecast_demand, energy_total, soc_d):
    'This penalty function contains the hard constraints'
    add_penalties = []

    for i in range(len(array)):
        hourly_final_yield = (array[i, :] * forecast_demand.iloc[i, :]).sum()
        if (hourly_final_yield > energy_total[i]):
            pen = PEN
            add_penalties = np.append(add_penalties, pen)

        if (soc_d[i] < BATT_THRESH_MIN):
            pen = PEN
            add_penalties = np.append(add_penalties, pen)

    sum_add_penalties = sum(add_penalties)
    return sum_add_penalties


def selection(init_pop, demand_sch, time_delta, bad_charges, least_soc):
    '''Selection by Tournament selection'''
    parents = np.empty((0, len(demand_sch.columns)))  # declares array for parents, two parents at a time
    for d in range(2):
        # random selection of 3 parents
        warrior_1_index = np.random.randint(0, pop)
        warrior_2_index = np.random.randint(0, pop)
        warrior_3_index = np.random.randint(0, pop)

        # This block ensures that the same parents are not selected more than once
        while warrior_1_index == warrior_2_index:
            warrior_1_index = np.random.randint(0, pop)

        while warrior_2_index == warrior_3_index:
            warrior_2_index = np.random.randint(0, pop)

        while warrior_3_index == warrior_1_index:
            warrior_3_index = np.random.randint(0, pop)

        # This block extracts the individual warriors from the initial population array.
        warrior_1 = init_pop[warrior_1_index*len(demand_sch):(warrior_1_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]
        warrior_2 = init_pop[warrior_2_index*len(demand_sch):(warrior_2_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]         
        warrior_3 = init_pop[warrior_3_index*len(demand_sch):(warrior_3_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]

        # Evaluation of objective function of randomly selected parents
        fit_val, soc_d, energy_total = obj_func_1(forecast_demand, warrior_1, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
        warrior_1_fitness = fit_val - penalty(warrior_1, forecast_demand, energy_total, soc_d)

        fit_val, soc_d, energy_total = obj_func_1(forecast_demand, warrior_2, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
        warrior_2_fitness = fit_val - penalty(warrior_2, forecast_demand, energy_total, soc_d)

        fit_val, soc_d, energy_total = obj_func_1(forecast_demand, warrior_3, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
        warrior_3_fitness = fit_val - penalty(warrior_3, forecast_demand, energy_total, soc_d)

        # selecting the warriors with the highest fitness functions as parents
        if warrior_1_fitness == max(warrior_1_fitness, warrior_2_fitness, warrior_3_fitness):
            winner = warrior_1

        elif warrior_2_fitness == max(warrior_1_fitness, warrior_2_fitness, warrior_3_fitness):
            winner = warrior_2

        else:
            winner = warrior_3

        parents = np.vstack((parents, winner))

    return(parents)


def crossover(parent_1, parent_2):
    
    rand_co = np.random.rand()  # choose a number at random
    if rand_co < p_c:  # check if number selected is less than probability of crossover

        r_r = np.random.randint(1, len(demand_sch))
        r_c = np.random.randint(1, len(demand_sch.columns)-1)

        rand_v_h = np.random.rand()

        if rand_v_h < 0.5: #horizontal crossover
            first_seg_par_1 = parent_1[:r_r, :]
            first_seg_par_2 = parent_2[:r_r, :]

            second_seg_par_1 = parent_1[r_r:, :]
            second_seg_par_2 = parent_2[r_r:, :]

            child_1 = np.concatenate((first_seg_par_1, second_seg_par_2))
            child_2 = np.concatenate((first_seg_par_2, second_seg_par_1))

            in_row_place_holder = copy.copy(child_1[r_r-1,r_c:])
            child_1[r_r-1,r_c:] = copy.copy(child_2[r_r-1,r_c:])
            child_2[r_r-1,r_c:] = copy.copy(in_row_place_holder)


        else:
            first_seg_par_1 = parent_1[:, : r_c]
            first_seg_par_2 = parent_2[:, : r_c]

            second_seg_par_1 = parent_1[:, r_c :]
            second_seg_par_2 = parent_2[:, r_c :]

            child_1 = np.concatenate((first_seg_par_1, second_seg_par_2), axis=1)
            child_2 = np.concatenate((first_seg_par_2, second_seg_par_1), axis=1)  

            in_col_place_id = copy.copy(child_1[r_r:,r_c-1])
            child_1[r_r:,r_c-1] = copy.copy(child_2[r_r:,r_c-1])
            child_2[r_r:,r_c-1] = copy.copy(in_col_place_id)
    
    else:
        child_1 = np.array(parent_1)
        child_2 = np.array(parent_2)

            
    return(child_1, child_2)


def mutation(child_1, child_2, time_delta, least_soc, bad_charges, ch_gen, dch_gen, energy_total_gen, fitness_values, soc_cummulative, soc_stack, new_population):

    for i in range(1, 3):
        rand_mut = np.random.rand()
        if rand_mut < p_m:
            mut_row = np.random.randint(0, np.size(globals()['child_' + str(i)], 0))
            mut_col = np.random.randint(0, np.size(globals()['child_' + str(i)], 1))

            if globals()['child_' + str(i)][mut_row][mut_col] == 0:
               globals()['child_' + str(i)][mut_row][mut_col] = 1

            else:
                globals()['child_' + str(i)][mut_row][mut_col] = 0

        globals()['mut_child_' + str(i)] = globals()['child_' + str(i)]

        globals()['of_mut_child_' + str(i)], soc_d, energy_total, charge_power, discharge_power = obj_func_1(forecast_demand, globals()['mut_child_' + str(i)], energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:5]
        ch_gen = np.vstack((ch_gen, charge_power))
        dch_gen = np.vstack((dch_gen, discharge_power))
        energy_total_gen = np.vstack((energy_total_gen, energy_total))
        globals()['pen_mut_child_' + str(i)] = penalty(globals()['mut_child_' + str(i)], forecast_demand, energy_total, soc_d)
        globals()['of_mut_child_' + str(i)] = globals()['of_mut_child_' + str(i)] - globals()['pen_mut_child_' + str(i)]
        fitness_values = np.vstack((fitness_values, globals()['of_mut_child_' + str(i)]))  # stack-up fitness values
        soc_cummulative = np.vstack((soc_cummulative, soc_d))  # stacks up the soc for each generation
        new_population = np.vstack((new_population, globals()['mut_child_' + str(i)]))
        soc_stack = final_soc

    return(fitness_values, new_population, soc_cummulative, ch_gen, dch_gen, energy_total_gen)

