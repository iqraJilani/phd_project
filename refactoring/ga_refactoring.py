# %%
import copy
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import wandb

now = datetime.now()
run_name = "ga_refactoring"
current_datetime =now.strftime("%d/%b/%Y_%H:%M:%S")
run_name = "_".join([run_name, current_datetime])

wandb.login(
    key="a9cbb8af7a708fc893fe0aaf2f50d9e51aed6f34"
)
wb = wandb.init(
    project = "GA_Parameter_Tuning",
    name = run_name,
    config ={
        "initial_prob_mutation": 0.3,
        "initial_prob_crossover": 0.9,
        "no_of_gen": 2,
        "population_szie": 2,
    }, 
    allow_val_change=True)

wb.define_metric("gen_num")
wb.define_metric("day_num")

wb.define_metric("best_fv_gen", step_metric="gen_num")
wb.define_metric("best_fv_day", step_metric="day_num")

results_table = wandb.Table(columns = ["Run_Name", "Performance_Ratio", "Absolute_Sat", "LCOE", "Best_fv", "Duration"])
#wb.log({"Results": results_table})

best_fv_days = np.empty((0, 1))





# %%
iter_no = 5
con = 25


W_1 = 0.4  # weight of objective 1
W_2 = 0.4  # weight of objective 2
W_3 = 0.2  # weight of objective 3


CH_EFF = 0.99  # battery charge efficiency
DIS_EFF = 0.00139  # battery discharge efficiency
BATT_CAP = 10560  # battery capacity
V_BATT = 48  # battery voltage
C_SOC_0 = 6.614 * (10**(-5))
C_SOC_MIN = 3.307 * (10**(-3))
C_BATT_INIT = 10000
DIS_EFF_COST = 0.885
BATTERY_CUT_OFF = 0.4  # soc threshold/ cut-off soc
ANNUAL_COST = 100
R = 0.07
INITIAL_COST = 100000
UNIT_COST = 0.0005
BATT_THRESH_MIN = 0.4
BATT_THRESH_MAX = 0.99
PEN = 5000
SLOT_DAY = 24
MONTH_SLOT = 24
KWH_TP = 5544000
MONTH = 12
WEEK_ONE_SLOTS = 168
WEEK_TWO_SLOTS = 336


# %%
globals()['as_day'] = 0
globals()['bl_day'] = 0
globals()['pr_day'] = 0

# %%


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



# %%


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



# %%


def obj_func(forecast_demand, schedule, energy_avail_gen, abs_sat, soc, soc_stack, least_soc, time_delta, bad_charges, final_soc):
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
    #print("Inside:", soc_stack.shape)
    globals()['as_day'] = a_s
    globals()['bl_day'] = s
    globals()['pr_day'] = p_r
    return(fit_val, soc_t, energy_total, charge_power, discharge_power, time_delta, least_soc, bad_charges)



# %%


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



# %%


def selection(init_pop, time_delta, bad_charges, least_soc, final_soc, demand_sch, forecast_demand, energy_avail_gen, abs_sat, soc_stack):
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
        fit_val, soc_d, energy_total = obj_func(forecast_demand, warrior_1, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
        warrior_1_fitness = fit_val - penalty(warrior_1, forecast_demand, energy_total, soc_d)

        fit_val, soc_d, energy_total = obj_func(forecast_demand, warrior_2, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
        warrior_2_fitness = fit_val - penalty(warrior_2, forecast_demand, energy_total, soc_d)

        fit_val, soc_d, energy_total = obj_func(forecast_demand, warrior_3, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
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



# %%
def crossover(parent_1, parent_2, p_c, demand_sch):
    
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



# %%


def mutation(p_m, child_1, child_2, time_delta, bad_charges, least_soc, final_soc, soc_stack, soc_cummulative, ch_gen, dch_gen, energy_total_gen, energy_avail_gen, forecast_demand, abs_sat, fitness_values, new_population):
    
    globals()['child_1'] = child_1
    globals()['child_2'] = child_2
    
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

        globals()['of_mut_child_' + str(i)], soc_d, energy_total, charge_power, discharge_power = obj_func(forecast_demand, globals()['mut_child_' + str(i)], energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:5]
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



# %%
def create_initial_pop(demand_sch):
    ''' The algorithm generates it's initial population by picking a column and shuffle the chromosomes in that column'''
    init_pop = np.empty((0, n_loads))
    for a in range(int(pop)):
        new_sch = demand_sch.apply(np.random.permutation, axis=1)
        new_sch = pd.DataFrame([list(x) for x in new_sch])
        init_pop = np.vstack((init_pop, new_sch))  # stack the populations together
#         lsoc_stack_gen = np.empty((0, 1))
#         tslfc_stack_gen = np.empty((0, 1))
#         n_stack_gen = np.empty((0, 1))
    return init_pop

    

# %%
def calculate_user_satisfaction(demand_data, n_days, n_loads):

    
    #globals, to be declared as self variables
#     n_days = int(len(demand_data)/SLOT_DAY)
#     n_loads = len(demand_data.columns)
    abs_sat_total = np.empty(((n_days*SLOT_DAY), n_loads))
    sat_index = 0
    
    for d in range(14, n_days):

        day = demand_data.iloc[(d*SLOT_DAY):(d*SLOT_DAY+SLOT_DAY)]
        one_week = demand_data.iloc[((d*SLOT_DAY)-WEEK_ONE_SLOTS):((d*SLOT_DAY)-WEEK_ONE_SLOTS)+SLOT_DAY]
        two_week = demand_data.iloc[((d*SLOT_DAY)-WEEK_TWO_SLOTS):((d*SLOT_DAY)-WEEK_TWO_SLOTS)+SLOT_DAY]

        day = day.reset_index(drop=True)
        one_week = one_week.reset_index(drop=True)
        two_week = two_week.reset_index(drop=True)

        day_sat = (day + one_week + two_week)/3
        #print(day_sat.shape)# Find the average
        
        abs_sat_total[(sat_index*SLOT_DAY): ((sat_index*SLOT_DAY)+SLOT_DAY)] = day_sat
        sat_index = sat_index + 1


    abs_sat_total = pd.DataFrame(abs_sat_total)
    abs_sat_total.columns = demand_data.columns
    return abs_sat_total
    

# %%
def calculate_fitness( time_delta, bad_charges, least_soc, final_soc,  soc_stack, demand_sch, forecast_demand, energy_avail_gen, abs_sat, n_loads, prob_m = 0.3, prob_c = 0.9):

    
    
    obj_bl_day = 0
    obj_pr_day = 0
    obj_as_day = 0

    # declaration of size of initial population
    sch_temp = len(demand_sch.columns)

    #final_schedule_sorting = np.empty((0, len(demand_sch.columns)))
    mean_fitness_value = np.empty((gen, 1))
    best_fv = np.empty((gen, 1)) # stack of best fitness values in each generation
    best_in_generations = np.empty(((gen * SLOT_DAY), len(demand_sch.columns)))  # stack of best schedules in generations.
    best_soc_in_generations = np.empty(((gen * SLOT_DAY), 1))  # stack of best soc in generations
    best_ch_in_generations = np.empty(((gen * SLOT_DAY), 1))  # stack of best charges in generations
    best_dch_in_generations = np.empty(((gen * SLOT_DAY), 1))  # stack of best discharges in generations
    best_et_in_generations = np.empty((0, 1))  # stack of best energy_total in every generation
    
    #tslfc = np.empty((0, 1))  # hourly time since last full charge
    #n = np.empty((0, 1))  # hourly number of bad charges
    #l_soc = np.empty((0, 1))  # least soc since last full charge

    capacity_shortage = 0  # capacity shortage value
    energy_avail_total = np.array(forecast_demand.sum(axis=1)*(1-capacity_shortage))
    best_energy_avail_total_generations = np.empty((0, 1))
    battery_current = np.empty((0, 1))  # battery current
    soc_cummulative = np.empty((0, 1))  # stack of all soc for all chromosomes/ schedules in a generation
    start = time.time()
    # Definition of GA parameters
    p_c = prob_c
    p_m = prob_m
    k = 3
    n_loads = len(demand_sch.columns)
    #init_pop = np.empty((0, len(demand_sch.columns)))


    # Generation of initial population
    init_pop = create_initial_pop(demand_sch)


    stop_crit_counter = 0
    init_fv = -300
    # generation iteration
    for g in range(int(gen)):

        print(f"gen no: {g}")

        fitness_values = np.empty((0, 1))  # declare array of fitness values
        new_population = np.empty((0, len(demand_sch.columns)))  # declare array of new population in new generation
        soc_cummulative = np.empty((0, 1))
        energy_available_total_cummulative = np.empty((0, 1))
        ch_gen = np.empty((0, 1))
        dch_gen = np.empty((0, 1))
        energy_total_gen = np.empty((0, 1))

        for c in range(int(pop/2)):

            parents = selection(init_pop, time_delta, bad_charges, least_soc, final_soc, demand_sch, forecast_demand, energy_avail_gen, abs_sat, soc_stack)
            parent_1 = parents[0:len(demand_sch), :]
            parent_2 = parents[len(demand_sch):, :]
            child_1 = np.empty((0, len(demand_sch)))
            child_2 = np.empty((0, len(demand_sch)))
            child_1, child_2 = crossover(parent_1, parent_2, p_c,demand_sch)
            globals()['child_1'] = child_1
            globals()['child_2'] = child_2
            #print("Outside:", soc_stack.shape)
            fitness_values, new_population, soc_cummulative, ch_gen, dch_gen, energy_total_gen = mutation(p_m, child_1, child_2, time_delta, bad_charges, least_soc, final_soc, soc_stack, soc_cummulative, ch_gen, dch_gen, energy_total_gen, energy_avail_gen, forecast_demand, abs_sat, fitness_values, new_population)

        sorted_index = np.argsort(fitness_values[:, 0])[::-1]  # sort position of fitness values in descending order #give position
        sorted_fitness_value = np.sort(fitness_values[:, 0])[::-1]  # sort in descending order
        best = new_population[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]  # fetch the best schedule from the stack of schedules in generation.
        best_soc = soc_cummulative[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_charge_power = ch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_discharge_power = dch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_energy_total = energy_total_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        
        #mean_fitness_value = np.vstack((mean_fitness_value, np.mean(sorted_fitness_value)))
        mean_fitness_value[g] = np.mean(sorted_fitness_value)
        #best_fv = np.vstack((best_fv, sorted_fitness_value[0]))  # stacks the best fitness value in every generation
        best_fv[g] = sorted_fitness_value[0]
        #best_in_generations = np.vstack((best_in_generations, best))  # stacks the best schedules in each generation together
        best_in_generations[(g*SLOT_DAY): ((g*SLOT_DAY)+SLOT_DAY)] = best
        #best_soc_in_generations = np.vstack((best_soc_in_generations, best_soc))
        best_soc_in_generations[(g*SLOT_DAY): ((g*SLOT_DAY)+SLOT_DAY)] = best_soc
        #best_ch_in_generations = np.vstack((best_ch_in_generations, best_charge_power))
        best_ch_in_generations[(g*SLOT_DAY): ((g*SLOT_DAY)+SLOT_DAY)] = best_charge_power
        #best_dch_in_generations = np.vstack((best_dch_in_generations, best_discharge_power))
        best_dch_in_generations[(g*SLOT_DAY): ((g*SLOT_DAY)+SLOT_DAY)] = best_discharge_power
        best_et_in_generations = np.vstack((best_et_in_generations, best_energy_total))
        #best_et_in_generations[(g*SLOT_DAY): ((g*SLOT_DAY)+SLOT_DAY)] = best_energy_total
        
        
        init_pop = new_population
        soc_stack = final_soc

        best_fv_gen = float(max(fitness_values))
        wb.log({"best_fv_gen": best_fv_gen, "gen_num": g}, commit=True)

        if sorted_fitness_value[0] - init_fv == 0:
            stop_crit_counter = stop_crit_counter + 1

        else:
            stop_crit_counter = 0

        if stop_crit_counter == con:
            break

        init_fv = sorted_fitness_value[0]

    sorted_schedule_index = np.argsort(best_fv[:, 0])[::-1]  # sorts the fitness values of the generational best values
    final_schedule = best_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_soc = best_soc_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_ch = best_ch_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_dch = best_dch_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_energy_total = best_et_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_energy_avail_total = best_energy_avail_total_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    
    
    
    return best_fv, final_schedule, final_soc, final_ch, final_dch, final_energy_total, final_energy_avail_total

# %%
def day_calculation(u, data, demand_data, abs_sat_total, final_soc, soc_stack, least_soc, soc, time_delta, bad_charges):
    
    #print("global soc before: ", globals()['soc_prev'])
    
    tslfc = np.empty((0, 1))  # hourly time since last full charge
    n = np.empty((0, 1))  # hourly number of bad charges
    l_soc = np.empty((0, 1)) 
        
    day_data = data.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # picks the consumption data for day x
    forecast_demand = day_data.filter(regex='CPE|LVL')  # extract CPE and socket data
    forecast_demand = forecast_demand.reset_index(drop=True)

    energy_avail_gen = pd.DataFrame(day_data['Potential_PV_power_W'])  # Potential PV generation data of day x
    energy_avail_gen.columns = ['gen_cap']  # renaming the PV generation data column
    energy_avail_gen = energy_avail_gen.reset_index(drop=True)
    abs_sat = abs_sat_total.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # absolute satisfaction of day x
    abs_sat = abs_sat.reset_index(drop=True)
    demand_sch = demand_data.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # binary demand of day x

    demand_sch = demand_sch.reset_index(drop=True)
    demand_sch = demand_sch.astype(int)
    actual_demand = copy.copy(demand_sch)
    schedule = np.array(copy.copy(demand_sch))  # Copy of schedule
    
    #self.soc_stack_prev = final_soc
    start = time.time()
    
    
    best_fv, final_schedule, final_soc, final_ch, final_dch, final_energy_total, final_energy_avail_total = calculate_fitness(time_delta, bad_charges, least_soc, final_soc,  soc_stack, demand_sch, forecast_demand, energy_avail_gen, abs_sat, n_loads, prob_m = 0.3, prob_c = 0.9)

    #print("final ch day: ", final_ch)
    soc_stack = final_soc
    soc = final_soc[-1][0]
    
    least_soc, time_delta, bad_charges = soc_factor(soc_stack, final_soc, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[3:]
    
    end = time.time()
    dur = end - start
    dur = int(dur/60)
    wb.log({"duration_day": dur, "day_num": u}, commit=True)
    
    best_fv_day = float(max(best_fv))
    wb.log({"best_fv_day": best_fv_day, "day_num": u}, commit=True)
    

    wb.log({"obj_func1_day": globals()['bl_day'], "day_num": u}, commit=True)
    wb.log({"obj_func2_day": globals()['as_day'], "day_num": u}, commit=True)
    wb.log({"obj_func3_day": globals()['pr_day'], "day_num": u}, commit=True)
    
    globals()['soc_prev'] = soc
    #print("global soc after: ", globals()['soc_prev'])
    globals()['final_soc_prev'] = final_soc
    globals()['soc_stack_prev'] = final_soc_prev
    #soc_stack_prev = soc_stack
    globals()['least_soc_prev'] = least_soc
    globals()['time_delta_prev'] = time_delta
    globals()['bad_charges_prev'] = bad_charges
    globals()['duration_prev'] = dur
    
    #globals()['best_fv_days'] = np.vstack((globals()['best_fv_days'], best_fv_day))
    globals()['best_fv_days'][u] = best_fv_day
    #globals()['duration'] = np.vstack((globals()['duration'], dur))
    globals()['duration_days'][u] = dur
    #globals()['final_schedule_cum'] = np.vstack((globals()['final_schedule_cum'], final_schedule))
    globals()['final_schedule_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_schedule
    #globals()['final_soc_cum'] = np.vstack((globals()['final_soc_cum'], final_soc))
    globals()['final_soc_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_soc
    #globals()['soc_prev_cum'] = np.vstack((globals()['soc_prev_cum'], soc))
    globals()['soc_prev_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = soc
    #globals()['final_ch_cum'] = np.vstack((globals()['final_ch_cum'], final_ch))
    globals()['final_ch_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_ch
    #globals()['final_dch_cum'] = np.vstack((globals()['final_dch_cum'], final_dch))
    globals()['final_dch_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_dch
    globals()['final_energy_total_cum'] = np.vstack((globals()['final_energy_total_cum'], final_energy_total))
    #globals()['final_energy_total_cum'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_energy_total
    globals()['final_energy_avail_total'] = np.vstack((globals()['final_energy_avail_total'], final_energy_avail_total))
    #globals()['final_energy_avail_total'] [(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_energy_avail_total
    
    return dur, best_fv_day

# %%


# user-satisfaction
data = pd.read_excel('scenario_1.xlsx', 'Sheet3')  # energy usage data
data['Potential_PV_power_W'][data['Potential_PV_power_W'] < 0] = 0
data = data.drop(['CPEHC', 'CPER1', 'CPESC'], 1)

demand_data = copy.copy(data)

data.drop(index=data.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
data = data.reset_index(drop=True)

demand_data = demand_data.filter(regex='CPE|LVL')  # extract CPE and socket data
demand_data = demand_data.astype(float)
demand_data.values[demand_data > 0] = 1


n_days = int(len(demand_data)/SLOT_DAY)
n_loads = len(demand_data.columns)

abs_sat_total = calculate_user_satisfaction(demand_data, n_days, n_loads)

demand_data.drop(index=demand_data.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
demand_data = demand_data.reset_index(drop=True)

best_fv_days = np.empty((n_days, 1))
final_schedule_cum = np.empty(((n_days * SLOT_DAY), len(demand_data.columns)))
final_soc_cum = np.empty(((n_days * SLOT_DAY), 1))
soc_prev_cum = np.empty(((n_days * SLOT_DAY), 1))
final_ch_cum = np.empty(((n_days * SLOT_DAY), 1))
final_dch_cum = np.empty(((n_days * SLOT_DAY), 1))
final_energy_total_cum = np.empty((0, 1))
final_energy_avail_total = np.empty((0, 1))
duration_days = np.empty((n_days, 1))

soc_prev = data.loc[0, 'Battery Monitor State of charge %']/100
least_soc_prev = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
final_soc_prev = np.full((SLOT_DAY, 1), soc_prev)
soc_stack_prev = final_soc_prev
time_delta_prev = 0.00001  # time since last full charge
bad_charges_prev = 0  # number of bad charges
dur_prev = 0

cap_shortage_prev = 0


pop = 400
gen = 200
p_c = 0.9
p_m = 0.3
best_fv_day = 0

day_counter = 0


# %%
for u in range(int((len(data))/SLOT_DAY)):
    print(f"DAY no: {u}")
    if u==6:
        break
    dur_prev, best_fv_day = day_calculation(u, data, demand_data, abs_sat_total, final_soc_prev, soc_stack_prev, least_soc_prev, soc_prev, time_delta_prev, bad_charges_prev)
    duration_days[u] = dur_prev
    best_fv_days[u] = best_fv_day
    day_counter = day_counter +1 

# %%
max(best_fv_days)
min(duration_days)

# %%
# ###### start_day = (day_counter*SLOT_DAY)-SLOT_DAY
# end_day = (day_counter*SLOT_DAY)

# last_day_data = data.iloc[start_day:end_day, :]
# last_day_demand_data = demand_data.iloc[start_day:end_day, :]
# last_day_final_schedule_cum = final_schedule_cum[start_day:end_day, :]
# last_day_final_soc_cum = final_soc_cum[start_day:end_day, :]
# last_day_soc_prev_cum =  soc_prev_cum[start_day:end_day, :]
# last_day_final_ch_cum =  final_ch_cum[start_day:end_day, :]
# last_day_final_dch_cum = final_dch_cum[start_day:end_day, :]
# last_day_final_energy_total_cum = final_energy_total_cum[start_day:end_day, :]
# last_day_final_energy_avail_total = final_energy_avail_total[start_day:end_day, :]



# %%
# e_load = data.filter(regex='CPE|LVL').iloc[0:day_counter*SLOT_DAY, :]*final_schedule_cum[:, :]
# e_demand = data.filter(regex='CPE|LVL').iloc[0:day_counter*SLOT_DAY, :]
# e_demand_plot = np.sum(e_demand, axis=1)
# e_load_plot = np.sum(e_load, axis=1)

# pv_load = np.empty((0, 1))

# for i in range(len(final_schedule_cum)):
#     if data['Potential_PV_power_W'][i] > 0:
#         if (data['Potential_PV_power_W'][i] - final_ch_cum[i]) >= e_load_plot[i]:
#             pv_load_t = e_load_plot[i]

#         elif (data['Potential_PV_power_W'][i] - final_ch_cum[i]) < e_load_plot[i]:
#             pv_load_t = data['Potential_PV_power_W'][i] - final_ch_cum[i]

#     else:
#         pv_load_t = 0

#     pv_load = np.vstack((pv_load, pv_load_t))

# y_f = (data.filter(regex='CPE|LVL').iloc[0:day_counter*SLOT_DAY, :]*final_schedule_cum[:, :]).sum().sum()  # [:,:-1] calculation of daily final yield for scheduling solution.
# y_r = data['Potential_PV_power_W'].sum().sum()  # calculation of daily reference yield
# p_r = y_f/y_r

# a_s = ((abs_sat_total.iloc[0:day_counter*SLOT_DAY, :] * final_schedule_cum[:, :]*demand_data).sum().sum())/(abs_sat_total.iloc[0:day_counter*SLOT_DAY, :]*demand_data).sum().sum()  # calculation of daily absolute index




# # Calculating the hourly battery current
# battery_current = np.empty((0, 1))
# for p in range(len(final_dch_cum)):
#     if final_dch_cum[p] > 0 and final_ch_cum[p] == 0:
#         i_bat = -final_dch_cum[p]/V_BATT

#     elif final_ch_cum[p] > 0 and final_dch_cum[p] == 0:
#         i_bat = (final_ch_cum[p]/V_BATT)*CH_EFF

#     elif final_ch_cum[p] == 0 and final_dch_cum[p] == 0:
#         i_bat = 0

#     battery_current = np.vstack((battery_current, i_bat))

# tslfc_cum = np.empty((0, 1))  # hourly time since last full charge
# n_cum = np.empty((0, 1))  # hourly number of bad charges
# l_soc_cum = np.empty((0, 1))

# first_discharge = np.empty((0, 1))

# if battery_current[0, 0] < 0:
#     f_d = battery_current[0, 0]

# else:
#     f_d = -0.0001

# least_soc = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
# time_delta = 0.00001  # time since last full charge
# bad_charges = 0  # number of bad charges

# tslfc = np.empty((0, 1))  # hourly time since last full charge
# n = np.empty((0, 1))  # hourly number of bad charges
# l_soc = np.empty((0, 1))

# soc_t = np.empty((MONTH_SLOT, 1))
# l_soc_cum, tslfc_cum, n_cum = soc_factor(final_soc_cum, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[0:3]

# # Calculating first_discharge current after full charge
# for j in range(1, len(final_soc_cum)):
#     if (final_soc_cum[j, 0] < BATT_THRESH_MAX) and (final_soc_cum[j-1, 0] >= BATT_THRESH_MAX):
#         f_d = battery_current[j, 0]
#     else:
#         f_d = f_d

#     first_discharge = np.vstack((first_discharge, f_d))

# if first_discharge[0, 0] == -0.0001:
#     first_discharge = np.vstack((-0.0001, first_discharge))
# else:
#     first_discharge = np.vstack((first_discharge[0, 0], first_discharge))

# # calculating weighted throughput and battery lifetime

# last_day_first_discharge= first_discharge[((day_counter*SLOT_DAY)-SLOT_DAY): (day_counter*SLOT_DAY), :]
# f_i_n = ((-43.9/last_day_first_discharge)**(1/2)) * ((np.exp(n_cum/3.6))**(1/3))
# f_soc = 1 + (C_SOC_0 + C_SOC_MIN*(1-l_soc_cum))*f_i_n*tslfc_cum#
# weighted_throughput = f_soc * -last_day_final_dch_cum
# weighted_througput_cummulative = np.sum(weighted_throughput) * MONTH
# battery_lifetime = KWH_TP / -(weighted_througput_cummulative)

# cap_shortage = (e_demand_plot.sum() - y_f) * MONTH


# num = 0
# den = 0
# for c in range(int(battery_lifetime)):
#     num = num + (y_f*UNIT_COST*MONTH/(1+R)**c)

#     den = den + ((y_f*MONTH)/(1+R)**c)


# lcoe = (INITIAL_COST + num) / den

# least_dur = min(duration)


# results = {
#     "performance_ratio": p_r,
#     "absolute_satisfaction": a_s,
#     "least_cost_of_energy": lcoe,
#     "duration": least_dur
    
# }

# max_fv = max(best_fv_days)[0]
# min_duration = min(duration)[0]


# # results_table = pd.DataFrame(results)
# # wb_table = wandb.Table(dataframe=results_table)
# # wb.log({"Resuls_Table": wb_table})

# %%
# results_table.add_data(run_name, p_r, a_s, lcoe, max_fv, min_duration)
# wb.log({"Evaluation_Results": results_table})

# %%
# wb.finish()

# %%
# soc shape, least_soc_shape, final_soc_shape (), (), (24, 1)
# final sch shape: (24, 13)
# final soc shape: (24, 1)
# final ch shape: (24, 1)
# final energy total shape: (24, 1)
# final energy avail total shape: (0, 1)

# %%
# results_table.get_column("LCOE")

# %%
# all_globals = globals()
# all_keys = [key for key in all_globals.keys()]
# all_keys = sorted(all_keys)
# all_keys

# %%
# my_table = run.use_artifact('csm_matthew_england/GA_Parameter_Tuning/run-u5gtzbiu-Evaluation_Results:v0', type='run_table').get('Evaluation_Results')

# %%
# everything returning from day_calculation and set to prev things, assign prev and cum as self things and then 


