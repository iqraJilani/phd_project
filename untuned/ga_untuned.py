import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import math
import time
import wandb
from datetime import datetime


now = datetime.now()
current_datetime =now.strftime("%d/%b/%Y_%H:%M:%S")
run_name = "_".join(["ga_untuned_baseline", current_datetime])



        
wandb.login(
         key="a9cbb8af7a708fc893fe0aaf2f50d9e51aed6f34"
         )
wandb.init(
            project = "GA_Parameter_Tuning",
            name = run_name,
            config ={
                "initial_prob_mutation": 0.3,
                "initial_prob_crossover": 0.9,
                "no_of_gen": 200,
                "population_szie": 400,
            }, 
            allow_val_change=True)
        
wandb.define_metric("gen_num")
wandb.define_metric("day_num")

wandb.define_metric("best_fv_gen", step_metric="gen_num")
wandb.define_metric("best_fv_day", step_metric="day_num")


iter_no = 2
con = 40

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
WEEK_TWO_SLOTS = 336#

best_fv_gen = 0



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

    l_so0c, tslfc, n = soc_factor(soc_stack, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[0:3]
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

# user-satisfaction
data = pd.read_excel('hall_declining_scenario_data.xlsx', 'Sheet3')  # energy usage data
data['Potential_PV_power_W'][data['Potential_PV_power_W'] < 0] = 0
#data = data.drop(['CPEHC', 'CPER1', 'CPESC'], 1)
data.drop(columns=['CPEHC', 'CPER1', 'CPESC'])
data_2 = copy.copy(data)
data.drop(index=data.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
data = data.reset_index(drop=True)

data_2 = data_2.filter(regex='CPE|LVL')  # extract CPE and socket data
data_2 = data_2.astype(float)
print(len(data_2))
abs_sat_total = np.empty((0, len(data_2.columns)))

data_2.values[data_2 > 0] = 1
#data_2[data_2>0] = 1

for b in range(14, int(len(data_2)/SLOT_DAY)):

    

    day = data_2.iloc[(b*SLOT_DAY):(b*SLOT_DAY+SLOT_DAY)]
    one_week = data_2.iloc[((b*SLOT_DAY)-WEEK_ONE_SLOTS):((b*SLOT_DAY)-WEEK_ONE_SLOTS)+SLOT_DAY]
    two_week = data_2.iloc[((b*SLOT_DAY)-WEEK_TWO_SLOTS):((b*SLOT_DAY)-WEEK_TWO_SLOTS)+SLOT_DAY]

    day = day.reset_index(drop=True)
    one_week = one_week.reset_index(drop=True)
    two_week = two_week.reset_index(drop=True)

    day_sat = (day + one_week + two_week)/3  # Find the average
    abs_sat_total = np.vstack((abs_sat_total, day_sat))


abs_sat_total = pd.DataFrame(abs_sat_total)
abs_sat_total.columns = data_2.columns

data_2.drop(index=data_2.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
data_2 = data_2.reset_index(drop=True)


soc = data.loc[0, 'Battery Monitor State of charge %']/100
least_soc = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
final_soc = np.full((SLOT_DAY, 1), soc)
time_delta = 0.00001  # time since last full charge
bad_charges = 0  # number of bad charges
final_schedule_cum = np.empty((0, len(day.columns)))
final_soc_cum = np.empty((0, 1))
final_ch_cum = np.empty((0, 1))
final_dch_cum = np.empty((0, 1))
final_energy_total_cum = np.empty((0, 1))
final_energy_avail_total = np.empty((0, 1))
day_counter = 0
best_fv_day = np.empty((0, 1))


for u in range(int((len(data))/SLOT_DAY)):
    print(f"DAY: {u}")
    df = data.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # picks the consumption data for day x
    forecast_demand = df.filter(regex='CPE|LVL')  # extract CPE and socket data
    forecast_demand = forecast_demand.reset_index(drop=True)

    energy_avail_gen = pd.DataFrame(data['Potential_PV_power_W'][u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY])  # Potential PV generation data of day x
    energy_avail_gen.columns = ['gen_cap']  # renaming the PV generation data column
    energy_avail_gen = energy_avail_gen.reset_index(drop=True)
    abs_sat = abs_sat_total.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # absolute satisfaction of day x
    abs_sat = abs_sat.reset_index(drop=True)
    demand_sch = data_2.iloc[u*SLOT_DAY:u*SLOT_DAY+SLOT_DAY, :]  # binary demand of day x

    demand_sch = demand_sch.reset_index(drop=True)
    demand_sch = demand_sch.astype(int)
    actual_demand = copy.copy(demand_sch)
    schedule = np.array(copy.copy(demand_sch))  # Copy of schedule
    soc_stack = final_soc
    mean_fitness_value = np.empty((0, 1))

    init_pop = np.empty((0, len(demand_sch.columns)))  # declaration of size of initial population
    sch_temp = len(demand_sch.columns)
    best_in_generations = np.empty((0, len(demand_sch.columns)))  # stack of best schedules in generations.
    best_fv = np.empty((0, 1))  # stack of best fitness values in each generation
    final_schedule_sorting = np.empty((0, len(demand_sch.columns)))
    best_soc_in_generations = np.empty((0, 1))  # stack of best soc in generations
    best_ch_in_generations = np.empty((0, 1))  # stack of best charges in generations
    best_dch_in_generations = np.empty((0, 1))  # stack of best discharges in generations
    best_et_in_generations = np.empty((0, 1))  # stack of best energy_total in every generation
    tslfc = np.empty((0, 1))  # hourly time since last full charge
    n = np.empty((0, 1))  # hourly number of bad charges
    l_soc = np.empty((0, 1))  # least soc since last full charge
    capacity_shortage = 0  # capacity shortage value
    energy_avail_total = np.array(forecast_demand.sum(axis=1)*(1-capacity_shortage))
    best_energy_avail_total_generations = np.empty((0, 1))
    battery_current = np.empty((0, 1))  # battery current
    soc_cummulative = np.empty((0, 1))  # stack of all soc for all chromosomes/ schedules in a generation
    start = time.time()
    # Definition of GA parameters
    p_c = 0.9
    p_m = 0.3
    k = 3
    pop = 400
    gen = 200

    # Generation of initial population
    ''' The algorithm generates it's initial population by picking a column and shuffle the chromosomes in that column'''
    for a in range(int(pop)):
        new_sch = demand_sch.apply(np.random.permutation, axis=1)
        new_sch = pd.DataFrame([list(x) for x in new_sch])
        init_pop = np.vstack((init_pop, new_sch))  # stack the populations together
        lsoc_stack_gen = np.empty((0, 1))
        tslfc_stack_gen = np.empty((0, 1))
        n_stack_gen = np.empty((0, 1))
    stop_crit_counter = 0
    init_fv = -300
    # generation iteration
    for b in range(int(gen)):

        print(f"gen no: {b}")
        
        fitness_values = np.empty((0, 1))  # declare array of fitness values
        new_population = np.empty((0, len(demand_sch.columns)))  # declare array of new population in new generation
        soc_cummulative = np.empty((0, 1))
        energy_available_total_cummulative = np.empty((0, 1))
        ch_gen = np.empty((0, 1))
        dch_gen = np.empty((0, 1))
        energy_total_gen = np.empty((0, 1))

        for c in range(int(pop/2)):
            
            parents = selection(init_pop, demand_sch, time_delta, bad_charges, least_soc)
            parent_1 = parents[0:len(demand_sch), :]
            parent_2 = parents[len(demand_sch):, :]
            child_1 = np.empty((0, len(demand_sch)))
            child_2 = np.empty((0, len(demand_sch)))
            child_1, child_2 = crossover(parent_1, parent_2)
            fitness_values, new_population, soc_cummulative, ch_gen, dch_gen, energy_total_gen = mutation(child_1, child_2, time_delta, least_soc, bad_charges, ch_gen, dch_gen, energy_total_gen, fitness_values, soc_cummulative, soc_stack, new_population)

        sorted_index = np.argsort(fitness_values[:, 0])[::-1]  # sort position of fitness values in descending order #give position
        sorted_fitness_value = np.sort(fitness_values[:, 0])[::-1]  # sort in descending order
        mean_fitness_value = np.vstack((mean_fitness_value, np.mean(sorted_fitness_value)))
        best_fv = np.vstack((best_fv, sorted_fitness_value[0]))  # stacks the best fitness value in every generation
        best = new_population[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]  # fetch the best schedule from the stack of schedules in generation.
        best_soc = soc_cummulative[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_charge_power = ch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_discharge_power = dch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_energy_total = energy_total_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
        best_in_generations = np.vstack((best_in_generations, best))  # stacks the best schedules in each generation together
        best_soc_in_generations = np.vstack((best_soc_in_generations, best_soc))
        best_ch_in_generations = np.vstack((best_ch_in_generations, best_charge_power))
        best_dch_in_generations = np.vstack((best_dch_in_generations, best_discharge_power))
        best_et_in_generations = np.vstack((best_et_in_generations, best_energy_total))
        init_pop = new_population
        soc_stack = final_soc

        best_fv_gen = float(max(fitness_values))
        wandb.log({"best_fv_gen": best_fv_gen, "gen_num": b}, commit=True)

        
        
        if (sorted_fitness_value[0] - init_fv) < 0.05:
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

    final_schedule_cum = np.vstack((final_schedule_cum, final_schedule))
    final_soc_cum = np.vstack((final_soc_cum, final_soc))
    final_ch_cum = np.vstack((final_ch_cum, final_ch))
    final_dch_cum = np.vstack((final_dch_cum, final_dch))
    final_energy_total_cum = np.vstack((final_energy_total_cum, final_energy_total))
    final_energy_avail_total = np.vstack((final_energy_avail_total, final_energy_avail_total))
    soc_stack = final_soc
    soc = final_soc[-1][0]
    day_counter = day_counter + 1

    least_soc, time_delta, bad_charges = soc_factor(soc_stack, final_soc, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[3:]
    end = time.time()
    max_day = float(max(best_fv))
    best_fv_day = np.vstack((best_fv_day, max_day))

    wandb.log({"best_fv_day": max_day, "day_num": u}, commit=True)
    
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
# a_s = ((abs_sat_total.iloc[0:day_counter*SLOT_DAY, :] * final_schedule_cum[:, :]*data_2).sum().sum())/(abs_sat_total.iloc[0:day_counter*SLOT_DAY, :]*data_2).sum().sum()  # calculation of daily absolute index
# p_r = y_f/y_r

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
# f_i_n = ((-43.9/first_discharge)**(1/2)) * ((np.exp(n_cum/3.6))**(1/3))
# f_soc = 1 + (C_SOC_0 + C_SOC_MIN*(1-l_soc_cum))*f_i_n*tslfc_cum
# weighted_throughput = f_soc * -final_dch_cum
# weighted_througput_cummulative = np.sum(weighted_throughput) * MONTH
# battery_lifetime = KWH_TP / -(weighted_througput_cummulative)

# cap_shortage = (e_demand_plot.sum() - y_f) * MONTH


# num = 0
# den = 0
# for c in range(int(battery_lifetime)):
#     num = num + (y_f*UNIT_COST*MONTH/(1+R)**c)

#     den = den + ((y_f*MONTH)/(1+R)**c)


# lcoe = (INITIAL_COST + num) / den

# fig, ax = plt.subplots()
# fig.subplots_adjust(right=1)
# plt.plot(list(range(1, b+2)), best_fv, marker='+', color='red', label='best fitness value')
# plt.legend()
# plt.savefig('best_fv_last_day.png')

# fig, ax = plt.subplots()
# fig.subplots_adjust(right=1)
# plt.plot(list(range(1, len(best_fv_day)+1)), best_fv_day, marker='+', color='red', label='best fitness value')
# plt.legend()
# plt.savefig('best_fv_each_day.png')

# duration = end - start

# results = pd.DataFrame([{
#             "run_name": run_name,
#             "a_s": a_s,
#             "p_r": p_r,
#             "lcoe": lcoe,
#             "best_fv": best_fv_day,
#             "duration": duration
#         }])
        
# results_table = wandb.Table(dataframe=results)
# wandb.log({"Results": results_table})
# print(results)