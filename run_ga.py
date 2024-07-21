# %%
import argparse
import copy
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import random
import time
import wandb


from ga import GeneticAlgorithm


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
    

def day_calculation(wb, u, data, demand_data, abs_sat_total, final_soc, soc_stack, least_soc, soc, time_delta, bad_charges, n_loads):
    
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
    #create a ga object for each day
    ga_obj = GeneticAlgorithm(pop_size, num_gen, p_c, p_m, obj_func_weights, wb)

    best_fv, final_schedule, final_soc, final_ch, final_dch, final_energy_total, final_energy_avail_total= ga_obj.calculate_fitness(time_delta, bad_charges, least_soc, final_soc,  soc_stack, demand_sch, forecast_demand, energy_avail_gen, abs_sat, n_loads)

    soc_stack = final_soc
    soc = final_soc[-1][0]
#     day_counter = day_counter + 1

    

    least_soc, time_delta, bad_charges = ga_obj.soc_factor(soc_stack, final_soc, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[3:]
    
    end = time.time()
    duration_day = end - start
    duration_day = float(duration_day / 60)

    wb.log({"duration_day": duration_day, "day_num": u}, commit=True)
    
    best_fv_day = float(max(best_fv)[0])
    wb.log({"best_fv_day": best_fv_day, "day_num": u}, commit=True)
    
    wb.log({"obj_func1_bl": ga_obj.obj_soc_fac, "day_num": u}, commit=True)
    wb.log({"obj_func2_as": ga_obj.obj_as, "day_num": u}, commit=True)
    wb.log({"obj_func3_pr": ga_obj.obj_pr, "day_num": u}, commit=True)

    globals()['soc_prev'] = soc
    #print("global soc after: ", globals()['soc_prev'])
    globals()['final_soc_prev'] = final_soc
    globals()['soc_stack_prev'] = globals()['final_soc_prev']
    #soc_stack_prev = soc_stack
    globals()['least_soc_prev'] = least_soc
    globals()['time_delta_prev'] = time_delta
    globals()['bad_charges_prev'] = bad_charges


    #globals()['best_fv_days'] = np.vstack((globals()['best_fv_days'], best_fv_day))
    globals()['best_fv_all_days'][u] = best_fv_day
    #globals()['duration'] = np.vstack((globals()['duration'], duration_day))
    globals()['duration'][u] = duration_day
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
    #globals()['final_energy_total_cum'] [(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_energy_total
    globals()['final_energy_avail_total'] = np.vstack((globals()['final_energy_avail_total'], final_energy_avail_total))
    #globals()['final_energy_avail_total'][(u*SLOT_DAY): ((u*SLOT_DAY)+SLOT_DAY)] = final_energy_avail_total
    
    

    
    return duration_day, best_fv_day


arg_parser = argparse.ArgumentParser("simple_example")
arg_parser.add_argument('--dataset_name')
arg_parser.add_argument('--run_name')
arg_parser.add_argument('--prob_crossover', type=float)
arg_parser.add_argument('--prob_mutation', type=float)
arg_parser.add_argument('--w_bl', type=float)
arg_parser.add_argument('--w_pr', type=float)
arg_parser.add_argument('--w_as', type=float)


#arg_parser.add_argument('--weights', type=list, nargs="+", action="append")
#arg_parser.add_argument('--dataset_path')

task = "training"
#dataset_name = "0cenario_2_training"
args = arg_parser.parse_args()
dataset_name = args.dataset_name
dataset_path = "/nvme/jilanii/parameter_tuning/data/data.xlsx"

# 0.6
# 0.4


final_pm = 0
final_pc = 0
obj_bl = 0
obj_as = 0
obj_pr = 0
pop_size = 400
num_gen = 300 
p_c = float(args.prob_crossover)
print(f"type p_c: {type(p_c)}")
p_m = float(args.prob_mutation )

w_1, w_2, w_3 = float(args.w_bl), float(args.w_pr), float(args.w_as)
obj_func_weights = [w_1, w_2, w_3]

# W_1 = 0.33  # weight of objective 1
# W_2 = 0.33 # weight of objective 2
# W_3 = 0.33  # weight of objective 3
# print(f"obj_weights: ", args.weights)
# obj_func_weights = args.weights
#obj_func_weights = [0.5, 0.25, 0.25]

CH_EFF = 0.99  # battery charge efficiency
BATT_CAP = 10560  # battery capacity
V_BATT = 48  # battery voltage
C_SOC_0 = 6.614 * (10**(-5))
C_SOC_MIN = 3.307 * (10**(-3))
C_BATT_INIT = 10000
LC = 500
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
KWH_TP = LC * BATT_CAP
MONTH = 12
WEEK_ONE_SLOTS = 168
WEEK_TWO_SLOTS = 336
dis_eff = 0.7
C_ten = 880
I_ten = C_ten/10


now = datetime.now()
run_name = args.run_name
current_datetime =now.strftime("%d/%b/%Y_%H:%M:%S")
run_name = "_".join([run_name, current_datetime])

wandb.login(key="a9cbb8af7a708fc893fe0aaf2f50d9e51aed6f34")
wb = wandb.init(project = "GA_Parameter_Tuning",name = run_name, 
                config ={
                        "initial_prob_mutation": p_m,
                        "initial_prob_crossover": p_c,
                        "final_prob_mutation": final_pm,
                        "final_prob_crossover": final_pc,
                        "no_of_gen": num_gen,
                        "population_szie": pop_size,
                        "best_fv_average": 0,
                        "battery_lifetime": 0,
                        "capacity_shortage": 0, 
                        "obj_func_weights": obj_func_weights
                    }, 
                allow_val_change=True)
wb.define_metric("gen_num")
wb.define_metric("day_num")
wb.define_metric("best_fv_gen", step_metric="gen_num")
wb.define_metric("best_fv_day", step_metric="day_num")


data = pd.read_excel(dataset_path, dataset_name) # energy usage data
data['Potential_PV_power_W'][data['Potential_PV_power_W'] < 0] = 0
#data = data.drop(['CPEHC', 'CPER1', 'CPESC'], axis = 1)

demand_data = copy.copy(data)

data.drop(index=data.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
data = data.reset_index(drop=True)

demand_data = demand_data.filter(regex='CPE|LVL')  # extract CPE and socket data
demand_data = demand_data.astype(float)
demand_data.values[demand_data > 0] = 1
n_days = int(len(data)/SLOT_DAY)
n_loads = len(demand_data.columns)

abs_sat_total = calculate_user_satisfaction(demand_data, n_days, n_loads)

demand_data.drop(index=demand_data.index[0:WEEK_TWO_SLOTS], axis=0, inplace=True)
demand_data = demand_data.reset_index(drop=True)

best_fv_all_days = np.empty((n_days, 1))
final_schedule_cum = np.empty(((n_days * SLOT_DAY), len(demand_data.columns)))
final_soc_cum = np.empty(((n_days * SLOT_DAY), 1))
soc_prev_cum = np.empty(((n_days * SLOT_DAY), 1))
final_ch_cum = np.empty(((n_days * SLOT_DAY), 1))
final_dch_cum = np.empty(((n_days * SLOT_DAY), 1))
final_energy_total_cum = np.empty((0, 1))
final_energy_avail_total = np.empty((0, 1))
duration = np.empty((n_days, 1))

soc_prev = data.loc[0, 'Battery Monitor State of charge %']/100
least_soc_prev = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
final_soc_prev = np.full((SLOT_DAY, 1), soc_prev)
soc_stack_prev = final_soc_prev
time_delta_prev = 0.00001  # time since last full charge
bad_charges_prev = 0  # number of bad charges
cap_shortage_prev = 0

day_counter = 0
skip_list = [0, 1, 2, 3, 4, 29, 28, 27, 26, 25]

for u in range(int((len(data))/SLOT_DAY)):
    # if u in [0, 1, 2, 3, 4, 29, 28, 27, 26, 25]:
    #     continue
    print(f"DAY no: {u}")
    dur_prev, best_fv_day = day_calculation(wb, u, data, demand_data, abs_sat_total, final_soc_prev, soc_stack_prev, least_soc_prev, soc_prev, time_delta_prev, bad_charges_prev, n_loads)
    day_counter = globals()['day_counter'] +1 

n_days_new = n_days - len(skip_list)
best_fv_avg = (np.sum(best_fv_all_days))/20
print("best fun average: ", best_fv_avg)

wb.config.update({"best_fv_average": best_fv_avg}, allow_val_change=True)


pickle_name = run_name.replace('/', '_').replace(':', '_')
fv_pickle_name = pickle_name +"_fv"+".pickle"
dur_pickle_name = pickle_name +"_dur"+".pickle"
with open(f'/nvme/jilanii/parameter_tuning/output_pickle/{fv_pickle_name}', 'wb') as f:
    pickle.dump(best_fv_all_days, f)

with open(f'/nvme/jilanii/parameter_tuning/output_pickle/{dur_pickle_name}', 'wb') as f:
    pickle.dump(duration, f)

# batt_state_prev = {
#     'dataset_name': dataset_name,
#     'soc':soc_prev,
#     'final_soc': final_soc_prev, 
#     'soc_stack': final_soc_prev,
#     'least_soc': least_soc_prev,
#     'time_delta': time_delta_prev,
#     'bad_charges': bad_charges_prev

#     }


# with open('battery_state_prev.jsob', 'w', encoding='utf-8') as f:
#      json.dump(batt_state_prev, f, ensure_ascii=False, indent=4)


battery_current = np.empty((0, 1))
for p in range(len(final_dch_cum)):
    if final_dch_cum[p] > 0 and final_ch_cum[p] == 0:
        i_bat = -final_dch_cum[p]/V_BATT

    elif final_ch_cum[p] > 0 and final_dch_cum[p] == 0:
        i_bat = (final_ch_cum[p]/V_BATT)*CH_EFF

    elif final_ch_cum[p] == 0 and final_dch_cum[p] == 0:
        i_bat = 0

    battery_current = np.vstack((battery_current, i_bat))

#first discharge
first_discharge = np.empty((0, 1))

if battery_current[0,0] < 0:
    f_d = battery_current[0,0]
    
else:
    f_d =-0.0001



n_cum = np.empty((0, 1))
tslfc_cum = np.empty((0, 1))
l_soc_cum = np.empty((0, 1))

least_soc = data.loc[0, 'Battery Monitor State of charge %']/100 #least soc since last full charge
time_delta = 0.00001  # time since last full charge
bad_charges = 0  # number of bad chargesexit


for j in range(len(final_soc_cum)):
# obtain least soc since last full charge 
    if final_soc_cum[j, 0] < least_soc:
        least_soc = final_soc_cum[j, 0]
    elif final_soc_cum[j, 0] >= 0.99:
        least_soc = 0.99            
    l_soc_cum = np.vstack((l_soc_cum, least_soc))
# time since last full charge            
    if final_soc_cum[j, 0] >= 0.99:
        time_delta = 0.00001        
    else:
        time_delta = time_delta + 1
    tslfc_cum = np.vstack((tslfc_cum, time_delta))
                 
# no of bad charges
    if (final_soc_cum[j-1, 0] > 0.90) and (final_soc_cum[j-1, 0] < 0.99) and (final_soc_cum[j, 0] < final_soc_cum[j-1, 0]) and (final_soc_cum[j-1, 0] > final_soc_cum[j-2, 0]): 
        bad_charges = (0.0025-((0.95-final_soc_cum[j, 0])**2))/0.0025
    elif final_soc_cum[j, 0] >= 0.99:
        bad_charges = 0
    else:
        bad_charges = bad_charges
    n_cum = np.vstack((n_cum, bad_charges))
    
for j in range(1, len(final_soc_cum)):
    if (final_soc_cum[j,0] < 0.99) and (final_soc_cum[j-1,0] >= 0.99):
        f_d = battery_current[j, 0]        
    else:
        f_d = f_d
        
    first_discharge = np.vstack((first_discharge, f_d))
    
if first_discharge[0,0] == -0.0001:
    first_discharge = np.vstack((-0.0001, first_discharge))
else:
    first_discharge = np.vstack((first_discharge[0,0], first_discharge))


hourly_battery_data = np.column_stack((final_soc_cum, final_ch_cum, final_dch_cum, tslfc_cum, l_soc_cum, n_cum))


f_i_n = ((-I_ten/first_discharge)**(1/2)) * ((np.exp(n_cum/3.6))**(1/3))
f_soc = 1 + (C_SOC_0 + C_SOC_MIN*(1-l_soc_cum))*f_i_n*tslfc_cum
weighted_throughput = f_soc * -final_dch_cum
weighted_througput_cummulative = np.sum(weighted_throughput) * (1+(1-dis_eff)) * 12
battery_lifetime = KWH_TP/ -(weighted_througput_cummulative)
wb.config.update({"battery_lifetime": battery_lifetime}, allow_val_change=True)

c_shortage = 1 - ((final_schedule_cum*data.filter(regex='CPE|LVL')).sum().sum()/(data.filter(regex='CPE|LVL')*demand_data).sum().sum())
wb.config.update({"capacity_shortage": c_shortage}, allow_val_change=True)


# old_run = wandb.init(project = "GA_Parameter_Tuning", resume = 'c370to8o')
# old_run.define_metric("hour")
# old_run.define_metric("soc_each_hour", step_metric="hour")
# # %%
# pickle_name = run_name.replace('/', '_').replace(':', '_')
# fv_pickle_name = pickle_name +"_fv"+".pickle"
# dur_pickle_name = pickle_name +"_dur"+".pickle"
# with open(f'/nvme/jilanii/parameter_tuning/output_pickle/{fv_pickle_name}', 'wb') as f:
#     pickle.dump(best_fv_all_days, f)