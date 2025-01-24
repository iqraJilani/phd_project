import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import wandb

from ga import GeneticAlgorithm


prob_crossover = 0.9
prob_mutation = 0.3
pop_size = 2
num_gen = 2
obj_func_weights = list([0.4, 0.4, 0.2])

batt_capacity = 10560
batt_thresh_min = 0.4
batt_thresh_max = 0.99
batt_voltage = 48
batt_cut_off = 0.4
slot_day = 24
week_one_slots = 168
week_two_slots = 336
month_slot = 24
month = 12
charge_eff = 0.99
dis_eff = 0.00139
dis_eff_cost = 0.885
initial_cost = 100000
annual_cost = 100
unit_cost = 0.0005
kwh_tp = 5544000
R = 0.07
pen = 5000
c_soc_0 = 6.614 * (10**(-5))
c_soc_min= 3.307 * (10**(-3))
c_batt_init = 10000


now = datetime.now()
run_name = "ga_untuned_functional"
current_datetime =now.strftime("%d/%b/%Y_%H:%M:%S")
run_name = "_".join([run_name, current_datetime])

early_stopped = np.zeros((44, 1), dtype=bool)


wandb.login(
    key="a9cbb8af7a708fc893fe0aaf2f50d9e51aed6f34"
)
wb = wandb.init(
    project = "GA_Parameter_Tuning",
    name = run_name,
    config ={
        "initial_prob_mutation": prob_mutation,
        "initial_prob_crossover": prob_crossover,
        "no_of_gen": num_gen,
        "population_szie": pop_size,
        "early_stopped_day": early_stopped
    }, 
    allow_val_change=True)

wb.define_metric("gen_num")
wb.define_metric("day_num")

wb.define_metric("best_fv_gen", step_metric="gen_num")
wb.define_metric("best_fv_day", step_metric="day_num")


def calculate_abs_sat_total(data):
    abs_sat_total = np.empty((0, len(data.columns)))
    for b in range(14, int(len(data)/slot_day)):
        print("create absolute user satisfaction")

        #Ij_to_do: rename one_week
        day = data.iloc[(b*slot_day):(b*slot_day+slot_day)] #data starting from 1st day of last two weeks e.g., day 14 so 14*24 =336 : 14*24+24= 360 ==> 336:360
        one_week = data.iloc[((b*slot_day)-week_one_slots):((b*slot_day)-week_one_slots)+slot_day]# Ij_to_do: ask daniel that this is not one week this is first day of week 1  # (14*24) - 168 : (14 *24)-168+ 24 ==> 336-168: 336-168+24 ==> 168: 192 ==> 
        two_week = data.iloc[((b*slot_day)-week_two_slots):((b*slot_day)-week_two_slots)+slot_day] # (14*24) - 336: (14*24)-366+24 ==> 0:24 

        day = day.reset_index(drop=True)
        one_week = one_week.reset_index(drop=True)
        two_week = two_week.reset_index(drop=True)

        
        day_sat = (day + one_week + two_week)/3  # Find the average
        abs_sat_total = np.vstack((abs_sat_total, day_sat)) # no of days x no of devices

    abs_sat_total = pd.DataFrame(abs_sat_total)
    abs_sat_total.columns = data.columns
    return abs_sat_total


print("data preprocessing")
#Ij_t0_do: sheet as param, var name for data
data = pd.read_excel('hall_declining_scenario_data.xlsx', 'Sheet3')  # energy usage data
data['Potential_PV_power_W'][data['Potential_PV_power_W'] < 0] = 0 #PV power cannot be in negative so transform data to map negative values to zero
#data = data.drop(['CPEHC', 'CPER1', 'CPESC'], 1)
data.drop(columns=['CPEHC', 'CPER1', 'CPESC']) #Ij_to_do: asked daniel why drop these columns, for now, because we do not need information from these switch boards or slots
binary_load_data = copy.copy(data)
data.drop(index=data.index[0:week_two_slots], axis=0, inplace=True) # Drop frist week's data (Ij_to_do: ask daniel why dropping first week's data)
data = data.reset_index(drop=True)
binary_load_data = binary_load_data.filter(regex='CPE|LVL')  # extract CPE and socket data
binary_load_data = binary_load_data.astype(float)
print(len(binary_load_data))
#binary_load_data.values[binary_load_data > 0] = 1
binary_load_data[binary_load_data>0] = 1 #transfrom data values into binary
abs_sat_total = calculate_abs_sat_total(binary_load_data)
binary_load_data.drop(index=binary_load_data.index[0:week_two_slots], axis=0, inplace=True) #ij_to_do: ask daniel why dropping first two week's data from data copy used for user satisfaction
binary_load_data = binary_load_data.reset_index(drop=True)


#**** set data dependent parameters *** 

soc_last_hour = data.loc[0, 'Battery Monitor State of charge %']/100 #latest state of charge
least_soc_prev = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
time_delta_prev = 0.00001  # time since last full charge
bad_charges_prev = 0  # number of bad charges
soc_prev_day = np.full((slot_day, 1), soc_last_hour)

batt_state = {
    "soc_last_hour": soc_last_hour,
    "soc_prev_day": soc_prev_day,
    "least_soc_prev": least_soc_prev,
    "time_delta_prev": time_delta_prev,
    "bad_charges_prev": bad_charges_prev
}
#add missing ones

final_schedule_cum = np.empty((0, len(binary_load_data.columns)))
final_soc_cum = np.empty((0, 1))
final_ch_cum = np.empty((0, 1))
final_dch_cum = np.empty((0, 1))
final_energy_total_cum = np.empty((0, 1))
final_energy_avail_total = np.empty((0, 1))
day_counter = 0
best_fv_day = np.empty((0, 1))


for u in range(int((len(data))/slot_day)): #loop for 0 to number of days-1 (0 to 43)
    
    print(f"Day no: {u}")

    ga = GeneticAlgorithm(pop_size, num_gen, prob_crossover, prob_mutation, obj_func_weights, batt_state)
    
    df = data.iloc[u*slot_day:u*slot_day+slot_day, :]  # picks the consumption data for day x e.g. u=0, (0*24: 0*24+24) ==> (0:24) day data
    forecast_demand = df.filter(regex='CPE|LVL')  # extract CPE and socket data
    forecast_demand = forecast_demand.reset_index(drop=True) #forecast demand is current day's CPE and socket data

    energy_avail_gen = pd.DataFrame(data['Potential_PV_power_W'][u*slot_day:u*slot_day+slot_day])  # Potential PV generation data of day x
    energy_avail_gen.columns = ['gen_cap']  # renaming the PV generation data column
    energy_avail_gen = energy_avail_gen.reset_index(drop=True)
    abs_sat = abs_sat_total.iloc[u*slot_day:u*slot_day+slot_day, :]  # absolute satisfaction of day x
    abs_sat = abs_sat.reset_index(drop=True)
    demand_sch = binary_load_data.iloc[u*slot_day:u*slot_day+slot_day, :]  #binary demand of day x, 

    # current day's socket and CPE consumption data is forecast demand and schedule , 
    # raw data contains the amount of power consumed from each socket and light and 
    # later that is converted into binary (0, 1) that is basically acting as demand schedule

    soc_stack = soc_prev_day #stack of hourly soc
    demand_sch = demand_sch.reset_index(drop=True)
    demand_sch = demand_sch.astype(int)
    n_loads = len(demand_sch.columns)
    actual_demand = copy.copy(demand_sch)
    schedule = np.array(copy.copy(demand_sch))  # Copy of schedule
    tslfc = np.empty((0, 1))  # hourly time since last full charge
    n = np.empty((0, 1))  # hourly number of bad charges
    l_soc = np.empty((0, 1))  # least soc since last full charge
    capacity_shortage = 0  # capacity shortage value
    energy_avail_total = np.array(forecast_demand.sum(axis=1)*(1-capacity_shortage))
    best_energy_avail_total_generations = np.empty((0, 1))
    energy_available_total_cummulative = np.empty((0, 1))
    


    best_fv_all_gen, duration, best_in_generations, best_soc_in_generations, best_ch_in_generations, best_dch_in_generations, best_et_in_generations = ga.calculate_fitness(wb,  n_loads, forecast_demand, demand_sch, abs_sat, energy_avail_gen, soc_stack)

    print("sorting and finding best one for the day")
    best_fv_day = float(max(best_fv_all_gen))
    best_fv_all_days = np.vstack((best_fv_all_days, best_fv_day))

    wb.log({"best_fv_day": best_fv_day, "day_num": u}, commit=True)


    sorted_schedule_index = np.argsort(best_fv_all_gen[:, 0])[::-1]  # sorts the fitness values of the generational best values
    final_schedule = best_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    soc_prev_day = best_soc_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_ch = best_ch_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_dch = best_dch_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_energy_total = best_et_in_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]
    final_energy_avail_total = best_energy_avail_total_generations[sorted_schedule_index[0]*len(demand_sch):sorted_schedule_index[0]*len(demand_sch) + len(demand_sch), :]

    final_schedule_cum = np.vstack((final_schedule_cum, final_schedule))
    final_soc_cum = np.vstack((final_soc_cum, soc_prev_day))
    final_ch_cum = np.vstack((final_ch_cum, final_ch))
    final_dch_cum = np.vstack((final_dch_cum, final_dch))
    final_energy_total_cum = np.vstack((final_energy_total_cum, final_energy_total))
    final_energy_avail_total = np.vstack((final_energy_avail_total, final_energy_avail_total))
    soc_stack = soc_prev_day
    soc_last_hour = soc_prev_day[-1][0]
    day_counter = day_counter + 1

    least_soc_prev, time_delta_prev, bad_charges_prev = ga.soc_factor(soc_prev_day, least_soc_prev, l_soc, tslfc, n, time_delta_prev, bad_charges_prev, soc_stack)[3:]
    end = time.time()
    #break #Ij_to_do why breaking days loop
        

# print("something about load")
# e_load = data.filter(regex='CPE|LVL').iloc[0:day_counter*slot_day, :]*final_schedule_cum[:, :]
# e_demand = data.filter(regex='CPE|LVL').iloc[0:day_counter*slot_day, :]
# e_demand_plot = np.sum(e_demand, axis=1)
# e_load_plot = np.sum(e_load, axis=1)

# pv_load = np.empty((0, 1))

# for i in range(len(final_schedule_cum)):
#     #Ij_to_do: ask daniel why is this one the other way around
#     if data['Potential_PV_power_W'][i] > 0:
#         if (data['Potential_PV_power_W'][i] - final_ch_cum[i]) >= e_load_plot[i]: #if the net energy available (from PV and in battery) is greater than the scheduled load, then schedule it to PV power
#             pv_load_t = e_load_plot[i]

#         elif (data['Potential_PV_power_W'][i] - final_ch_cum[i]) < e_load_plot[i]: #else schedule it to net power remaining  
#             pv_load_t = data['Potential_PV_power_W'][i] - final_ch_cum[i]

#     else:
#         pv_load_t = 0

#     pv_load = np.vstack((pv_load, pv_load_t))


# print("performance raio")
# y_f = (data.filter(regex='CPE|LVL').iloc[0:day_counter*slot_day, :]*final_schedule_cum[:, :]).sum().sum() #used energy # [:,:-1] calculation of daily final yield for scheduling solution.
# y_r = data['Potential_PV_power_W'].sum().sum()  # calculation of daily reference yield #available energy (solocast)
# a_s = ((abs_sat_total.iloc[0:day_counter*slot_day, :] * final_schedule_cum[:, :]*binary_load_data).sum().sum())/(abs_sat_total.iloc[0:day_counter*slot_day, :]*binary_load_data).sum().sum()  # calculation of daily absolute index
# p_r = y_f/y_r

# # Calculating the hourly battery current
# battery_current = np.empty((0, 1))
# for p in range(len(final_dch_cum)):
#     if final_dch_cum[p] > 0 and final_ch_cum[p] == 0: #when more current is drawn out of battery then it is charged for, the battery current is in negative
#         i_bat = -final_dch_cum[p]/batt_voltage

#     elif final_ch_cum[p] > 0 and final_dch_cum[p] == 0: #when battery is not being discharged/used, and has some charge left in it, then final battery current is given according to formula
#         i_bat = (final_ch_cum[p]/batt_voltage)*charge_eff 

#     elif final_ch_cum[p] == 0 and final_dch_cum[p] == 0: #battery is not used and it's not charged either
#         i_bat = 0

#     battery_current = np.vstack((battery_current, i_bat))

# tslfc_cum = np.empty((0, 1))  # hourly time since last full charge
# n_cum = np.empty((0, 1))  # hourly number of bad charges
# l_soc_cum = np.empty((0, 1))

# first_discharge = np.empty((0, 1))

# #f_d: first discharge current after first full charge
# if battery_current[0, 0] < 0: #when more drained then charged
#     f_d = battery_current[0, 0]

# else:
#     f_d = -0.0001

# least_soc = data.loc[0, 'Battery Monitor State of charge %']/100  # least soc since last full charge
# time_delta = 0.00001  # time since last full charge
# bad_charges = 0  # number of bad charges

# soc_t = np.empty((month_slot, 1))
# l_soc_cum, tslfc_cum, n_cum = ga.soc_factor(final_soc_cum, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[0:3]

# # Calculating first_discharge current after full charge
# for j in range(1, len(final_soc_cum)):
#     if (final_soc_cum[j, 0] < batt_thresh_max) and (final_soc_cum[j-1, 0] >= batt_thresh_max):
#         f_d = battery_current[j, 0]
#     else:
#         f_d = f_d

#     first_discharge = np.vstack((first_discharge, f_d))

# if first_discharge[0, 0] == -0.0001:
#     first_discharge = np.vstack((-0.0001, first_discharge))
# else:
#     first_discharge = np.vstack((first_discharge[0, 0], first_discharge))

# print("calculating weighted throughput and battery lifetime")

# #refer to objective function 1 formula
# f_i_n = ((-43.9/first_discharge)**(1/2)) * ((np.exp(n_cum/3.6))**(1/3))
# f_soc = 1 + (c_soc_0 + c_soc_min*(1-l_soc_cum))*f_i_n*tslfc_cum
# weighted_throughput = f_soc * -final_dch_cum
# weighted_througput_cummulative = np.sum(weighted_throughput) * month
# battery_lifetime = kwh_tp / -(weighted_througput_cummulative)

# cap_shortage = (e_demand_plot.sum() - y_f) * month


# #Ij_to_do: ask daniel what is den and number here , numerator and denominator
# num = 0
# den = 0
# for c in range(int(battery_lifetime)):
#     num = num + (y_f*unit_cost*month/(1+R)**c)

#     den = den + ((y_f*month)/(1+R)**c)


# lcoe = (initial_cost + num) / den

# fig, ax = plt.subplots()
# fig.subplots_adjust(right=1)
# plt.plot(list(range(1, b+2)), best_fv_all_gen, marker='+', color='red', label='best fitness value')
# plt.title(f"Day {day_counter} fitness value")
# plt.legend()
# plt.savefig(f"day_{day_counter}.png")

# #The existing CSV file
# #file_source = "ga_results.xlsx"

# # #Read the existing CSV file
# # df = pd.read_excel('ga_results_2.xlsx','Sheet1')

# # #Insert"Name"into cell A1 to replace "Customer"

# #iter no is basically for row counter for storing results
# # print(f"duration was: {duration}")
# # df.iloc[iter_no-1,1] = float(max(best_fv))
# # df.iloc[iter_no-1,2] = duration

# # #Save the file
# # df.to_excel('ga_results_2.xlsx', 'Sheet1', index=False)

# #Ij_to_do: ask daniel why two iteration counter, one is con and other one is iter no, 
# #best_fv = best_fv[0][0]
best_fv_all_days_sorted = np.sort(best_fv_day[:, 0])[::-1]
best_fv_all_days = best_fv_all_days_sorted[0]
best_fv_last_day = best_fv_day[-1, :]

results = pd.DataFrame([{
    "run_name": run_name,
    "a_s": a_s,
    "p_r": p_r,
    "lcoe": lcoe,
    "best_fv": best_fv_all_days,
    "duration": duration
}])

results_table = wandb.Table(dataframe=results)
wandb.log({"Results": results_table})
