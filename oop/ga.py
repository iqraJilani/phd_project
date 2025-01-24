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


class GeneticAlgorithm:
    """
    Add doc string later on
    """
    
    def __init__(self, pop_size, num_generations,  prob_crossover, prob_mutation, obj_func_weights, wb_run):
        """
        Add doc strings later on
        """
        

        self.con= 40
        self.iter_no = 2

        self.early_stopped = False

        self.pop_size = pop_size
        self.num_gen = num_generations

        self.p_c = prob_crossover
        self.p_m = prob_mutation

        #self.task = task

        #print(obj_func_weights)
        self.w_1 = obj_func_weights[0]
        self.w_2 = obj_func_weights[1]
        self.w_3 = obj_func_weights[2]

        self.obj_soc_fac = 0
        self.obj_as = 0
        self.obj_pr = 0

        self.wb = wb_run

        # self.battery_state = battery_state
        
        # self.soc_stack = None
        # self.soc_all_day_prev = battery_state["soc_all_day_prev"]
        # #self.soc_last_hour = battery_state["soc_last_hour"]
        # self.least_soc_prev = battery_state["least_soc_prev"]
        # self.time_delta_prev = battery_state["time_delta_prev"]
        # self.bad_charges_prev = battery_state["bad_charges_prev"]

        self.batt_capacity = 10560
        self.batt_thresh_min = 0.4
        self.batt_thresh_max = 0.99
        self.batt_voltage = 48
        self.batt_cut_off = 0.4
        self.slot_day = 24
        self.week_one_slots = 168
        self.week_two_slots = 336
        self.month_slot = 24
        self.month = 12
        self.charge_eff = 0.99
        self.dis_eff = 0.7
        self.dis_eff_cost = 0.885
        self.initial_cost = 100000
        self.annual_cost = 100
        self.unit_cost = 0.0005
        self.lc = 500
        self.kwh_tp = self.lc * self.batt_capacity
        self.R = 0.07
        self.pen = 5000
        self.c_soc_0 = 6.614 * (10**(-5))
        self.c_soc_min= 3.307 * (10**(-3))
        self.c_batt_init = 10000
        

    def soc_factor(self, soc_stack, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges):
        for j in range(len(soc_stack) - len(soc_t), len(soc_stack)):

            # obtain least soc since last full charge
            if soc_stack[j, 0] < least_soc:
                least_soc = soc_stack[j, 0]
            elif soc_stack[j, 0] >= self.batt_thresh_max:
                least_soc = self.batt_thresh_max #BATT_THRESH_MAX
            else:
                least_soc = least_soc
            l_soc = np.vstack((l_soc, least_soc))

            # time since last full charge
            if soc_stack[j, 0] >= self.batt_thresh_max:
                time_delta = 0.00001
            else:
                time_delta = time_delta + 1
            tslfc = np.vstack((tslfc, time_delta))

        # no of bad charges
            if (soc_stack[j-1, 0] > 0.90) and (soc_stack[j-1, 0] < self.batt_thresh_max) and (soc_stack[j, 0] < soc_stack[j-1, 0]) and (soc_stack[j-1, 0] > soc_stack[j-2, 0]):
                bad_charges = (0.0025-((0.95-soc_stack[j-1, 0])**2))/0.0025
            elif soc_stack[j, 0] >= self.batt_thresh_max:
                bad_charges = 0
            else:
                bad_charges = bad_charges
            n = np.vstack((n, bad_charges))

        return(l_soc, tslfc, n, least_soc, time_delta, bad_charges)
    

    def soc_ch_dch(self, forecast_demand, energy_avail_gen, schedule, soc, least_soc):
        energy_total = np.empty((0, 1))  # hourly total energy (PV generation + available battery power)
        max_charge_power = 1000
        charge_power = np.empty((0, 1))  # hourly charge power of battery
        discharge_power = np.empty((0, 1))  # hourly discharge power of battery
        soc_t = np.empty((0, 1))
        energy_user_batt = 0
        for i in range(len(forecast_demand)):  # soc calculation model , soc, schedule

            battery_soc_to_full = self.batt_capacity - (self.batt_capacity *soc)  # energy required to fill up the battery

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

            energy_avail_total_single = energy_avail_gen['gen_cap'][i] + ((soc*self.batt_capacity ) - (self.batt_thresh_min*self.batt_capacity ))
            energy_total = np.vstack((energy_total, energy_avail_total_single))  # sum of avilable energy from battery and PV
            soc = (soc) + (ch/self.batt_capacity) - (energy_user_batt/self.batt_capacity )  # soc estimation model - coulomb counting
            soc_t = np.vstack((soc_t, soc))  # stack of hourly soc values per day
            
        return(soc_t, charge_power, discharge_power, energy_total)


    def obj_func(self, forecast_demand, schedule, energy_avail_gen, abs_sat, soc, soc_stack, least_soc, time_delta, bad_charges, final_soc):
        '''This function computes the performance ratio of the per schedule solution'''
        y_f = (forecast_demand*schedule).sum().sum()  # calculation of daily final yield (total energy consumption) for scheduling solution.
        y_r = energy_avail_gen.sum().sum()  # calculation of daily reference yield (potential energy)
        a_s = ((abs_sat * schedule).sum().sum())/(abs_sat).sum().sum()  # calculation of daily absolute index
        p_r = y_f/y_r  # performance ratio /capacity utilisation factor
        bad_charges = 0  # no. of bad charges
        tslfc = np.empty((0, 1))  # hourly 'time since last full charge'
        n = np.empty((0, 1))  # hourly 'number of bad charges'
        l_soc = np.empty((0, 1))  # hourly 'least charge since last full recharge'

        soc_t, charge_power, discharge_power, energy_total = self.soc_ch_dch(forecast_demand, energy_avail_gen, schedule, soc, least_soc)
        soc_stack = np.vstack((soc_stack, soc_t))

        l_soc, tslfc, n = self.soc_factor(soc_stack, soc_t, least_soc, l_soc, tslfc, n, time_delta, bad_charges)[0:3]
        s = np.sum((n/10.8) + np.log(tslfc) + np.log(1-l_soc))
        s = (s + (16.1181*self.slot_day))/((5.2156*self.slot_day)+(16.1181*self.slot_day))

        # globals()['obj_bl'] = s
        # globals()['obj_as'] = a_s
        # globals()['obj_pr'] = p_r
        self.obj_soc_fac = s
        self.obj_as = a_s
        self.obj_pr = p_r

        fit_val = (self.w_3*p_r) + (self.w_2*a_s) - (self.w_1*s)  # weighted objective function
        soc_stack = final_soc  # redeclaring the original soc
        #print("Inside:", soc_stack.shape)
        return(fit_val, soc_t, energy_total, charge_power, discharge_power, time_delta, least_soc, bad_charges)
    

    def penalty(self, array, forecast_demand, energy_total, soc_d):
        'This penalty function contains the hard constraints'
        add_penalties = []

        for i in range(len(array)):
            hourly_final_yield = (array[i, :] * forecast_demand.iloc[i, :]).sum()
            if (hourly_final_yield > energy_total[i]):
                pen = self.pen
                add_penalties = np.append(add_penalties, pen)

            if (soc_d[i] < self.batt_thresh_min):
                pen = self.pen
                add_penalties = np.append(add_penalties, pen)

        sum_add_penalties = sum(add_penalties)
        return sum_add_penalties
    

    def selection(self, init_pop, time_delta, bad_charges, least_soc, final_soc, demand_sch, forecast_demand, energy_avail_gen, abs_sat, soc_stack):
        '''Selection by Tournament selection'''
        winner_fitness = 0
        parents = np.empty((0, len(demand_sch.columns)))  # declares array for parents, two parents at a time
        for d in range(2):
            # random selection of 3 parents
            warrior_1_index = np.random.randint(0, self.pop_size)
            warrior_2_index = np.random.randint(0, self.pop_size)
            warrior_3_index = np.random.randint(0, self.pop_size)

            # This block ensures that the same parents are not selected more than once
            while warrior_1_index == warrior_2_index:
                warrior_1_index = np.random.randint(0, self.pop_size)

            while warrior_2_index == warrior_3_index:
                warrior_2_index = np.random.randint(0, self.pop_size)

            while warrior_3_index == warrior_1_index:
                warrior_3_index = np.random.randint(0, self.pop_size)

            # This block extracts the individual warriors from the initial population array.
            warrior_1 = init_pop[warrior_1_index*len(demand_sch):(warrior_1_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]
            warrior_2 = init_pop[warrior_2_index*len(demand_sch):(warrior_2_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]         
            warrior_3 = init_pop[warrior_3_index*len(demand_sch):(warrior_3_index*len(demand_sch)) + len(demand_sch), 0:len(demand_sch.columns)]

            # Evaluation of objective function of randomly selected parents
            fit_val, soc_d, energy_total = self.obj_func(forecast_demand, warrior_1, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
            warrior_1_fitness = fit_val - self.penalty(warrior_1, forecast_demand, energy_total, soc_d)

            fit_val, soc_d, energy_total = self.obj_func(forecast_demand, warrior_2, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
            warrior_2_fitness = fit_val - self.penalty(warrior_2, forecast_demand, energy_total, soc_d)

            fit_val, soc_d, energy_total = self.obj_func(forecast_demand, warrior_3, energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:3]
            warrior_3_fitness = fit_val - self.penalty(warrior_3, forecast_demand, energy_total, soc_d)

            # selecting the warriors with the highest fitness functions as parents
            if warrior_1_fitness == max(warrior_1_fitness, warrior_2_fitness, warrior_3_fitness):
                winner = warrior_1
                winner_fitness = warrior_1_fitness

            elif warrior_2_fitness == max(warrior_1_fitness, warrior_2_fitness, warrior_3_fitness):
                winner = warrior_2
                winner_fitness = warrior_2_fitness

            else:
                winner = warrior_3
                winner_fitness = warrior_3_fitness


            self.wb.log({"selected parent fitness": winner_fitness})

            parents = np.vstack((parents, winner))

        return(parents)
    

    def crossover(self, parent_1, parent_2, demand_sch):
    
        rand_co = np.random.rand()  # choose a number at random
        if rand_co < self.p_c:  # check if number selected is less than probability of crossover

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


    def mutation(self, child_1, child_2, time_delta, bad_charges, least_soc, final_soc, soc_stack, soc_cummulative, ch_gen, dch_gen, energy_total_gen, energy_avail_gen, forecast_demand, abs_sat, fitness_values, new_population):
        for i in range(1, 3):
            rand_mut = np.random.rand()
            if rand_mut < self.p_m:
                mut_row = np.random.randint(0, np.size(globals()['child_' + str(i)], 0))
                mut_col = np.random.randint(0, np.size(globals()['child_' + str(i)], 1))

                if globals()['child_' + str(i)][mut_row][mut_col] == 0:
                    globals()['child_' + str(i)][mut_row][mut_col] = 1

                else:
                    globals()['child_' + str(i)][mut_row][mut_col] = 0

            globals()['mut_child_' + str(i)] = globals()['child_' + str(i)]

            globals()['of_mut_child_' + str(i)], soc_d, energy_total, charge_power, discharge_power = self.obj_func(forecast_demand, globals()['mut_child_' + str(i)], energy_avail_gen, abs_sat, soc_stack[-1][0], soc_stack, least_soc, time_delta, bad_charges, final_soc)[0:5]
            ch_gen = np.vstack((ch_gen, charge_power))
            dch_gen = np.vstack((dch_gen, discharge_power))
            energy_total_gen = np.vstack((energy_total_gen, energy_total))
            globals()['pen_mut_child_' + str(i)] = self.penalty(globals()['mut_child_' + str(i)], forecast_demand, energy_total, soc_d)
            globals()['of_mut_child_' + str(i)] = globals()['of_mut_child_' + str(i)] - globals()['pen_mut_child_' + str(i)]
            fitness_values = np.vstack((fitness_values, globals()['of_mut_child_' + str(i)]))  # stack-up fitness values
            soc_cummulative = np.vstack((soc_cummulative, soc_d))  # stacks up the soc for each generation
            new_population = np.vstack((new_population, globals()['mut_child_' + str(i)]))
            soc_stack = final_soc

        return(fitness_values, new_population, soc_cummulative, ch_gen, dch_gen, energy_total_gen)
    

    def create_initial_pop(self, demand_sch, n_loads):
        ''' The algorithm generates it's initial population by picking a column and shuffle the chromosomes in that column'''
        init_pop = np.empty((0, n_loads))
        for a in range(int(self.pop_size)):
            new_sch = demand_sch.apply(np.random.permutation, axis=1)
            new_sch = pd.DataFrame([list(x) for x in new_sch])
            init_pop = np.vstack((init_pop, new_sch))  # stack the populations together
        #         lsoc_stack_gen = np.empty((0, 1))
        #         tslfc_stack_gen = np.empty((0, 1))
        #         n_stack_gen = np.empty((0, 1))
        return init_pop
    

    def calculate_fitness(self, time_delta, bad_charges, least_soc, final_soc,  soc_stack, demand_sch, forecast_demand, energy_avail_gen, abs_sat, n_loads):

        mean_fitness_value = np.empty((0, 1))

        # declaration of size of initial population
        sch_temp = len(demand_sch.columns)
        best_in_generations = np.empty((0, len(demand_sch.columns)))  # stack of best schedules in generations.
        best_fv = np.empty((0, 1))  # stack of best fitness values in each generation
        final_schedule_sorting = np.empty((0, len(demand_sch.columns)))
        best_soc_in_generations = np.empty((0, 1))  # stack of best soc in generations
        best_ch_in_generations = np.empty((0, 1))  # stack of best charges in generations
        best_dch_in_generations = np.empty((0, 1))  # stack of best discharges in generations
        best_et_in_generations = np.empty((0, 1))  # stack of best energy_total in every generation
        
    #     tslfc = np.empty((0, 1))  # hourly time since last full charge
    #     n = np.empty((0, 1))  # hourly number of bad charges
    #     l_soc = np.empty((0, 1))  # least soc since last full charge

        capacity_shortage = 0  # capacity shortage value
        energy_avail_total = np.array(forecast_demand.sum(axis=1)*(1-capacity_shortage))
        best_energy_avail_total_generations = np.empty((0, 1))
        battery_current = np.empty((0, 1))  # battery current
        soc_cummulative = np.empty((0, 1))  # stack of all soc for all chromosomes/ schedules in a generation
        start = time.time()
        # Definition of GA parameters
        k = 3
        #n_loads = len(demand_sch.columns)
        #init_pop = np.empty((0, len(demand_sch.columns)))


        # Generation of initial population
        init_pop = self.create_initial_pop(demand_sch, n_loads)


        stop_crit_counter = 0
        init_fv = -300
        # generation iteration
        for b in range(int(self.num_gen)):

            print(f"gen no: {b}")

            fitness_values = np.empty((0, 1))  # declare array of fitness values
            new_population = np.empty((0, len(demand_sch.columns)))  # declare array of new population in new generation
            soc_cummulative = np.empty((0, 1))
            energy_available_total_cummulative = np.empty((0, 1))
            ch_gen = np.empty((0, 1))
            dch_gen = np.empty((0, 1))
            energy_total_gen = np.empty((0, 1))

            for c in range(int(self.pop_size/2)):

                parents = self.selection(init_pop, time_delta, bad_charges, least_soc, final_soc, demand_sch, forecast_demand, energy_avail_gen, abs_sat, soc_stack)
                parent_1 = parents[0:len(demand_sch), :]
                parent_2 = parents[len(demand_sch):, :]
                child_1 = np.empty((0, len(demand_sch)))
                child_2 = np.empty((0, len(demand_sch)))
                child_1, child_2 = self.crossover(parent_1, parent_2,demand_sch)
                globals()['child_1'] = child_1
                globals()['child_2'] = child_2
                #print("Outside:", soc_stack.shape)
                fitness_values, new_population, soc_cummulative, ch_gen, dch_gen, energy_total_gen = self.mutation(child_1, child_2, time_delta, bad_charges, least_soc, final_soc, soc_stack, soc_cummulative, ch_gen, dch_gen, energy_total_gen, energy_avail_gen, forecast_demand, abs_sat, fitness_values, new_population)

            sorted_index = np.argsort(fitness_values[:, 0])[::-1]  # sort position of fitness values in descending order #give position
            sorted_fitness_value = np.sort(fitness_values[:, 0])[::-1]  # sort in descending order
            mean_fitness_value = np.vstack((mean_fitness_value, np.mean(sorted_fitness_value)))
            best = new_population[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]  # fetch the best schedule from the stack of schedules in generation.
            best_soc = soc_cummulative[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
            best_charge_power = ch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
            best_discharge_power = dch_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]
            best_energy_total = energy_total_gen[sorted_index[0]*len(demand_sch):sorted_index[0]*len(demand_sch) + len(demand_sch), :]


            best_fv = np.vstack((best_fv, sorted_fitness_value[0]))  # stacks the best fitness value in every generation
            best_in_generations = np.vstack((best_in_generations, best))  # stacks the best schedules in each generation together
            best_soc_in_generations = np.vstack((best_soc_in_generations, best_soc))
            best_ch_in_generations = np.vstack((best_ch_in_generations, best_charge_power))
            best_dch_in_generations = np.vstack((best_dch_in_generations, best_discharge_power))
            best_et_in_generations = np.vstack((best_et_in_generations, best_energy_total))
            
            init_pop = new_population
            soc_stack = final_soc

            best_fv_gen = float(max(fitness_values)[0])
            self.wb.log({"best_fv_gen": best_fv_gen, "gen_num": b}, commit=True)

            if sorted_fitness_value[0] - init_fv == 0:
                stop_crit_counter = stop_crit_counter + 1

            else:
                stop_crit_counter = 0

            if stop_crit_counter == self.con:
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