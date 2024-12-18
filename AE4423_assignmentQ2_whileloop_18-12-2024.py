# %% [markdown]
# Connected to base (Python 3.11.5)

# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:20:54 2024

@author: suzev
"""
# Import necessary libraries
import gurobipy as gp  # Gurobi optimization library
from gurobipy import *  # Import all Gurobi functions
from gurobipy import GRB  # Import Gurobi constants
import numpy as np  # For numerical operations
import math  # For mathematical functions
import copy  # For copying objects
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.linear_model import LinearRegression

#name model
m2 = Model('AirlinePlanningAssingment1_Q2')

# #import data Suze
# pop_data_path = "C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment Q2\Group_1.xlsx"
# demand_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\DemandGroup1.xlsx'

# Path to the excel file
excel_file = r"C:\Users\Marjan\Documents\Study\AE44232o Airline planning and optimisation\AE4423Data2\Group_1.xlsx"
sheet_name_flights = "Flights"
sheet_name_itineraries = "Itineraries"
sheet_name_recapture = "Recapture"

# Read tabs of excel
df_flights = pd.read_excel(excel_file, sheet_name=sheet_name_flights)
df_itineraries = pd.read_excel(excel_file, sheet_name=sheet_name_itineraries)
df_recapture = pd.read_excel(excel_file, sheet_name=sheet_name_recapture)

# Print example of sheets
print(df_flights)
print(df_itineraries)
print(df_recapture)

# Sets
L = set(df_flights['Flight No.']) # Set of flights
P = set(df_itineraries[['Flight 1', 'Flight 2']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths)
P_p = set(df_recapture[['From Itinerary', 'To Itinerary']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths) with recapture from itinerary p

print("Set of Flights:", L)
print("Set of Passenger Itineraries (Paths):", P)
print("Set of passenger itineraries (paths) with recapture from itinerary p", P_p)

# Parameters
flight = df_flights['Flight No.'].tolist()
average_fares = df_itineraries.groupby(['Origin', 'Destination'])['Price [EUR]'].mean()
averagefare = average_fares.to_dict() # Average fare for itinerary p
fare = df_itineraries['Price [EUR]']
Dp = df_itineraries['Demand'].tolist() # Daily unconstrained demand for itinerary p
CAPi = df_flights['Capacity'] # Capacity on flight (leg) i
b = df_recapture['Recapture Rate'] # Recapture rate of a pax that desires itinerary p and is allocated to r
itinerary_flights = df_itineraries['Itinerary']


#%% Computing needed parameters

# Computing binary variable delta[i, p] for checking if leg i is in path p
L_list = df_flights['Flight No.'].tolist() 
flight1 = df_itineraries['Flight 1'].tolist()
flight2 = df_itineraries['Flight 2'].tolist()
delta_matrix = np.zeros((len(L_list), len(flight1)))

# Change sets to ranges
P = range(len(flight1))
L = range(len(L_list))

# Making delta[i, p] matrix
for i in range(len(L_list)):
    for j in range(len(flight1)):
        if flight1[j] == L_list[i]:
            delta_matrix[i][j] = 1
        if flight2[j] == L_list[i]:
            delta_matrix[i][j] = 1

for row in delta_matrix:
    print(row)

#Computing unconstrained demand Q[i]
demand_i = np.zeros(len(L))
for i in range(len(L)):  
    demand_i[i] = sum(Dp[p] * delta_matrix[i,p] for p in range(len(P)))  # Summing demand over all paths

# Computing the recapture rate b_pr
recapture_p_list = df_recapture['From Itinerary']
recapture_r_list = df_recapture['To Itinerary']
rate = df_recapture['Recapture Rate']
bpr = {(p,r): rate for p, r, rate in zip(recapture_p_list, recapture_r_list, rate)}

# Filling nan values of flight2 with 0 
flight2_series = pd.Series(flight2)
flight2_filled = flight2_series.fillna(0)
print(flight2_filled.tolist())  

# Itinerary dictionary for matching flights with itineraries
itinerary_arc_good = {}
for p in range(len(flight1)):
    for i in range(len(L_list)):
        if flight1[p] == flight[i]:
            itinerary_arc_good[p] = i

p_i_dict = {}
for key, value in itinerary_arc_good.items():
    for i in range(len(L_list)):
        if flight2_filled[key] == 0:
            p_i_dict[key] = [value]
        if flight2_filled[key] != 0:
            if flight2_filled[key] == flight[i]:
                p_i_dict[key] = [value, i]


#%% Checking paths with at least one flight exceeding capacity
for p in P:  # Assuming 'P' is the list of all paths
    for i in L:  # 'L' is the list of legs
        if delta_matrix[i, p] == 1:  # Check if leg i is part of path p
            if demand_i[i] > CAPi[i]:  # Check if demand exceeds capacity for leg i
                print(f"Path {p} has a capacity problem at leg {i}")
                break  # No need to check further legs for this path


#%% Initial Restricted Master problem 

# Add ficitious itinirary in P and in parameters: fare, bpr, Demand and delta
P_i = list(range(len(P) + 1)) # Make P_i : set of itinerary plus fictitious itinerary (p = 422)
fare_i = pd.concat([fare, pd.Series([0])], ignore_index=True) # Add fare = 0 for fictitious itinerary
for p in P_i:
    for r in P_i:
        if (p, r) not in bpr:
            bpr[(p, r)] = 0 # adds recapture rate of 0 for all non existing reallocations
    bpr[(p, 422)] = 1 # adds recapture rate of 1 for all reallocation to the ficitious itinerary
    bpr[(p, p)] = 1 # adds recapture rate of 1 for all reallocations to itself
    bpr[(422, 422)] = 0 # adds a recpature rate of 0 for reallocations from fictitious to fictitious
Dp = Dp[:len(P)]  # Trim to match existing itineraries
Dp.append(1000) # Adds a high limit of demand for the fictitious itinerary 
delta_matrix = np.append(delta_matrix,np.zeros([len(delta_matrix),1]),1) # delta is zero for each flight leg in the ficitious itinerary
p_i_dict[422] = [] # Add empty list of legs for the fictitious itinerary 

# Set initial set for r in all reallocation options to the fictitious only
P_initial = [422]




# %% Restricted master problem with intial solution included


terminate_iteration = False
iteration_count = 0

columns = []   

while not terminate_iteration and iteration_count < 20:
    if iteration_count == 0:
        # Decision variable: number of reallocated pax from itinerary p to r 
        t = {} 
        for p in P_i:
            for r in P_initial:
                t[p,r] = m2.addVar(lb=0,vtype=GRB.CONTINUOUS)

        # Objective function: minimizing spill costs
        m2.setObjective(
            quicksum((fare_i[p] - bpr[p, r] * fare_i[r]) * t[p, r] for p in P_i for r in P_initial),
            GRB.MINIMIZE
        )

        # Capacity constraint
        for i in L: 
            m2.addConstr(
                quicksum(delta_matrix[i][p] * t[p, r] for p in P_i for r in P_initial)
                - quicksum(delta_matrix[i][p] * bpr[r, p] * t[r, p] for r in P_i for p in P_initial)
                >= demand_i[i] - CAPi[i],
                name=f'CAP_{i}'
            )

        # Demand constraint
        for p in P_i:
            m2.addConstr(
                quicksum(t[p, r] for r in P_initial) <= Dp[p],
                name=f'D_{p}'
            )
        # Run model
        m2.optimize()

        initial_objective = m2.objVal
        print('Results of initial RMP (Iteration 0)')
        print('\nInitial objective function value: \t %g' % initial_objective)

        t_values = [] # list for t values
        for key, var in t.items(): 
            if var.X != 0: # Collect non zero values from t[p,r]
                t_values.append((key[0], key[1], var.X))  # Add the p and r values
                columns.append((key[0], key[1]))  # Voeg een dictionary toe

        # Make dataframe for t values
        df_t_values = pd.DataFrame(t_values, columns=["from itinerary p", "to fictitious", "Value"])

        # Print t table for initial solution
        print('\nThe t[p,r] of the initial solution of the RMP (Iteration 0)')
        print(df_t_values)
        print(f'\nCOUNT #{iteration_count}')

        iteration_count += 1

    else:
        # Calculate duals of constraints
        # Verkrijg de duale waarden van de CAP- en D-constrainten
        dual_CAP = [c.pi for c in m2.getConstrs() if c.ConstrName.startswith('CAP_')]
        dual_D = [c.pi for c in m2.getConstrs() if c.ConstrName.startswith('D_')]

        # # Doorloop de duale waarden in dual_CAP en controleer ze
        # for i, dual_value in enumerate(dual_CAP): 
        #     if dual_value != 0 and dual_value != -0:
        #         print(i, dual_value)

        # # Doorloop de duale waarden in dual_CAP en controleer ze
        # for i, dual_value in enumerate(dual_D): 
        #     if dual_value != 0 and dual_value != -0:
        #         print(i, dual_value)

        # Calculating reduced cost C_pr'
        reduced_costs = {(p, r): (fare_i[p] - sum(dual_CAP[i] for i in p_i_dict[p]) 
                                    - bpr[p, r] * (fare_i[r] - sum(dual_CAP[i] for i in p_i_dict[r])) 
                                    - dual_D[p] ) for p in P_i for r in P_i}

        for key, value in reduced_costs.items():
            if value < 0:
                print(f"{key}: {value}")

        # Add (key[0], key[1]) to 'columns' for each key in reduced costs
        for key, value in reduced_costs.items():
            if value < 0:
                columns.append((key[0], key[1]))  # Voeg (key[0], key[1]) toe 

        # 
        columns_df = pd.DataFrame(columns, columns = ["p", "r"])

        if all(reduced_costs[p, r] >= 0 for p, r in columns_df.itertuples(index=False)):
            terminate_iteration = True
            print('exit')


        # iteration Restricted Master Problem
        # Decision variable: number of reallocated pax from itinerary p to r 
        t = {} 
        for n in range(len(columns_df)):
            t[columns_df.iloc[n,0], columns_df.iloc[n,1]] = m2.addVar(lb=0,vtype=GRB.CONTINUOUS)

        # Objective function: minimizing spill costs
        m2.setObjective(
            quicksum((fare_i[columns_df.iloc[n,0]] 
                    - bpr[columns_df.iloc[n,0], columns_df.iloc[n,1]] 
                    * fare_i[columns_df.iloc[n,1]]) 
                    * t[columns_df.iloc[n,0], columns_df.iloc[n,1]] 
                    for n in range(len(columns_df))),
            GRB.MINIMIZE
        )

        # Capacity constraint
        for i in L:
            m2.addConstr(
                quicksum(delta_matrix[i][columns_df.iloc[n, 0]] * t[columns_df.iloc[n, 0], columns_df.iloc[n, 1]] 
                        for n in range(len(columns_df)))
                - quicksum(delta_matrix[i][columns_df.iloc[n, 0]] * bpr[columns_df.iloc[n, 1], columns_df.iloc[n, 0]] 
                        * t.get((columns_df.iloc[n, 1], columns_df.iloc[n, 0]), 0)  # Check if (r, p) exists
                        for n in range(len(columns_df)))
                >= demand_i[i] - CAPi[i],
                name=f'CAP_{i}'
            )
                    
        # Make combinations set for t(p,r) in columns
        valid_combinations = set(zip(columns_df['p'], columns_df['r']))

        # Maak de constraints
        for p in P_i:
            m2.addConstr(
                quicksum(
                    t[p, r] for r in P_i if (p, r) in valid_combinations
                ) <= Dp[p], 
                name="constraint5_{}".format(p)
            )

        # Run model
        m2.optimize()

        t_values = [] # list for t values
        for key, var in t.items(): 
            if var.X != 0: # Collect non zero values from t[p,r]
                t_values.append((key[0], key[1], var.X))  # Add the p and r values
                columns.append((key[0], key[1]))  # Voeg een dictionary toe

        # Make dataframe for t values
        df_t_values = pd.DataFrame(t_values, columns=["from itinerary p", "to itinerary r", "Value"])

        # Print t table
        print(f'\nThe t[p,r] of the solution of the {itaterion_count} iteration of the RMP')
        print(df_t_values)

        # for i in range(len(df_t_values)):
        #     print(f'From itinerary p: {df_t_values.iloc[i]["from itinerary p"]}, '
        #         f'To itinerary r: {df_t_values.iloc[i]["to itinerary r"]}, '
        #         f'Value: {df_t_values.iloc[i]["Value"]}')
        
        iteration_count += 1
        print(f'\nCOUNT #{iteration_count}')

# %%
