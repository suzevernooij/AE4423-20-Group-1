# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:00:14 2024

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

# Path to the excel file
excel_file = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment Q2\Group_1.xlsx"
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
#%%
## RESULTAAT!! -> 3 iteraties obj = 1758366.314067442
# Iterative column generation
terminate_iteration = False
iteration_count = 0
columns = [(p, 422) for p in P_i[:-1]]

while not terminate_iteration and iteration_count < 5:
    # Start a new model for each iteration
    m2 = gp.Model(f"RMP_Iteration_{iteration_count}")
    
    # Add decision variables
    t = {}
    for p, r in columns:
        t[p, r] = m2.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{p}_{r}")

    # Objective function
    m2.setObjective(
        gp.quicksum((fare_i[p] - bpr.get((p, r), 0) * fare_i[r]) * t[p, r] for p, r in columns),
        GRB.MINIMIZE
    )

    # Capacity constraints
    for i in L:
        m2.addConstr(
            gp.quicksum(delta_matrix[i][p] * t[p, r] for p, r in columns)
            - gp.quicksum(delta_matrix[i][p] * bpr.get((r, p), 0) * t[r, p] for r, p in columns)
            >= demand_i[i] - CAPi[i],
            name=f"CAP_{i}"
        )

    # Demand constraints
    for p in P_i:
        m2.addConstr(
            gp.quicksum(t[p, r] for r in P_i if (p, r) in columns) <= Dp[p],
            name=f"D_{p}"
        )

    # Optimize the model
    m2.optimize()

    # Store solution variables
    t_values = []
    for key, var in t.items():
        if var.X > 0:
            t_values.append((key[0], key[1], var.X))

    # Print iteration results
    print(f"\nIteration {iteration_count}")
    print(f"Objective Value: {m2.objVal}")
    print(pd.DataFrame(t_values, columns=["from_itinerary_p", "to_itinerary_r", "value"]))

    # Calculate reduced costs
    dual_CAP = [c.Pi for c in m2.getConstrs() if c.ConstrName.startswith('CAP_')]
    dual_D = [c.Pi for c in m2.getConstrs() if c.ConstrName.startswith('D_')]

    reduced_costs = {}
    for p in P_i:
        for r in P_i:
            reduced_costs[p, r] = (
                fare_i[p]
                - sum(dual_CAP[i] for i in range(len(L)) if delta_matrix[i][p] == 1)
                - bpr.get((p, r), 0) * (
                    fare_i[r]
                    - sum(dual_CAP[i] for i in range(len(L)) if delta_matrix[i][r] == 1)
                )
                - dual_D[p]
            )

    # Filter and print reduced costs < 0
    negative_reduced_costs = [
        (p, r, cost) for (p, r), cost in reduced_costs.items() if cost < -0.001
    ]
    if negative_reduced_costs:
        print("\nReduced Costs (only < 0):")
        print(pd.DataFrame(negative_reduced_costs, columns=["from_itinerary_p", "to_itinerary_r", "reduced_cost"]))

    # Check for negative reduced costs
    new_columns = [(p, r) for (p, r), cost in reduced_costs.items() if cost < -0.001]
    
    if not new_columns:
        terminate_iteration = True
        print("Column generation successfully completed.")
        print(f"Final Objective Value: {m2.objVal}")
    else:
        for col in new_columns:
            if col not in columns:
                columns.append(col)

    iteration_count += 1
    
    # Handle the case where maximum iterations are reached
    if iteration_count >= 5 and not terminate_iteration:
        print("\nMaximum number of iterations reached. Termination.")