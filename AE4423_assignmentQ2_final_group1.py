#%%
# # -*- coding: utf-8 -*-
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
import time


# --------------- Importing data -----------------------

# Path to the excel file
excel_file = r"C:\Users\Marjan\Documents\Study\AE44232o Airline planning and optimisation\AE4423Data2\Group_1.xlsx"
# excel_file = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment Q2\Group_1.xlsx"
sheet_name_flights = "Flights"
sheet_name_itineraries = "Itineraries"
sheet_name_recapture = "Recapture"

# Read tabs of excel
df_flights = pd.read_excel(excel_file, sheet_name=sheet_name_flights)
df_itineraries = pd.read_excel(excel_file, sheet_name=sheet_name_itineraries)
df_recapture = pd.read_excel(excel_file, sheet_name=sheet_name_recapture)

# Making Sets
L = set(df_flights['Flight No.']) # Set of flights
P = set(df_itineraries[['Flight 1', 'Flight 2']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths)
P_p = set(df_recapture[['From Itinerary', 'To Itinerary']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths) with recapture from itinerary p

print("Set of Flights:", L)
print("Set of Passenger Itineraries (Paths):", P)
print("Set of passenger itineraries (paths) with recapture from itinerary p", P_p)

# Defining Parameters
flights = df_flights['Flight No.'].tolist() # List of flights with flight numbers
fare = df_itineraries['Price [EUR]'] # Fare per itinerary in EU
Dp = df_itineraries['Demand'].tolist() # Daily unconstrained demand for itinerary p
CAPi = df_flights['Capacity'] # Capacity on flight (leg) i
b = df_recapture['Recapture Rate'] # Recapture rate of a pax that desires itinerary p and is allocated to r
flight1 = df_itineraries['Flight 1'].tolist() # List of flight leg 1 in paths
flight2 = df_itineraries['Flight 2'].tolist() #List of flight leg 2 in paths
recapture_p_list = df_recapture['From Itinerary'] # P's from recapture rates from path p to path r
recapture_r_list = df_recapture['To Itinerary'] # R's from recapture rates from path p to path r
rate = df_recapture['Recapture Rate'] # Recapture rate from path p to r



# ------- Computing needed parameters ---------------

# Change sets to ranges
P = range(len(flight1))
L = range(len(flights))

# Computing binary variable delta[i, p] for checking if leg i is in path p
delta_matrix = np.zeros((len(L), len(P))) # Making zero matrix with rows: flights legs, columns: all itineraries
for i in range(len(flights)):
    for j in range(len(flight1)):
        if flight1[j] == flights[i]:
            delta_matrix[i][j] = 1
        if flight2[j] == flights[i]:
            delta_matrix[i][j] = 1

#Computing unconstrained demand per flight leg Q[i]
demand_i = np.zeros(len(L))
for i in range(len(L)):  
    demand_i[i] = sum(Dp[p] * delta_matrix[i,p] for p in range(len(P)))  # Summing demand over all paths

# Computing the recapture rate b_pr
bpr = {(p,r): rate for p, r, rate in zip(recapture_p_list, recapture_r_list, rate)}

# Checking paths with at least one flight exceeding capacity
for p in P:  
    for i in L:  
        if delta_matrix[i, p] == 1:  # Check if leg i is part of path p
            if demand_i[i] > CAPi[i]:  # Check if demand exceeds capacity for leg i
                print(f"Path {p} has a capacity problem at leg {i}")
                break  # No need to check further legs for this path


# -------- Adding Parameters For Initial Restricted Master problem  ----------

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
Dp.append(100000) # Adds a high limit of demand for the fictitious itinerary 
delta_matrix = np.append(delta_matrix,np.zeros([len(delta_matrix),1]),1) # delta is zero for each flight leg in the ficitious itinerary


#%%
## ----------- Column Generation Algorithm -----------------------

start_time = time.time() # Set start time Algorithm
terminate_iteration = False # Column generation status for optimal solution
iteration_count = 0 
columns = [(p, 422) for p in P_i[:-1]] # Initial set of columns (only to the fictitioys itinerary p = 422)

while not terminate_iteration and iteration_count < 5: #Looping til optimal solution is found or iteration count < 5
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

    # Store non null solutions of the decision variables
    t_values = []
    for key, var in t.items():
        if var.X > 0:
            t_values.append((key[0], key[1], var.X))
    
    # Print iteration results for #iteration
    print(f"\nIteration {iteration_count}")
    print(f"Objective Value: {m2.objVal}")
    print(pd.DataFrame(t_values, columns=["from_itinerary_p", "to_itinerary_r", "value"]))

    # Generate duals of model with iteration count #.
    dual_CAP = [c.Pi for c in m2.getConstrs() if c.ConstrName.startswith('CAP_')]
    dual_D = [c.Pi for c in m2.getConstrs() if c.ConstrName.startswith('D_')]

    # Calculate reduced costs with generated duals
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
        terminate_iteration = True # Stop algorithm if no new reduced costs columns are added
        print("Column generation successfully completed.")
        print(f"Final Objective Value: {m2.objVal}")
    else:
        for col in new_columns:
            if col not in columns: # Add new columns if not already in columns
                columns.append(col)

    # # Printing code for initial solutions in report
    # if iteration_count == 0:
    #     # Print first five dual values of Capacity and Demand constraints
    #     for i, dual_value in [(i, dv) for i, dv in enumerate(dual_CAP) if dv != 0][:5]:
    #         print(f"Dual_CAP[{i}] = {dual_value}")
    #     for i, dual_value in [(i, dv) for i, dv in enumerate(dual_D) if dv != 0][:5]:
    #         print(f"Dual_D[{i}] = {dual_value}")

    iteration_count += 1 # If there are new negative reduced costs, make a new iteration
    
    # Handle the case where maximum iterations are reached
    if iteration_count >= 5 and not terminate_iteration:
        print("\nMaximum number of iterations reached. Termination.")

# Print Total runtime
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime of columng generation algorithm: {total_runtime}")
