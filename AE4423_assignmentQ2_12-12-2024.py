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

#import data Maaike
pop_data_path = r'C:\Users\Marjan\OneDrive\Documents\Study\AE44232o Airline planning and optimisation\AE4423Data2\Group_1.xlsx'
demand_data_path = r'C:\Users\Marjan\OneDrive\Documents\Study\AE44232o Airline planning and optimisation\AE4423Data2\DemandGroup1.xlsx'

# #import data Suze
# pop_data_path = "C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment Q2\Group_1.xlsx"
# demand_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\DemandGroup1.xlsx'

# Path to the excel file
excel_file = r"C:\Users\Marjan\Documents\Study\AE44232o Airline planning and optimisation\AE4423Data2\Group_1.xlsx"
sheet_name_flights = "Flights"
sheet_name_itineraries = "Itineraries"
sheet_name_recapture = "Recapture"

# Lees het tabblad in een DataFrame
df_flights = pd.read_excel(excel_file, sheet_name=sheet_name_flights)
df_itineraries = pd.read_excel(excel_file, sheet_name=sheet_name_itineraries)
df_recapture = pd.read_excel(excel_file, sheet_name=sheet_name_recapture)

# Print een voorbeeld van de ingeladen data
print(df_flights)
print(df_itineraries)
print(df_recapture)

#%%

# Sets
L = set(df_flights['Flight No.']) # Set of flights
# P = set(df_itineraries['Itinerary'])
P = set(df_itineraries[['Flight 1', 'Flight 2']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths)
P_p = set(df_recapture[['From Itinerary', 'To Itinerary']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths) with recapture from itinerary p

print("Set of Flights:", L)
print("Set of Passenger Itineraries (Paths):", P)
print("Set of passenger itineraries (paths) with recapture from itinerary p", P_p)

#%%
# Parameters
flight = df_flights['Flight No.'].tolist()
average_fares = df_itineraries.groupby(['Origin', 'Destination'])['Price [EUR]'].mean()
averagefare = average_fares.to_dict() # Average fare for itinerary p
fare = df_itineraries['Price [EUR]']
Dp = df_itineraries.groupby(['Origin', 'Destination'])['Demand'].sum() # Daily unconstrained demand for itinerary p
CAPi = df_flights['Capacity'] # Capacity on flight (leg) i
b = df_recapture['Recapture Rate'] # Recapture rate of a pax that desires itinerary p and is allocated to r
itinerary_flights = df_itineraries['Itinerary']

# Computing binary variable delta[i, p] for checking if leg i is in path p
L_list = df_flights['Flight No.'].tolist() 
flight1 = df_itineraries['Flight 1'].tolist()
flight2 = df_itineraries['Flight 2'].tolist()
delta_matrix = np.zeros((len(L_list), len(flight1)))

# Making delta[i, p] matrix
for i in range(len(L_list)):
    for j in range(len(flight1)):
        if flight1[j] == L_list[i]:
            delta_matrix[i][j] = 1
        if flight2[j] == L_list[i]:
            delta_matrix[i][j] = 1

for row in delta_matrix:
    print(row)

# # Making b[p, r] matrix
# b_matrix = np.zerps ((len(flight1, len(flight1))))
# for p in P:
#     for r in P:
#         if bpr[0] == p and bpr[1] == r:
#             b[p][r] = 


#Computing unconstrained demand Q[i]
Dp_array = df_itineraries['Demand'].tolist()
demand_i = np.zeros(len(L))
for i in range(len(L)):  
    demand_i[i] = sum(Dp_array[p] * delta_matrix[i,p] for p in range(len(P)))  # Summing demand over all paths

#Computing the recapture rate b_pr
recapture_p_list = df_recapture['From Itinerary']
recapture_r_list = df_recapture['To Itinerary']
rate = df_recapture['Recapture Rate']
bpr = {(p,r): rate for p, r, rate in zip(recapture_p_list, recapture_r_list, rate)}

#Itinerary dictionary
itinerary_arc_good = {}
for p in range(len(flight1)):
    for i in range(len(L_list)):
        if flight1[p] == flight[i]:
            itinerary_arc_good[p] = i

itinerary_dict = {}
for key, value in itinerary_arc_good.items():
    for i in range(len(L_list)):
        if flight2[key] == 0:
            itinerary_dict[key] = [value]
        if flight2[key] != 0:
            if flight2[key] == flight[i]:
                itinerary_dict[key] = [value, i]


# Initial Restricted Master problem 
# Decision Variables
t = {}           # Number of passengers from itinerary p that will travel on itinerary r
for p in P:
    for r in P:
        t[p,r] = m2.addVar(lb=0,vtype=GRB.INTEGER)

# Set objective of minimizing spill costs
m2.setObjective(
    quicksum((fare[p] - fare[r]) * t[p][r] for p in P for r in P),
    GRB.MINIMIZE)
m2.update()

##Constraints 
#Capacity constraint

for i in L: 
    m2.addConstr(quicksum(delta_matrix[i][p] * t[p, r] for p in P for r in P) 
                - quicksum(delta_matrix[i][p] * bpr[p][r] * t[r, p] for p in P for r in P)
                >= demand_i[i] - CAPi[i], name = 'CAP')
    
#Demand constraint
for p in P:
    m2.addConstr(quicksum(t[p, r] for r in P_p) <= Dp[p], name = 'D')
m2.update()

#Restricted master problem

terminate_iteration = False
iteration_count = 0

added_columns = []
columns = []

while not terminate_iteration and iteration_count < 20:
    print('n\nStart new iteration')
    start_time = time.time()
    
    m2.optimize()
    m2_relaxed = m2.relax()
    m2_relaxed.optimize()
    m2.setParam('OutputFlag', 0)

    #generate duals of capacity and demand constraint
    dual_CAP = [c.pi for c in m2_relaxed.getConstrs() if c.ConstrName == 'CAP']
    dual_D = [c.pi for c in m2_relaxed.getConstrs() if c.ConstrName == 'D']

    #Calculating reduced cost C_pr'
    reduced_costs = {(p, r): (fare[p] - sum(dual_CAP[i] * delta_matrix[i, p] for i in L) 
                                - bpr.get((p, r), 0) * (fare[r] - sum(dual_CAP[i] * delta_matrix[i, p] for i in L)) 
                                - dual_D[p] ) for p in P for r in P}
    # Columns with negative dual value
    negative_columns = [(p, r) for p in P for r in P if (p, r) not in columns and reduced_costs[p, r] < -0.0001]

    for i in negative_columns:
        new_column = Column()
        for b in itinerary_dict[i[0]]:
            new_column.addTerms((delta_matrix[i[0]][b] - delta_matrix[i[1]][b] * bpr.get((i[0], i[1]), 0)), m2.getConstrByName('C4_%s'%(b)))
        new_column.addTerms(1, m2.getConstrByName("D"%(i[0])))

        t[i[0], i[1]] = m2.addVar(obj=fare[i[0]] - bpr.get((i[0], i[1]), 0) * fare[i[1]], lb=0, vtype=GRB.INTEGER, column=new_column)
        added_columns.append((i[0], i[1]))
        columns.append((i[0], i[1]))

    m2.update()





# %%
