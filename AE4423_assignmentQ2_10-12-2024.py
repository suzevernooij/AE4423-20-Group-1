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
m = Model('AirlinePlanningAssingment1_Q2')

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
P = set(df_itineraries[['Flight 1', 'Flight 2']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths)
P_p = set(df_recapture[['From Itinerary', 'To Itinerary']].itertuples(index=False, name=None)) # Set of passenger itineraries (paths) with recapture from itinerary p

print("Set of Flights:", L)
print("Set of Passenger Itineraries (Paths):", P)
print("Set of passenger itineraries (paths) with recapture from itinerary p", P_p)

#%%
# Parameters
average_fares = df_itineraries.groupby(['Origin', 'Destination'])['Price [EUR]'].mean()
fare = average_fares.to_dict() # Average fare for itinerary p
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

#Computing unconstrained demand Q[i]
Dp_array = df_itineraries['Demand'].tolist()
demand_i = np.zeros(len(L))
for i in range(len(L)):  
    demand_i[i] = sum(Dp_array[p] * delta_matrix[i,p] for p in range(len(P)))  # Summing demand over all paths

# delta = {}
# for _, row in df_itineraries.iterrows():
#     itinerary = row['Itinerary']
#     if pd.notna(row['Flight 1']):  # If Flight 1 is not NaN, add to delta
#         delta[(row['Flight 1'], itinerary)] = 1
#     if pd.notna(row['Flight 2']):  # If Flight 2 is not NaN, add to delta
#         delta[(row['Flight 2'], itinerary)] = 1  # Flight-path inclusion

# #%%
# # Define δp as a dictionary: (itinerary, flight) -> 1 if flight is part of the itinerary, else 0
# delta = {
#     (itinerary, flight): 1 if flight in flights else 0
#     for itinerary, flights in itinerary_flights.items()  # Iterate over each itinerary and its flights
#     for flight in df_flights['Flight No.'].unique()  # Iterate over all unique flights in df_flights
# }

# print("Delta Mapping (δp):")
# print(δp)

# %%
# Parameters
average_fares = df_itineraries.groupby(['Origin', 'Destination'])['Price [EUR]'].mean()
fare = average_fares.to_dict() # Average fare for itinerary p
Dp = df_itineraries.groupby(['Origin', 'Destination'])['Demand'].sum() # Daily unconstrained demand for itinerary p
CAPi = df_flights['Capacity'] # Capacity on flight (leg) i
b = df_recapture['Recapture Rate'] # Recapture rate of a pax that desires itinerary p and is allocated to r
itinerary_flights = df_itineraries['Itinerary']

# Computing binary variable delta[i, p] for checking if leg i is in path p
L_list = df_flights['Flight No.'].tolist() 
flight1 = df_itineraries['Flight 1'].tolist()
flight2 = df_itineraries['Flight 2'].tolist()
delta_matrix = np.zeros((len(L_list), len(flight1)))

for i in range(len(L_list)):
    for j in range(len(flight1)):
        if flight1[j] == L_list[i] or flight2[j] == L_list[i]:
            delta_matrix[i][j] = 1
'
print("Delta matrix:")
for row in delta_matrix:
    print(row)

# Vraag per pad (direct uit de DataFrame halen)
Dp = df_itineraries['Demand'].tolist()


# Berekening van de vraag per vlucht
demand_i = np.zeros(len(L))  # Dit gaat de vraag per vlucht bevatten

for i, flight in enumerate(L):  # Itereren over alle vluchten
    demand_i[i] = sum(Dp[p] * delta_matrix[p, i] for p in range(len(P)))  # Som van de vraag gewogen door delta_matrix

# Output de vraag per vlucht
print("Vraag per vlucht (demand_i):")
print(demand_i)
# delta = {}
# for _, row in df_itineraries.iterrows():
#     itinerary = row['Itinerary']
#     if pd.notna(row['Flight 1']):  # If Flight 1 is not NaN, add to delta
#         delta[(row['Flight 1'], itinerary)] = 1
#     if pd.notna(row['Flight 2']):  # If Flight 2 is not NaN, add to delta
#         delta[(row['Flight 2'], itinerary)] = 1  # Flight-path inclusion

# #%%
# # Define δp as a dictionary: (itinerary, flight) -> 1 if flight is part of the itinerary, else 0
# delta = {
#     (itinerary, flight): 1 if flight in flights else 0
#     for itinerary, flights in itinerary_flights.items()  # Iterate over each itinerary and its flights
#     for flight in df_flights['Flight No.'].unique()  # Iterate over all unique flights in df_flights
# }

# print("Delta Mapping (δp):")
# print(δp)

# %%
demand

# %%
demand_i

# %%
for i, flight in enumerate(L):  # Itereren over alle vluchten
    demand_i[i] = sum(Dp[p] * delta_matrix[p, i] for p in range(len(P)))  # Som van de vraag gewogen door delta_matrix

# %%
demand

# %%
demand_i

# %%
len(demand_i)

# %%
Dp

# %%
for i in L:  # Itereren over alle vluchten
    demand_i[i] = sum(Dp[p] * delta_matrix[p, i] for p in range(len(P)))  # Som van de vraag gewogen door delta_matrix

# %%
for i in range(len(L)):  # Itereren over alle vluchten
    demand_i[i] = sum(Dp[p] * delta_matrix[p, i] for p in range(len(P)))  # Som van de vraag gewogen door delta_matrix

# %%
deman

# %%
demand_i

# %%
DELTA

# %%
delta_matrix


