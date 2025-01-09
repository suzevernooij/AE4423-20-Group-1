# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:13:45 2024

@author: julia
"""

# Import necessary libraries
import gurobipy as gp  
from gurobipy import *  
from gurobipy import GRB  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

#name model
m = Model('AirlinePlanningAssingment2')

#import data Suze
# airport_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\Group1.xlsx"

#import data Maaike

# #import data Julia
# airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"

#import data Julia
airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"


# Read Excel Files
df_airports = pd.read_excel(airport_data)
df_fleet = pd.read_excel(fleet_data)
df_demand = pd.read_excel(demand_data)

# print(df_airports)
# print(df_fleet)
# print(df_demand)

#Defining Parameters Fleet
fleet       = pd.read_excel(fleet_data, skiprows=0, header=0, index_col=0)
speed       = np.array(fleet.iloc[0])
capacity    = np.array(fleet.iloc[1])
TAT         = np.array(fleet.iloc[2])
R           = np.array(fleet.iloc[3])
runway_AC   = np.array(fleet.iloc[4])
Cl          = np.array(fleet.iloc[5])
Cf          = np.array(fleet.iloc[6])
Ch          = np.array(fleet.iloc[7])
cfuel       = np.array(fleet.iloc[8])
fleet       = np.array(fleet.iloc[9])

#Defining Parameters Airport
airport     = pd.read_excel(airport_data, skiprows=0, header=0, index_col=0)
IATA        = np.array(airport.iloc[0])
latitude    = airport.iloc[1].tolist()
longitude   = airport.iloc[2].tolist()
runway_AP   = np.array(airport.iloc[3])

Re = 6371 
f = 1.42
Yield = 0.26

#Defining Sets
n = len(IATA)
N = range(len(IATA))
K = range(len(speed))

# Haversine Distance Calculation
lat_rad = np.radians(latitude)
lon_rad = np.radians(longitude)
d = np.zeros((n, n))  # Distance matrix
for i in range(n):
    for j in range(n):
        delta_lat = lat_rad[i] - lat_rad[j]
        delta_lon = lon_rad[i] - lon_rad[j]
        a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[i]) * math.cos(lat_rad[j]) * math.sin(delta_lon / 2)**2
        sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d[i, j] = Re * sigma
d = np.array(d)

total_costs = [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]
for k in K:
    for i in N:
        for j in N:
            if i == j:
                CF = 0
            else:
                CF = Cf[k] 
            CT = Ch[k] * d[i,j] / speed[k]
            CFUEL = (cfuel[k] * f / 1.5) * d[i,j]
            total = CF + CT + CFUEL
            total_costs[k][i,j] = total

flight_time = np.zeros((n,n)) #Nu in minutes, willen we dat????
for k in K:
    for i in N:
        for j in N:
            if i != j:
                flight_time[i,j] =  d[i,j]/speed[k] * 60 + TAT[k] + 30

#%%
# Clean the df_demand dataframe
df_demand.columns = df_demand.iloc[1]  # Set the second row as header
df_demand = df_demand[4:]  # Drop rows above the data
df_demand.reset_index(drop=True, inplace=True)

# Select the relevant columns: Origin, Destination, and all time bins (6 time bins * 5 days)
time_bins = ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-00:00']
df_demand_clean = df_demand.iloc[:, [1, 2] + [i for i, col in enumerate(df_demand.columns) if any(bin in str(col) for bin in time_bins)]]

# # Rename columns for time-bins (each day has 6 bins)
# df_demand_clean.columns = ['Origin', 'Destination'] + [f"Day_{i//6+1}_Bin_{i%6+1}" for i in range(len(df_demand_clean.columns) - 2)]
df_demand_clean.columns = ['Origin', 'Destination'] + [f"Day_{i//6+1}_Bin_{i%6+1}" for i in range(len(df_demand_clean.columns) - 2)]

# Filter the data for only FRA as Origin or Destination
df_demand_hub = df_demand_clean[(df_demand_clean['Origin'] == 'FRA') | (df_demand_clean['Destination'] == 'FRA')]

# Convert the demand data to integers and extract it into a NumPy array
demand_array = df_demand_hub.iloc[:, 2:].astype(int).values  # Extracting the demand values and converting to integers

# Print the resulting demand array
print(demand_array)
print(df_demand_hub)
            
#%%
### MOET NOG AANGEPAST WORDEN! DENK DAT WE MOETEN KIJKEN NAAR DE CODE VAN AIRLINEPLANNING3
# Initialize a new array for adjusted demand
adjusted_demand = np.zeros_like(demand_array)

# Number of time bins per day (6)
n_bins = 6

# Adjust the demand according to the rule (capture 100% of the current bin and 20% of previous two bins)
for i in range(demand_array.shape[0]):  # Loop over all rows
    for t in range(n_bins * 5):  # Loop over all time bins (5 days * 6 bins per day)
        # Current bin (t)
        adjusted_demand[i, t] = demand_array[i, t]
        
        # Two previous bins (20% of each)
        if t > 0:
            adjusted_demand[i, t] += 0.2 * demand_array[i, t - 1]  # 20% of previous bin
        
        if t > 1:
            adjusted_demand[i, t] += 0.2 * demand_array[i, t - 2]  # 20% of the bin before the previous one

# Print the adjusted demand array
print(adjusted_demand)

#%%

adjusted_demand = np.zeros_like(demand_array)

# Number of time bins per day (6)
n_bins = 6

# Adjust the demand according to the rule (capture 100% of the current bin and 20% of previous two bins)
def fly_demand_def(k, demand_hour):
    for i in range(demand_array.shape[0]):  # Loop over all rows
        for t in range(n_bins * 5):  # Loop over all time bins (5 days * 6 bins per day)
            # Current bin (t)
            adjusted_demand[i, t] = demand_array[i, t]
            if adjusted_demand[i,t] < capacity[k]:
                adjusted_demand[i,t] += demand_hour[i,t]
            # Two previous bins (20% of each)
                if t > 0:
                    if adjusted_demand[i, t] + demand_array[i, t-1] > capacity[k]:
                        adjusted_demand[i, t] = capacity[k]
                    else:
                        adjusted_demand[i, t] = 0.2 * demand_array[i, t - 1]  # 20% of previous bin
                
                if t > 1:
                    if adjusted_demand[i, t] + demand_array[i, t-2] > capacity[k]:
                        adjusted_demand[i, t] = capacity[k]
                    else:
                        adjusted_demand[i, t] = 0.2 * demand_array[i, t - 2]  # 20% of the bin before the previous one
            else:
                adjusted_demand[i,t] = capacity[k]
    # a = fly_demand
    # b = np.clip(a, 0, np.inf)
    return adjusted_demand ##hier iets toevoegen zodat de demand niet onder nul komt.       

# Print the adjusted demand array
print(adjusted_demand)

# Kies een specifieke fleet-type index
k = 0  # Voorbeeld fleet-type
demand_hour = demand_array  # Stel demand_hour in

# Roep de functie aan
adjusted_demand_result = fly_demand_def(k, demand_hour)

# Print het resultaat
print(adjusted_demand_result)




#%%
# Define revenue
# Define total profit
# Define travel time
# Interval time is 6 minutes 
# Flight time per flight leg in minutes
#
def operating_costs(distance, k):
    fixed_cost = Cf[k]
    time_cost = Ch[k]*distance/speed[k]
    fuel_cost = cfuel[k]*(f/1.5)*distance
    total_costs = (fixed_cost + time_cost + fuel_cost)
    return total_costs

def revenue(distance, fleet):
    revenue = Yield*distance*fleet
    return revenue

def total_profit(distance, k, fleet):
    profit = revenue(distance,fleet) - operating_costs(distance, k) 
    return profit

def travel_time(distance, k): #in minutes
    duration =  distance/speed[k] * 60 + TAT[k] + 30
    return duration

# one stage is 4 hour. There are 6 stages in one day and 5 days in total so 30
#stages = np.arange(0, 30)
flight_times = []
for i in range(0,n):
    for j in range(0,n):
        flight_time = travel_time(distance[i,j], 0) #ac type 1
        flight_times.append(flight_time)
flight_times_matrix = np.array(flight_times).reshape((n, n))
flight_times_units = flight_times_matrix/6
#%%
# Check if aircraft k can land/take-off at airport i
for k in K:
    for i in N:
        if runway_AC[k] > runway_AP[i] or runway_AC[k] > runway_AP[j]:
            print(f'ac {k} cannot land, runway {runway_AP[j]} too short')
            

#%%
# check which aircraft can fly which leg. Range ac > distance_ij
ac_per_leg = np.ones((3, 20, 20))
for k in K:
    for i in N:
        for j in N:
            if i != j:
                # Convert values to float if they are numeric
                try:
                    distance_ij = float(distance[i, j])
                    range_ac_k = float(R[k])
                    if distance_ij > range_ac_k:
                        ac_per_leg[k,i,j] = 0 # use ac_per_leg to find if ac 1, 2 or 3 flies the specific path
                        # print(f'range ac {k} is too short for leg {i} to {j}')
                except ValueError:
                    pass

#%%
### Dynamic programming ###

#%%
### Scheduling ###

#%%
### Selecting best option (highest profit) ###

#%%
### Remove Demand ###
# Compare the profits, if profit is highest --> save aircraft route --> remove demand transported

#%%
### Results ###




