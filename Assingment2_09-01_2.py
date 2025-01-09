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

# import data Suze
airport_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\Group1.xlsx"

#import data Maaike

# #import data Julia
# airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"


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
        flight_time = travel_time(d[i,j], 0) #ac type 1
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
                    distance_ij = float(d[i, j])
                    range_ac_k = float(R[k])
                    if distance_ij > range_ac_k:
                        ac_per_leg[k,i,j] = 0 # use ac_per_leg to find if ac 1, 2 or 3 flies the specific path
                        # print(f'range ac {k} is too short for leg {i} to {j}')
                except ValueError:
                    pass

#%%
# Define a function to get possible actions with i and j
def get_possible_actions(i, k, time):
    possible_actions = []
    time_bin = time // 40  # Map 6-minute intervals to 4-hour bins

    if time_bin < adjusted_demand.shape[1]:  # Ensure time_bin is within bounds
        for j in N:  # Loop through destinations
            if i != j and d[i, j] <= R[k] and runway_AC[k] <= runway_AP[j]:  # Range and runway constraints
                cargo_load = min(adjusted_demand[i, time_bin], capacity[k])
                possible_actions.append((j, cargo_load))
    
    return possible_actions

#%%
### Dynamic programming ###
# Define states with i and j
states = []
for time in range(0, 24*10*5):  # 5 days, 10 stages/hour
    for i in range(len(IATA)):  # Each airport
        for k in range(len(fleet)):  # Each aircraft type
            states.append((time, i, k))  # Use i instead of airport
#%%
# Define a function for the Dynamic Programming initialization and execution
def dynamic_programming():
    # For each aircraft type do
    for k in K:
        # Set time step duration, total time steps, end_vector, and state space matrices
        time_step_duration = 6  # 6 time bins per day
        total_time_steps = 24 * 5 * 6  # 5 days, 6 time bins per day = 30 stages per day
        
        # Initialize DP and backtracking matrices
        dp = np.zeros((total_time_steps, n, len(K)))  # DP matrix: time x airports x aircraft types
        backtrack = np.zeros((total_time_steps, n, len(K)))  # Backtracking matrix
        
        # For each iteration from 0 to 240 (backtracking) do
        for t in range(total_time_steps - 1, -1, -1):  # Start from last timestep
        
            # Set the last 240 timesteps in state space matrix (initialize profit for last timestep)
            if t == total_time_steps - 1:
                dp[t] = 0  # No profit at the end of the planning period
            
            # For each timestep starting from the end of the day until the last timestep do
            for i in N:  # Use i as the current airport (previously airport)
                for k in K:
                    possible_actions = get_possible_actions(i, k, t)  # Calculate actions for each state

                    # For each arrival airport do
                    for action in possible_actions:
                        j, cargo_load = action  # Unpack the action (destination airport, cargo load)
                        distance = d[i, j]  # Distance from current airport i to destination j
                        flight_time = flight_times_matrix[i, j]  # Flight time in minutes
                        
                        # Check conditions for penalties, calculate profit, and update state space matrices
                        immediate_profit = total_profit(distance, k, cargo_load)
                        next_time = t + int(np.ceil(flight_time / time_step_duration))  # Update to next time step

                        # Ensure the next state is within the allowed time range
                        if next_time < total_time_steps:
                            # Calculate the total profit including future profits (from next state)
                            next_state_profit = dp[next_time, j, k]  # Use j as the destination airport
                            total_profit_value = immediate_profit + next_state_profit

                            # Update DP and backtracking matrices if the current action is better
                            if total_profit_value > dp[t, i, k]:
                                dp[t, i, k] = total_profit_value
                                backtrack[t, i, k] = j  # Store the destination airport index (use j)
        
        # Append resulting state space matrices to aircraft-specific lists
        # For now, just return the dp and backtrack matrices for each aircraft type
        print(f"DP and Backtrack for aircraft {k}:")
        print(dp)
        print(backtrack)

# Execute the dynamic programming function
dynamic_programming()
#%%
print("DP Matrix Sample:", dp[:5, :5, 0])  # Print a subset for aircraft type 0
print("Backtrack Matrix Sample:", backtrack[:5, :5, 0])

#%%
### Scheduling ###
def extract_routes(dp, backtrack):
    routes = []  # To store routes for all aircraft
    utilizations = []  # To track utilization for each aircraft
    flightID = []  # Detailed flight schedule
    
    for k in K:  # Iterate over aircraft types
        t, i = 0, 0  # Start at hub
        route = []
        total_block_time = 0  # Track total block time for utilization
        
        while t < total_time_steps:
            j = int(backtrack[t, i, k])  # Get next destination
            if j == -1:  # No valid destination
                break
            
            # Calculate travel time and update time
            travel_duration = travel_time(d[i, j], k)
            arrival_time = t + int(np.ceil(travel_duration / time_step_duration))
            
            # Ensure arrival time is within the planning horizon
            if arrival_time >= total_time_steps:
                break
            
            # Calculate block time (travel + turnaround)
            block_time = int(np.ceil(travel_duration)) + TAT[k]
            
            # Record the flight details
            cargo = min(adjusted_demand[i, t], capacity[k])  # Cargo transported
            flightID.append({
                "Aircraft": k,
                "From": i,
                "To": j,
                "DepTime": t,
                "ArrTime": arrival_time,
                "BlockTime": block_time,
                "Cargo": cargo,
            })
            
            # Update state and metrics
            route.append((t, i, j))
            total_block_time += block_time
            t = arrival_time
            i = j
        
        routes.append(route)  # Append the route for this aircraft
        utilizations.append(total_block_time / (total_time_steps * time_step_duration) * 100)  # Utilization percentage
    
    return routes, flightID, utilizations

dp, backtrack = dynamic_programming()
routes, flightID, utilizations = extract_routes(dp, backtrack)

# Print results
print("Optimal Routes:", routes)
print("Flight Schedule:")
for flight in flightID:
    print(flight)
print("Utilization Metrics:", utilizations)

#%%
# # Dynamic programming table
# dp = {}  # Dictionary to store the best profit for each state

# # Backtracking table
# backtrack = {}

# # Starting point: Initialize costs for the final time stage
# for state in states:
#     time, airport, aircraft = state
#     if time == (24*10*5) - 1:  # 24*10*5: Represents the total number of time stages. 24*10*5 - 1: Last time stage
#         dp[state] = 0  # No profit at the end
#         backtrack[state] = None

# # Recursive case: Iterate backward through time
# for time in range((24 * 10 * 5) - 2, -1, -1):  # Iterate from second-to-last time stage
#     for state in states:  # Iterate over all states
#         current_airport, current_aircraft = state[1], state[2]
#         possible_actions = get_possible_actions(current_airport, current_aircraft, time)  # Define this function
        
#         # Evaluate each possible action
#         for action in possible_actions:
#             destination, load = action  # Example action: (destination_airport, cargo_load)
#             distance = d[current_airport, destination]  # Distance between airports
#             flight_time = flight_time[current_airport, destination]  # Flight time
            
#             # Calculate transition costs and profits
#             immediate_profit = total_profit(distance, current_aircraft, load)
#             next_state_time = time + int(np.ceil(flight_time / 6))  # Update time in stages
#             next_state = (next_state_time, destination, current_aircraft)  # Define the next state
            
#             # Future profit from the next state
#             future_profit = dp.get(next_state, float('-inf'))
#             total_profit_value = immediate_profit + future_profit
            
#             # Update DP and backtrack tables
#             if state not in dp or total_profit_value > dp[state]:
#                 dp[state] = total_profit_value
#                 backtrack[state] = action
# #%%
# # Initialize the starting state
# hub_code = "FRA"  # Replace with your hub's IATA code
# hub_index = np.where(IATA == hub_code)[0][0]
# starting_aircraft = 0  # Use the first aircraft

# current_state = (0, hub_index, starting_aircraft)

# # Backtrack to extract the optimal policy
# optimal_policy = []
# while current_state in backtrack and backtrack[current_state] is not None:
#     optimal_policy.append(backtrack[current_state])
#     current_state = calculate_next_state(current_state, backtrack[current_state])

# print("Optimal Policy:", optimal_policy)

#%%
### Scheduling ###

#%%
### Selecting best option (highest profit) ###

#%%
### Remove Demand ###
# Compare the profits, if profit is highest --> save aircraft route --> remove demand transported

#%%
### Results ###




