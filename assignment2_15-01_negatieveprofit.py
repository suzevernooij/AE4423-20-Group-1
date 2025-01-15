# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:22:32 2025

@author: suzev
"""

# Import necessary libraries
  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

#import data Suze
airport_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\Group1.xlsx"

#import data Maaike




#import data Julia
# airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"

# Read Excel Files
df_airports = pd.read_excel(airport_data)
df_fleet = pd.read_excel(fleet_data)
df_demand = pd.read_excel(demand_data, header=None)

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
T = range(120*10)

#%%
# Correct demand function to include contributions from previous time bins
def compute_step_demand(demand_data, IATA, hub_index):

    step_demand = np.zeros((2, len(IATA), 1200))  # Shape: direction (to/from hub) x airports x time steps
    time_bins = 240  # 1200 minutes / 6-minute intervals per bin

    # Iterate over each airport and time window
    for i in range(len(IATA)):
        for t in range(1200):
            # Determine the current 4-hour bin
            current_bin = int(t / 40)

            # Demand from hub to airport
            demand_from_current = float(demand_data[(demand_data[1] == IATA[hub_index]) & (demand_data[2] == IATA[i])][current_bin + 3])
            demand_from_prev1 = 0.2 * float(demand_data[(demand_data[1] == IATA[hub_index]) & (demand_data[2] == IATA[i])][current_bin + 2]) if current_bin > 0 else 0
            demand_from_prev2 = 0.2 * float(demand_data[(demand_data[1] == IATA[hub_index]) & (demand_data[2] == IATA[i])][current_bin + 1]) if current_bin > 1 else 0

            # Assign to step_demand for the current time step
            step_demand[0][i][t] = demand_from_current + demand_from_prev1 + demand_from_prev2

            # Demand from airport to hub
            demand_to_current = float(demand_data[(demand_data[1] == IATA[i]) & (demand_data[2] == IATA[hub_index])][current_bin + 3])
            demand_to_prev1 = 0.2 * float(demand_data[(demand_data[1] == IATA[i]) & (demand_data[2] == IATA[hub_index])][current_bin + 2]) if current_bin > 0 else 0
            demand_to_prev2 = 0.2 * float(demand_data[(demand_data[1] == IATA[i]) & (demand_data[2] == IATA[hub_index])][current_bin + 1]) if current_bin > 1 else 0

            # Assign to step_demand for the current time step
            step_demand[1][i][t] = demand_to_current + demand_to_prev1 + demand_to_prev2

    return step_demand

# Hub index is 3
hub_index = 3

# Calculate demand
step_demand = compute_step_demand(df_demand, IATA, hub_index)

# Debugging: Check initial demand values
print("Initial step demand (first 10 values):", step_demand[:, :, :10])


#%%
# def demand(dep, arr, tw):
#     return float(df_demand[(df_demand[1] == dep) & (df_demand[2] == arr)][tw+3])

# step_demand = np.zeros((2,20,1200))
# #initalize hub = 6
# for i in N:
#     for t in T:
#         step_demand[0][i][t] = demand(IATA[3], IATA[i], int(t/40))
# for i in N:
#     for t in T:
#         step_demand[1][i][t] = demand(IATA[i], IATA[3], int(t/40))

#%%
# Haversine Distance Calculation
lat_rad = np.radians(latitude)
lon_rad = np.radians(longitude)

IATA_index = list(IATA)
def distance(dep, arr):
    delta_lat = lat_rad[IATA_index.index(dep)] - lat_rad[IATA_index.index(arr)]
    delta_lon = lon_rad[IATA_index.index(dep)] - lon_rad[IATA_index.index(arr)]
    a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[IATA_index.index(dep)]) * math.cos(lat_rad[IATA_index.index(arr)]) * math.sin(delta_lon / 2)**2
    sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = Re * sigma
    return distance

#%%
#Define operating_costs
def costs(dep, arr, k):
    if dep == arr:
        operating_costs = 0
    else:
        CT = Ch[k] * (distance(dep, arr) / speed[k])
        CFuel = cfuel[k] * f * distance(dep, arr) / 1.5
        operating_costs = CT + Cf[k] + CFuel  # Zorg ervoor dat Cf[k] wordt gebruikt
    return operating_costs

#Define revenue 
def revenue(dep, arr, k, t):
    flow = 0
    Yield = 0.26  # Zet een standaardwaarde voor Yield
    timew = int(t / 40)
    if dep == IATA[3]:
        if step_demand[0][IATA_index.index(arr)][t] <= capacity[k]:
            flow = step_demand[0][IATA_index.index(arr)][t]
        else:
            flow = capacity[k]
    
    elif arr == IATA[3]:
        if step_demand[1][IATA_index.index(dep)][t] <= capacity[k]:
            flow = step_demand[1][IATA_index.index(dep)][t]
        else:
            flow = capacity[k]
    flow = flow / 1000
    revenue = Yield * distance(dep, arr) * flow
    return revenue

# Define total profit
M = -10000000000 
def total_profit(dep, arr, k, t):
    if R[k] < distance(dep, arr):
        profit = M
    elif runway_AC[k] > runway_AP[IATA_index.index(dep)] or runway_AC[k] > runway_AP[IATA_index.index(arr)]:
        profit = M
    else:
        profit = revenue(dep, arr, k, t) - costs(dep, arr, k)
    return profit 

#%%
#calculate flight times
flight_time = np.zeros((20,3))
for i in N:
    for k in K:
        if i != 3:
            flight_time[i][k] = np.ceil((distance(IATA[3], IATA[i])/speed[k] + 0.5 + 0.5*TAT[k]/60)*10) 
        else:
            flight_time[i][k] = 0

#%%
# Fleet management and scheduling
available_fleet = {"small": 2, "medium": 2, "large": 1}  # Adjust per assignment
fleet_used = {"small": 0, "medium": 0, "large": 0}

remaining_demand = step_demand.copy()  # Track remaining demand
total_flights = []  # Track all planned flights
# total_profit = 0  # Track total profit across all flights
flight_schedules = {aircraft: [] for aircraft in range(len(speed))}  # Store schedules
stop = False

# Initialize best_aircraft before the loop
final_profit = np.zeros(3)

# Calculate initial profits for all aircraft types (prior to dynamic programming)
for k in K:
    final_profit[k] = 0  # Initializing without costs for better profit tracking.

# Define best_aircraft based on initial profit calculations
best_aircraft = np.argmax(final_profit)

# Initialize arrays for storing action profits
from_hub_action = np.zeros((20, 1200))  # Profit for flights from hub to other airports
to_hub_action = np.zeros((20, 1200))    # Profit for flights from other airports to hub
stay_action = np.zeros((20, 1200))      # Profit for staying at a location (if needed)

#%%
# Initialize data structures to store schedule and profits
aircraft_schedule = {k: [] for k in range(len(speed))}  # Schedule for each aircraft type
aircraft_profits = {k: 0 for k in range(len(speed))}    # Profit for each aircraft type

#%%
# Main loop for scheduling flights
while not stop:
    # Reset state arrays
    state = np.zeros((20, 1200, 3))
    hub_ap = np.zeros((1200, 3), dtype=int)
    hub_ap[-1, :] = hub_index  # Start from the hub

    # Compute profits for flying from/to the hub using the selected best_aircraft
    for t in T:
        for a in N:
            if a == hub_index:  # From hub
                for dest in range(len(IATA)):
                    from_hub_action[dest][t] = total_profit(
                        IATA[hub_index], IATA[dest], best_aircraft, t
                    )
            else:  # To hub
                to_hub_action[a][t] = total_profit(
                    IATA[a], IATA[hub_index], best_aircraft, t
                )

    # Dynamic programming backward pass
    for t in range(1198, -1, -1):
        for a in N:
            if a == hub_index:  # Hub
                profits = np.zeros(len(IATA))
                for dest in range(len(IATA)):
                    if t + flight_time[dest][best_aircraft] < 1200:
                        next_time = int(t + flight_time[dest][best_aircraft])
                        profits[dest] = (
                            from_hub_action[dest][t]
                            + state[dest][next_time][best_aircraft]
                        )
                state[a][t][best_aircraft] = max(profits)
                hub_ap[t][best_aircraft] = np.argmax(profits)
            else:
                if t + flight_time[a][best_aircraft] < 1200:
                    next_time = int(t + flight_time[a][best_aircraft])
                    state[a][t][best_aircraft] = max(
                        to_hub_action[a][t]
                        + state[hub_index][next_time][best_aircraft],
                        stay_action[a][t + 1] + state[a][t + 1][best_aircraft],
                    )

    # Calculate the profit for each aircraft
    for k in K:
        final_profit[k] = state[hub_index][0][k] - 5 * Cl[k]

    max_profit = max(final_profit)
    best_aircraft = np.argmax(final_profit)

    # Print profits of each aircraft type for the current iteration
    print(f"\nIteration results (Profits for each aircraft type):")
    for k in K:
        print(f"  Aircraft Type {k}: Profit = {final_profit[k]:.2f}")

    # Print the selected aircraft type based on the max profit
    print(f"\nSelected Aircraft Type: {best_aircraft} with Profit: {max_profit:.2f}")

    # Only proceed with scheduling if the profit is greater than zero
    if max_profit <= 0:  # No profitable aircraft available
        print("No more profitable flights available.")
        stop = True
        break

    # Update the profit for the best aircraft
    aircraft_profits[best_aircraft] += max_profit

    # Record the schedule for the selected aircraft only if the profit is positive
    schedule = []
    if max_profit > 0:
        for t in range(1200):
            for dest in N:
                if step_demand[0][dest][t] > 0:  # If there is demand
                    schedule.append({
                        "origin": IATA[hub_index],
                        "destination": IATA[dest],
                        "departure_time": t,
                        "flow": step_demand[0][dest][t],
                    })
        aircraft_schedule[best_aircraft].extend(schedule)

        # Update demand
        for flight in schedule:
            destination_index = np.where(IATA == flight["destination"])[0][0]  # Correct way to find index
            step_demand[0][destination_index][flight["departure_time"]] -= flight["flow"]

    # Check stop condition (no more demand or fleet fully utilized)
    if np.sum(step_demand) == 0 or all(
        fleet_used[aircraft_type] >= available_fleet[aircraft_type]
        for aircraft_type in ["small", "medium", "large"]
    ):
        stop = True

# Print the final schedule and profits
print("\nSummary of aircraft profits:")
for k, profit in aircraft_profits.items():
    print(f"Aircraft Type {k+1}: Total Profit = {profit:.2f}")

# # Print flight schedules for aircraft that have been used
# for k, schedule in aircraft_schedule.items():
#     if schedule:
#         print(f"\nAircraft Type {k}:")
#         print(f"  Profit: {aircraft_profits[k]:.2f}")
#         for flight in schedule:
#             print(f"    Flight: {flight['origin']} -> {flight['destination']}, "
#                   f"Departure: {flight['departure_time']}, Flow: {flight['flow']:.2f}")
#     else:
#         print(f"\nAircraft Type {k+1} not used.")

#%%
# # Main loop for scheduling flights
# while not stop:
#     # Reset state arrays
#     state = np.zeros((20, 1200, 3))
#     hub_ap = np.zeros((1200, 3), dtype=int)
#     hub_ap[-1, :] = hub_index  # Start from the hub

#     # Compute profits for flying from/to the hub using the selected best_aircraft
#     for t in T:
#         for a in N:
#             if a == hub_index:  # From hub
#                 for dest in range(len(IATA)):
#                     from_hub_action[dest][t] = total_profit(
#                         IATA[hub_index], IATA[dest], best_aircraft, t
#                     )
#             else:  # To hub
#                 to_hub_action[a][t] = total_profit(
#                     IATA[a], IATA[hub_index], best_aircraft, t
#                 )

#     # Dynamic programming backward pass
#     for t in range(1198, -1, -1):
#         for a in N:
#             if a == hub_index:  # Hub
#                 profits = np.zeros(len(IATA))
#                 for dest in range(len(IATA)):
#                     if t + flight_time[dest][best_aircraft] < 1200:
#                         next_time = int(t + flight_time[dest][best_aircraft])
#                         profits[dest] = (
#                             from_hub_action[dest][t]
#                             + state[dest][next_time][best_aircraft]
#                         )
#                 state[a][t][best_aircraft] = max(profits)
#                 hub_ap[t][best_aircraft] = np.argmax(profits)
#             else:
#                 if t + flight_time[a][best_aircraft] < 1200:
#                     next_time = int(t + flight_time[a][best_aircraft])
#                     state[a][t][best_aircraft] = max(
#                         to_hub_action[a][t]
#                         + state[hub_index][next_time][best_aircraft],
#                         stay_action[a][t + 1] + state[a][t + 1][best_aircraft],
#                     )

#     # Calculate the profit for each aircraft
#     for k in K:
#         final_profit[k] = state[hub_index][0][k] - 5 * Cl[k]

#     max_profit = max(final_profit)
#     best_aircraft = np.argmax(final_profit)

#     # Only proceed with scheduling if the profit is greater than zero
#     if max_profit <= 0:  # No profitable aircraft available
#         print("No more profitable flights available.")
#         stop = True
#         break

#     # Update the profit for the best aircraft
#     aircraft_profits[best_aircraft] += max_profit

#     # Record the schedule for the selected aircraft only if the profit is positive
#     schedule = []
#     if max_profit > 0:
#         for t in range(1200):
#             for dest in N:
#                 if step_demand[0][dest][t] > 0:  # If there is demand
#                     schedule.append({
#                         "origin": IATA[hub_index],
#                         "destination": IATA[dest],
#                         "departure_time": t,
#                         "flow": step_demand[0][dest][t],
#                     })
#         aircraft_schedule[best_aircraft].extend(schedule)

#         # Update demand
#         for flight in schedule:
#             destination_index = np.where(IATA == flight["destination"])[0][0]  # Correct way to find index
#             step_demand[0][destination_index][flight["departure_time"]] -= flight["flow"]

#     # Check stop condition (no more demand or fleet fully utilized)
#     if np.sum(step_demand) == 0 or all(
#         fleet_used[aircraft_type] >= available_fleet[aircraft_type]
#         for aircraft_type in ["small", "medium", "large"]
#     ):
#         stop = True

# # Print the final schedule and profits
# print("\nSummary:")
# for k, profit in aircraft_profits.items():
#     print(f"Aircraft Type {k}: Total Profit = {profit:.2f}")

# # # Print flight schedules for aircraft that have been used
# # for k, schedule in aircraft_schedule.items():
# #     if schedule:
# #         print(f"\nAircraft Type {k}:")
# #         print(f"  Profit: {aircraft_profits[k]:.2f}")
# #         for flight in schedule:
# #             print(f"    Flight: {flight['origin']} -> {flight['destination']}, "
# #                   f"Departure: {flight['departure_time']}, Flow: {flight['flow']:.2f}")
# #     else:
# #         print(f"\nAircraft Type {k+1} not used.")

# #%%
# # Main loop for scheduling flights
# while not stop:
#     # Reset state arrays
#     state = np.zeros((20, 1200, 3))
#     hub_ap = np.zeros((1200, 3), dtype=int)
#     hub_ap[-1, :] = hub_index  # Start from the hub

#     # Compute profits for flying from/to the hub using the selected best_aircraft
#     for t in T:
#         for a in N:
#             if a == hub_index:  # From hub
#                 for dest in range(len(IATA)):
#                     from_hub_action[dest][t] = total_profit(
#                         IATA[hub_index], IATA[dest], best_aircraft, t
#                     )
#             else:  # To hub
#                 to_hub_action[a][t] = total_profit(
#                     IATA[a], IATA[hub_index], best_aircraft, t
#                 )

#     # Dynamic programming backward pass
#     for t in range(1198, -1, -1):
#         for a in N:
#             if a == hub_index:  # Hub
#                 profits = np.zeros(len(IATA))
#                 for dest in range(len(IATA)):
#                     if t + flight_time[dest][best_aircraft] < 1200:
#                         next_time = int(t + flight_time[dest][best_aircraft])
#                         profits[dest] = (
#                             from_hub_action[dest][t]
#                             + state[dest][next_time][best_aircraft]
#                         )
#                 state[a][t][best_aircraft] = max(profits)
#                 hub_ap[t][best_aircraft] = np.argmax(profits)
#             else:
#                 if t + flight_time[a][best_aircraft] < 1200:
#                     next_time = int(t + flight_time[a][best_aircraft])
#                     state[a][t][best_aircraft] = max(
#                         to_hub_action[a][t]
#                         + state[hub_index][next_time][best_aircraft],
#                         stay_action[a][t + 1] + state[a][t + 1][best_aircraft],
#                     )

#     # Determine the most profitable aircraft type
#     for k in K:
#         final_profit[k] = state[hub_index][0][k] - 5 * Cl[k]

#     max_profit = max(final_profit)
#     best_aircraft = np.argmax(final_profit)

#     # Update the profit for the best aircraft
#     aircraft_profits[best_aircraft] += max_profit

#     if max_profit <= 0:  # No profitable aircraft available
#         print("No more profitable flights available.")
#         stop = True
#         break

#     # Record the schedule for the selected aircraft
#     schedule = []
#     for t in range(1200):
#         for dest in N:
#             if step_demand[0][dest][t] > 0:  # If there is demand
#                 schedule.append({
#                     "origin": IATA[hub_index],
#                     "destination": IATA[dest],
#                     "departure_time": t,
#                     "flow": step_demand[0][dest][t],
#                 })
#     aircraft_schedule[best_aircraft].extend(schedule)

#     # Update demand and stop condition
#     # Update demand and stop condition
#     for flight in schedule:
#         destination_index = np.where(IATA == flight["destination"])[0][0]  # Correct way to find index
#         step_demand[0][destination_index][flight["departure_time"]] -= flight["flow"]


#     if np.sum(step_demand) == 0 or all(
#         fleet_used[aircraft_type] >= available_fleet[aircraft_type]
#         for aircraft_type in ["small", "medium", "large"]
#     ):
#         stop = True

# # # Print the final schedule and profits
# # print("\nFlight Scheduling Results:")
# # for k, schedule in aircraft_schedule.items():
# #     print(f"\nAircraft Type {k}:")
# #     if schedule:
# #         print(f"  Profit: {aircraft_profits[k]:.2f}")
# #         for flight in schedule:
# #             print(f"    Flight: {flight['origin']} -> {flight['destination']}, "
# #                   f"Departure: {flight['departure_time']}, Flow: {flight['flow']:.2f}")
# #     else:
# #         print("  Not used.")

# print("\nSummary:")
# for k, profit in aircraft_profits.items():
#     print(f"Aircraft Type {k}: Total Profit = {profit:.2f}")

#%%
# # Main loop for scheduling flights
# while not stop:
# # Reset state arrays
#     state = np.zeros((20, 1200, 3))
#     hub_ap = np.zeros((1200, 3), dtype=int)
#     hub_ap[-1, :] = hub_index  # Start from the hub

#     # Initialize action arrays
#     from_hub_action = np.zeros((20, 1200))
#     to_hub_action = np.zeros((20, 1200))
#     stay_action = np.zeros((20, 1200))
    
#     # Compute profits for flying from/to the hub using the selected best_aircraft
#     for t in T:
#         for a in N:
#             if a == hub_index:  # From hub
#                 for dest in range(len(IATA)):
#                     from_hub_action[dest][t] = total_profit(
#                         IATA[hub_index], IATA[dest], best_aircraft, t
#                     )
#             else:  # To hub
#                 to_hub_action[a][t] = total_profit(
#                     IATA[a], IATA[hub_index], best_aircraft, t
#                 )

#     # Dynamic programming backward pass
#     for t in range(1198, -1, -1):
#         for a in N:
#             if a == hub_index:  # Hub
#                 profits = np.zeros(len(IATA))
#                 for dest in range(len(IATA)):
#                     if t + flight_time[dest][best_aircraft] < 1200:
#                         next_time = int(t + flight_time[dest][best_aircraft])  # Ensure integer index
#                         profits[dest] = (
#                             from_hub_action[dest][t]
#                             + state[dest][next_time][best_aircraft]
#                         )
#                 state[a][t][best_aircraft] = max(profits)
#                 hub_ap[t][best_aircraft] = np.argmax(profits)
#             else:
#                 if t + flight_time[a][best_aircraft] < 1200:
#                     next_time = int(t + flight_time[a][best_aircraft])  # Ensure integer index
#                     state[a][t][best_aircraft] = max(
#                         to_hub_action[a][t]
#                         + state[hub_index][next_time][best_aircraft],
#                         stay_action[a][t + 1] + state[a][t + 1][best_aircraft],
#                     )
                  
#     # Determine the most profitable aircraft type
#     for k in K:
#         final_profit[k] = state[hub_index][0][k] - 5 * Cl[k]

#     max_profit = max(final_profit)
#     best_aircraft = np.argmax(final_profit)

#     if max_profit <= 0:  # No profitable aircraft available
#         print("No more profitable flights available.")
#         stop = True
#         break

#     # Add the selected aircraft type to the fleet usage
#     aircraft_type = ["small", "medium", "large"][best_aircraft]
#     fleet_used[aircraft_type] += 1
    
#     # Add the selected aircraft type to the fleet usage
#     aircraft_type = ["small", "medium", "large"][best_aircraft]
#     fleet_used[aircraft_type] += 1

#     # Update demand and flight schedule
#     for t in T:
#         for dest in N:
#             if t + flight_time[dest][best_aircraft] < 1200:
#                 flow = min(step_demand[0][dest][t], capacity[best_aircraft])
#                 step_demand[0][dest][t] -= flow
#                 total_flights.append({
#                     "origin": IATA[hub_index],
#                     "destination": IATA[dest],
#                     "aircraft": best_aircraft,
#                     "departure_time": t,
#                     "flow": flow
#                 })

#     # Print the remaining demand for tracking
#     print(f"Remaining demand after this scheduling step: {step_demand[0][0][:10]}")

#     # Check stop conditions
#     if np.sum(step_demand) == 0:
#         print("All demand has been fulfilled.")
#         stop = True
#         break
#     if all(fleet_used[aircraft_type] >= available_fleet[aircraft_type] for aircraft_type in ["small", "medium", "large"]):
#         print("Fleet is fully utilized.")
#         stop = True
#         break

# # Final results
# print("Flight scheduling complete.")
# print(f"Total flights planned: {len(total_flights)}")
# print(f"Fleet usage: {fleet_used}")
